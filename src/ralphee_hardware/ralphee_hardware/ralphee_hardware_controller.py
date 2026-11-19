"""Manual ROS control using the shared wheel geometry.

Parameters are read at startup. Hardware output defaults off; no device module
is imported in that mode. Reference angle and output-rps limit default to zero
(unconfigured). Set explicit values for calculation-only use, and verify physical
calibration, stale-input stopping and the motor watchdog before enabling output.
"""

import asyncio
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Int16

from ralphee_hardware.motion_geometry import (
    MEASURED_GEOMETRY,
    curvature_from_normalized_steering,
    solve_curvature,
)


class RalpheeHardwareController(Node):
    """Keep manual [base output-rps, normalized steering] commands separate from I/O."""

    def __init__(self):
        super().__init__('ralphee_hardware_controller')
        self.hardware_output_enabled = self.declare_parameter(
            'hardware_output_enabled', False).value
        self.steering_reference_angle_rad = self.declare_parameter(
            'steering_reference_angle_rad', 0.0).value
        self.max_output_rps = self.declare_parameter('max_output_rps', 0.0).value
        self.state = 1
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.nav2_listener_callback, 10)
        self.controller_sub = self.create_subscription(
            Float64MultiArray, 'angle_vel_controller',
            self.controller_angle_vel_listener_callback, 10)
        self.state_sub = self.create_subscription(
            Int16, 'state', self.controller_state_listener_callback, 10)

        self._wheel_circumference_m = math.tau * MEASURED_GEOMETRY.nominal_wheel_radius_m
        self._zero_velocities_mps = {
            name: 0.0 for name in MEASURED_GEOMETRY.wheel_positions_m}
        self.motors = self.arduino = None
        self._loop = asyncio.new_event_loop()
        try:
            if self.hardware_output_enabled:
                self._validate_motion_settings()
                # These imports and all device initialization are inside the output gate.
                from ralphee_hardware.ralphee_motor_interface import (
                    init_motors, update_wheel_velocities)
                from ralphee_hardware.ralphee_servo_interface import init_servos, update_servos
                self._init_motors = init_motors
                self._init_servos = init_servos
                self._update_wheel_velocities = update_wheel_velocities
                self._update_servos = update_servos
                self.motors, self.arduino = self._loop.run_until_complete(
                    self._initialize_hardware())
                self.get_logger().info('Initialized hardware; manual control only.')
            else:
                self.get_logger().info(
                    'Hardware output disabled: calculating and logging commands only.')
                if self.steering_reference_angle_rad == 0.0 or self.max_output_rps == 0.0:
                    self.get_logger().warning(
                        'Set steering_reference_angle_rad and max_output_rps explicitly '
                        'to calculate nonzero manual commands. Zero means unconfigured.')
        except BaseException:
            self.destroy_node()
            raise

    def _validate_motion_settings(self):
        if (not math.isfinite(self.steering_reference_angle_rad)
                or not 0 < self.steering_reference_angle_rad < math.pi / 2):
            raise ValueError('steering_reference_angle_rad must be finite and between 0 and pi/2.')
        if not math.isfinite(self.max_output_rps) or self.max_output_rps <= 0:
            raise ValueError('max_output_rps must be finite and positive.')

    async def _initialize_hardware(self):
        return await asyncio.gather(self._init_motors(), self._init_servos())

    async def _update_hardware(self, wheel_velocities_mps, steering):
        # Finish both operations before handling an error or issuing a later stop.
        results = await asyncio.gather(
            self._update_wheel_velocities(
                wheel_velocities_mps, self.motors, max_output_rps=self.max_output_rps),
            self._update_servos(steering, self.arduino),
            return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def _dispatch_command(self, wheel_velocities_mps, steering):
        # Check final wheel speeds in calculation mode too, before either output.
        for velocity_mps in wheel_velocities_mps.values():
            if (not math.isfinite(velocity_mps)
                    or (velocity_mps != 0.0
                        and abs(velocity_mps) / self._wheel_circumference_m > self.max_output_rps)):
                raise ValueError('Calculated wheel velocity exceeds max_output_rps or is nonfinite.')
        self.get_logger().info(
            f'Calculated wheel velocities (m/s): {wheel_velocities_mps}; '
            f'Arduino normalized steering: {steering}')
        if self.hardware_output_enabled:
            self._loop.run_until_complete(
                self._update_hardware(wheel_velocities_mps, steering))

    def _request_stop(self, reason):
        self.get_logger().warning(f'{reason} Requesting six zero velocities and centered steering.')
        try:
            self._dispatch_command(self._zero_velocities_mps, 0.0)
        except Exception as error:
            # A failed request is not confirmation that the motors have stopped.
            self.get_logger().error(f'Stop request failed: {error}')

    def destroy_node(self):
        if not self._loop.is_closed():
            if self.hardware_output_enabled and self.motors is not None:
                self._request_stop('Controller shutting down.')
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()
        return super().destroy_node()

    def nav2_listener_callback(self, msg: Twist):
        """Nav2 cannot drive in this manual-only stage."""
        if self.state == 2:
            self._request_stop('Nav2 drive output is disabled.')

    def controller_state_listener_callback(self, msg: Int16):
        if msg.data not in (1, 2):
            self._request_stop('Unknown control mode.')
            return
        if msg.data != self.state:
            self.state = msg.data
            self._request_stop('Control mode changed.')

    def controller_angle_vel_listener_callback(self, msg: Float64MultiArray):
        if self.state != 1:
            return
        try:
            if len(msg.data) != 2:
                raise ValueError('Expected exactly [speed, steering].')
            base_output_rps, steering = msg.data
            if (isinstance(base_output_rps, bool) or isinstance(steering, bool)
                    or not math.isfinite(base_output_rps) or not math.isfinite(steering)):
                raise ValueError('Speed and steering must be finite numbers.')
            if not -1.0 <= steering <= 1.0:
                raise ValueError('Normalized steering must lie in [-1, 1].')
            if base_output_rps == 0.0:
                self._request_stop('Zero manual speed.')
                return
            self._validate_motion_settings()

            # The legacy gamepad speed is base moteus output rps, not m/s.
            # Under the documented wheel-referenced output assumption, this
            # cancels the motor interface's m/s-to-rps conversion on a straight.
            velocity_mps = base_output_rps * self._wheel_circumference_m
            curvature = curvature_from_normalized_steering(
                steering, reference_angle_rad=self.steering_reference_angle_rad,
                geometry=MEASURED_GEOMETRY)
            # A conservative common wheel-heading bound, NOT four assumed physical
            # limits: the chosen reference must be safe for every corner and match Arduino.
            limits = {
                name: (-self.steering_reference_angle_rad, self.steering_reference_angle_rad)
                for name, (x, _) in MEASURED_GEOMETRY.wheel_positions_m.items() if x != 0}
            wheels = solve_curvature(
                velocity_mps, curvature, geometry=MEASURED_GEOMETRY,
                steering_limits_rad=limits)
            velocities = {name: wheel.linear_velocity_mps for name, wheel in wheels.items()}
            self._dispatch_command(velocities, steering)
        except (TypeError, ValueError, OverflowError) as error:
            self._request_stop(f'Invalid manual command: {error}')
        except Exception as error:
            self.get_logger().error(f'Hardware command failed: {error}')
            self._request_stop('Hardware command failure.')


def main(args=None):
    rclpy.init(args=args)
    controller = None
    try:
        controller = RalpheeHardwareController()
        rclpy.spin(controller)
    finally:
        if controller is not None:
            controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
