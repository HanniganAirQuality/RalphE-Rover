"""Manual control with fresh-input, hold-A arming and bounded stop requests.

Parameters are read at startup. Hardware output defaults off, including device
imports. Steering reference and output-rps limit default to zero (unconfigured).
State topic: 0 disarm/manual, 1 select manual, 2 select Nav2 (blocked),
3 manual deadman heartbeat. Arming requires release, neutral, then a new press.
"""

import asyncio
import math
from time import monotonic

import rclpy
from rclpy.clock import Clock, ClockType
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
        # 10 Hz publisher: three missed periods stop ROS, five expire moteus.
        self.manual_command_timeout_s = self.declare_parameter(
            'manual_command_timeout_s', 0.3).value
        self.motor_watchdog_timeout_s = self.declare_parameter(
            'motor_watchdog_timeout_s', 0.5).value
        self.hardware_io_timeout_s = self.declare_parameter(
            'hardware_io_timeout_s', 0.2).value
        self.state = 1
        self.armed = False
        self._enable_held = self._release_seen = self._neutral_seen = False
        self._last_command_time = self._last_enable_time = None
        self._actuator_fault = False  # Latched until node restart.
        self._motors_stopped = False  # True only after all six stop acknowledgements.
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.nav2_listener_callback, 1)
        self.controller_sub = self.create_subscription(
            Float64MultiArray, 'angle_vel_controller',
            self.controller_angle_vel_listener_callback, 1)
        self.state_sub = self.create_subscription(
            Int16, 'state', self.controller_state_listener_callback, 1)

        self._wheel_circumference_m = math.tau * MEASURED_GEOMETRY.nominal_wheel_radius_m
        self._zero_velocities_mps = {
            name: 0.0 for name in MEASURED_GEOMETRY.wheel_positions_m}
        self.motors = self.arduino = None
        self._loop = asyncio.new_event_loop()
        try:
            for value in (self.manual_command_timeout_s, self.motor_watchdog_timeout_s,
                          self.hardware_io_timeout_s):
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    raise ValueError('Timeouts must be finite and positive seconds.')
            if self.hardware_io_timeout_s >= min(
                    self.manual_command_timeout_s, self.motor_watchdog_timeout_s):
                raise ValueError('I/O timeout must be shorter than both command watchdogs.')
            if self.hardware_output_enabled:
                self._validate_motion_settings()
                # Device imports and initialization stay inside the output gate.
                from ralphee_hardware.ralphee_motor_interface import (
                    init_motors, stop_motors, update_wheel_velocities)
                from ralphee_hardware.ralphee_servo_interface import init_servos, update_servos
                self._init_motors = init_motors
                self._init_servos = init_servos
                self._stop_motors = stop_motors
                self._update_wheel_velocities = update_wheel_velocities
                self._update_servos = update_servos
                self._loop.run_until_complete(self._initialize_hardware())
                self.get_logger().info('Initialized stopped and disarmed; manual control only.')
            else:
                self.get_logger().info(
                    'Hardware output disabled: calculating and logging commands only.')
                if self.steering_reference_angle_rad == 0.0 or self.max_output_rps == 0.0:
                    self.get_logger().warning(
                        'Set steering_reference_angle_rad and max_output_rps explicitly. '
                        'Zero means unconfigured.')
            # A steady timer and monotonic ages also work if ROS simulated time pauses.
            self.command_watchdog = self.create_timer(
                0.05, self._check_command_timeout,
                clock=Clock(clock_type=ClockType.STEADY_TIME))
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
        # Retain each handle immediately so a later startup failure can stop/close it.
        self.motors = await self._init_motors()
        self._motors_stopped = True  # init_motors confirms all six stopped.
        self.arduino = await self._init_servos(write_timeout_s=self.hardware_io_timeout_s)
        await self._update_servos(0.0, self.arduino)

    async def _update_hardware(self, wheel_velocities_mps, steering):
        results = await asyncio.gather(
            self._update_wheel_velocities(
                wheel_velocities_mps, self.motors, max_output_rps=self.max_output_rps,
                watchdog_timeout_s=self.motor_watchdog_timeout_s),
            self._update_servos(steering, self.arduino),
            return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def _dispatch_command(self, wheel_velocities_mps, steering):
        # Validate every final speed before sending either output; never clip.
        for velocity_mps in wheel_velocities_mps.values():
            if (not math.isfinite(velocity_mps)
                    or abs(velocity_mps) / self._wheel_circumference_m > self.max_output_rps):
                raise ValueError('Calculated wheel velocity exceeds max_output_rps or is nonfinite.')
        self.get_logger().info(
            f'Calculated wheel velocities (m/s): {wheel_velocities_mps}; '
            f'Arduino normalized steering: {steering}')
        if self.hardware_output_enabled:
            # Even a partial/failed drive attempt invalidates the stop confirmation.
            self._motors_stopped = False
            self._loop.run_until_complete(asyncio.wait_for(
                self._update_hardware(wheel_velocities_mps, steering),
                timeout=self.hardware_io_timeout_s))

    async def _stop_hardware(self):
        errors = []
        if self.motors is not None and not self._motors_stopped:
            try:
                # Prioritize all motor stops over the synchronous serial write.
                # timeout_s is one deadline for the entire six-motor batch.
                await self._stop_motors(self.motors, timeout_s=self.hardware_io_timeout_s)
                self._motors_stopped = True
            except Exception as error:
                errors.append(f'Motor stop: {error}')
        if self.arduino is not None:
            try:
                await self._update_servos(0.0, self.arduino)
            except Exception as error:
                errors.append(f'Steering center: {error}')
        if errors:
            raise RuntimeError('; '.join(errors))

    def _stop_outputs(self):
        """Stop all six unless already acknowledged stopped; always center steering."""
        if self.hardware_output_enabled:
            try:
                self._loop.run_until_complete(self._stop_hardware())
            except Exception as error:
                self._actuator_fault = True
                self.armed = False
                self.get_logger().error(
                    f'Stop request failed; stopped motion is NOT confirmed: {error}. '
                    'Drive remains disabled until restart.')
        else:
            self.get_logger().debug('Calculated stop: six zero velocities and centered steering.')

    def _request_stop(self, reason):
        self.armed = self._enable_held = self._release_seen = self._neutral_seen = False
        self._last_command_time = self._last_enable_time = None
        self.get_logger().warning(f'{reason} Disarmed; requesting all motors stop and steering center.')
        self._stop_outputs()

    def _check_command_timeout(self):
        if self.armed or self._enable_held:
            now = monotonic()
            if any(stamp is None or now - stamp >= self.manual_command_timeout_s
                   for stamp in (self._last_command_time, self._last_enable_time)):
                self._request_stop('Manual command or deadman heartbeat timed out.')

    def destroy_node(self):
        if not self._loop.is_closed():
            # An interrupted callback must not finish an old drive command while
            # the final stop is being sent.
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            # Shutdown gets one fresh stop attempt even if idle messages were skipped.
            self._motors_stopped = False
            self._request_stop('Controller shutting down.')
            if self.hardware_output_enabled:
                if self.arduino is not None:
                    try:
                        self.arduino.close()
                    except Exception as error:
                        self.get_logger().error(f'Serial close failed: {error}')
            self._loop.close()
        return super().destroy_node()

    def nav2_listener_callback(self, msg: Twist):
        """Nav2 cannot drive in this manual-only stage."""
        if self.state == 2:
            self._request_stop('Nav2 drive output is disabled.')

    def controller_state_listener_callback(self, msg: Int16):
        self._check_command_timeout()
        if msg.data not in (0, 1, 2, 3):
            self._request_stop('Unknown control mode.')
            return
        if msg.data in (0, 1, 2):
            mode = 2 if msg.data == 2 else 1
            if mode != self.state or self.armed or self._enable_held or msg.data == 1:
                self._request_stop('Deadman released or control mode changed.')
            self.state = mode
            # State 1 only selects manual; it cannot enable movement.
            if msg.data == 0:
                self._release_seen = True
            return
        if self.state != 1 or self._actuator_fault:
            return
        rising = not self._enable_held
        self._enable_held = True
        self._last_enable_time = monotonic()
        if rising:
            if (self._release_seen and self._neutral_seen
                    and self._last_command_time is not None
                    and monotonic() - self._last_command_time < self.manual_command_timeout_s):
                self.armed = True
                self.get_logger().info('Manual control armed while neutral; hold A to drive.')
            else:
                self.get_logger().warning('Cannot arm: release A, center both sticks, then press A.')
            self._release_seen = self._neutral_seen = False

    def controller_angle_vel_listener_callback(self, msg: Float64MultiArray):
        self._check_command_timeout()
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
                self._last_command_time = monotonic()
                if not self._enable_held:
                    self._neutral_seen = self._release_seen and steering == 0.0
                self._stop_outputs()
                return
            self._neutral_seen = False
            self._validate_motion_settings()
            # Preserve legacy base output-rps using the wheel-referenced assumption.
            velocity_mps = base_output_rps * self._wheel_circumference_m
            curvature = curvature_from_normalized_steering(
                steering, reference_angle_rad=self.steering_reference_angle_rad,
                geometry=MEASURED_GEOMETRY)
            # This common bound and the Arduino reference require physical verification.
            limits = {
                name: (-self.steering_reference_angle_rad, self.steering_reference_angle_rad)
                for name, (x, _) in MEASURED_GEOMETRY.wheel_positions_m.items() if x != 0}
            wheels = solve_curvature(
                velocity_mps, curvature, geometry=MEASURED_GEOMETRY,
                steering_limits_rad=limits)
            velocities = {name: wheel.linear_velocity_mps for name, wheel in wheels.items()}
            if any(abs(v) / self._wheel_circumference_m > self.max_output_rps
                   for v in velocities.values()):
                raise ValueError('A wheel exceeds max_output_rps.')
        except (TypeError, ValueError, OverflowError) as error:
            self._request_stop(f'Invalid manual command: {error}')
            return
        self._last_command_time = monotonic()
        if not self.armed or not self._enable_held or self._actuator_fault:
            return
        try:
            self._dispatch_command(velocities, steering)
        except Exception as error:
            self._actuator_fault = True
            self.get_logger().error(f'Actuator command failed; restart required: {error}')
            self._request_stop('Actuator communication fault.')


def main(args=None):
    rclpy.init(args=args)
    controller = None
    try:
        controller = RalpheeHardwareController()
        rclpy.spin(controller)
    finally:
        try:
            if controller is not None:
                controller.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
