"""Publish manual commands only while fresh input and the hold-A deadman permit."""

import math

import rclpy
from rclpy.node import Node
from ralphee_remote_control.remote_controller import Controller

from std_msgs.msg import Float64MultiArray, Int16


class ControllerPublisher(Node):

    def __init__(self):
        super().__init__('controller_publisher')
        self.input_timeout_s = self.declare_parameter('input_timeout_s', 0.3).value
        if not math.isfinite(self.input_timeout_s) or self.input_timeout_s <= 0:
            raise ValueError('input_timeout_s must be finite and positive.')
        self.cmd_vel_pub = self.create_publisher(Float64MultiArray, 'angle_vel_controller', 1)
        self.state_pub = self.create_publisher(Int16, 'state', 1)
        self.state = 1  # Existing modes: 1 manual, 2 Nav2 (drive disabled).
        self._deadman_active = False
        self._ready_to_enable = False
        self._previous_enable = True  # Must observe a release, including at startup.
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.controller = Controller()

    def timer_callback(self):
        vel = angle = 0.0
        try:
            vel, angle, enable, x, y, healthy = self.controller.read_command(self.input_timeout_s)
            healthy = (healthy and math.isfinite(vel) and math.isfinite(angle)
                       and -1.0 <= angle <= 1.0)
        except Exception as error:
            self.get_logger().error(f'Cannot confirm gamepad health: {error}')
            healthy = False

        if not healthy:
            self._deadman_active = self._ready_to_enable = False
            self._previous_enable = True
        else:
            mode = 2 if y else 1 if x else self.state
            if mode != self.state:
                self.state = mode
                self._deadman_active = self._ready_to_enable = False
                self._previous_enable = True
            if self.state != 1:
                self._deadman_active = self._ready_to_enable = False
            elif not enable:
                self._deadman_active = False
                self._ready_to_enable = vel == 0.0 and angle == 0.0
            elif not self._previous_enable:
                self._deadman_active = (
                    self._ready_to_enable and vel == 0.0 and angle == 0.0)
                self._ready_to_enable = False
            self._previous_enable = enable

        # Same topics and command format. State 0 = disarmed manual; 3 = held-A
        # heartbeat. State 1 still selects manual without enabling; 2 selects Nav2.
        state = 2 if self.state == 2 else 3 if self._deadman_active else 0
        self._publish(vel if state == 3 else 0.0, angle if state == 3 else 0.0, state)

    def _publish(self, velocity, steering, state):
        self.cmd_vel_pub.publish(Float64MultiArray(data=[velocity, steering]))
        self.state_pub.publish(Int16(data=state))


def main(args=None):
    rclpy.init(args=args)

    controller_publisher = None
    try:
        controller_publisher = ControllerPublisher()
        rclpy.spin(controller_publisher)
    finally:
        if controller_publisher is not None:
            try:
                controller_publisher._publish(0.0, 0.0, 0)
            except Exception as error:
                controller_publisher.get_logger().warning(
                    f'Final zero/disarm publication failed; ROS watchdog remains the backup: {error}')
            finally:
                controller_publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
