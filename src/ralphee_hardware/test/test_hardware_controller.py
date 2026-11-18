"""Stage 1 callback checks using only fake ROS and hardware modules."""

import asyncio
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch


class FakeNode:
    def __init__(self, name):
        self.subscriptions = {}
        self.logger = Mock()
        self.destroyed = False

    def create_subscription(self, message_type, topic, callback, depth):
        subscription = SimpleNamespace(message_type=message_type, callback=callback)
        self.subscriptions[topic] = subscription
        return subscription

    def get_logger(self):
        return self.logger

    def destroy_node(self):
        self.destroyed = True
        return True


def fake_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


class TestHardwareController(unittest.TestCase):
    def setUp(self):
        self.motors = object()
        self.arduino = object()
        self.motor_updates = []
        self.servo_updates = []
        self.loops = []

        async def init_motors():
            self.loops.append(asyncio.get_running_loop())
            return self.motors

        async def init_servos():
            self.loops.append(asyncio.get_running_loop())
            return self.arduino

        async def update_motors(velocity, radius, controller_groups):
            self.loops.append(asyncio.get_running_loop())
            self.motor_updates.append((velocity, radius, controller_groups))

        async def update_servos(angle, arduino):
            self.loops.append(asyncio.get_running_loop())
            self.servo_updates.append((angle, arduino))

        forbidden_ros_call = Mock(side_effect=AssertionError('Real ROS must not start.'))
        modules = {
            'rclpy': fake_module('rclpy', __path__=[], init=forbidden_ros_call,
                                 spin=forbidden_ros_call, shutdown=forbidden_ros_call),
            'rclpy.node': fake_module('rclpy.node', Node=FakeNode),
            'geometry_msgs': fake_module('geometry_msgs', __path__=[]),
            'geometry_msgs.msg': fake_module('geometry_msgs.msg', Twist=SimpleNamespace),
            'std_msgs': fake_module('std_msgs', __path__=[]),
            'std_msgs.msg': fake_module('std_msgs.msg', Float64MultiArray=SimpleNamespace,
                                        Int16=SimpleNamespace),
            'ralphee_hardware': fake_module('ralphee_hardware', __path__=[]),
            'ralphee_hardware.ralphee_motor_interface': fake_module(
                'ralphee_hardware.ralphee_motor_interface',
                init_motors=init_motors, update_motors=update_motors),
            'ralphee_hardware.ralphee_servo_interface': fake_module(
                'ralphee_hardware.ralphee_servo_interface',
                init_servos=init_servos, update_servos=update_servos),
        }
        # None blocks these imports even if a device library is installed locally.
        self.blocked_imports = ('moteus', 'serial', 'usb', 'inputs', 'ralphee_remote_control')
        modules.update({name: None for name in self.blocked_imports})
        isolation = patch.dict(sys.modules, modules)
        isolation.start()
        self.addCleanup(isolation.stop)

        source = Path(__file__).resolve().parents[1] / 'ralphee_hardware'
        spec = importlib.util.spec_from_file_location(
            'stage1_controller_under_test', source / 'ralphee_hardware_controller.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.node = self.module.RalpheeHardwareController()
        self.addCleanup(self.node.destroy_node)

    def test_fake_initialization_and_state_subscription(self):
        self.assertIs(self.node.motors, self.motors)
        self.assertIs(self.node.arduino, self.arduino)
        self.assertFalse(self.loops[0].is_closed())
        self.assertEqual(self.node.state, 1)
        self.assertIsNot(self.node.controller_sub, self.node.state_sub)
        state_callback = self.node.subscriptions['state'].callback
        state_callback(SimpleNamespace(data=2))
        self.assertEqual(self.node.state, 2)
        state_callback(SimpleNamespace(data=99))
        self.assertEqual(self.node.state, 2)
        state_callback(SimpleNamespace(data=1))
        self.assertEqual(self.node.state, 1)
        for name in self.blocked_imports:
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(name)

    def test_consecutive_manual_updates_preserve_handles_and_loop(self):
        callback = self.node.subscriptions['angle_vel_controller'].callback
        with patch.object(self.module, 'get_radius', side_effect=[3.0, 4.0]) as get_radius:
            callback(SimpleNamespace(data=[0.4, 0.25]))
            callback(SimpleNamespace(data=[-0.3, -0.2]))
        self.assertEqual(get_radius.call_args_list, [call(0.25), call(-0.2)])
        self.assertEqual(self.motor_updates, [
            (0.4, 3.0, self.motors), (-0.3, 4.0, self.motors)])
        self.assertEqual(self.servo_updates, [
            (0.25, self.arduino), (-0.2, self.arduino)])
        self.assertIs(self.node.motors, self.motors)
        self.assertIs(self.node.arduino, self.arduino)
        self.assertTrue(all(loop is self.loops[0] for loop in self.loops))

    def test_incomplete_or_nonfinite_manual_commands_do_not_update_hardware(self):
        commands = ([], [0.4], [math.nan, 0.1], [0.4, math.inf],
                    [-math.inf, 0.1], [0.4, math.nan])
        with patch.object(self.module, 'get_radius') as get_radius:
            for data in commands:
                with self.subTest(data=data):
                    self.node.controller_angle_vel_listener_callback(SimpleNamespace(data=data))
            get_radius.assert_not_called()
        self.assertEqual(self.motor_updates, [])
        self.assertEqual(self.servo_updates, [])

    def test_state_gates_commands_and_nav2_dispatch_uses_existing_conversion(self):
        command = SimpleNamespace(linear=SimpleNamespace(x=0.6), angular=SimpleNamespace(z=0.2))
        self.node.nav2_listener_callback(command)
        self.assertEqual(self.motor_updates, [])
        self.node.subscriptions['state'].callback(SimpleNamespace(data=2))
        self.node.controller_angle_vel_listener_callback(SimpleNamespace(data=[0.4, 0.25]))
        self.assertEqual(self.motor_updates, [])
        # Stub mathematics: this checks dispatch without validating steering geometry.
        with patch.object(self.module, 'convert_trans_rot_vel_to_radius_and_inner_angle',
                          return_value=(5.0, 1.2)) as convert:
            with patch.object(self.module, 'inverse_lerp_angle', return_value=0.2) as inverse:
                self.node.nav2_listener_callback(command)
        convert.assert_called_once_with(0.6, 0.2)
        inverse.assert_called_once_with(1.2)
        self.assertEqual(self.motor_updates, [(0.6, 5.0, self.motors)])
        self.assertEqual(self.servo_updates, [(0.2, self.arduino)])
        self.assertIs(self.node.motors, self.motors)
        self.assertIs(self.node.arduino, self.arduino)
        self.assertTrue(all(loop is self.loops[0] for loop in self.loops))

    def test_destroy_closes_loop_and_destroys_fake_node(self):
        self.node.destroy_node()
        self.assertTrue(self.loops[0].is_closed())
        self.assertTrue(self.node.destroyed)

    def test_failed_initialization_cancels_pending_task_and_closes_loop(self):
        cancelled = []

        async def waiting_motor_init():
            try:
                await asyncio.Future()
            finally:
                cancelled.append(True)

        async def failing_servo_init():
            raise RuntimeError('Fake initialization failure')

        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        with patch.object(self.module.asyncio, 'new_event_loop', return_value=loop):
            with patch.multiple(self.module, init_motors=waiting_motor_init,
                                init_servos=failing_servo_init):
                with self.assertRaisesRegex(RuntimeError, 'Fake initialization failure'):
                    self.module.RalpheeHardwareController()
        self.assertEqual(cancelled, [True])
        self.assertTrue(loop.is_closed())


if __name__ == '__main__':
    unittest.main()
