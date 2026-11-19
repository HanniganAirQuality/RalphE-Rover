"""Manual control checks using real geometry and fake ROS/device interfaces."""

import asyncio
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class FakeNode:
    parameter_values = {}

    def __init__(self, name):
        self.subscriptions = {}
        self.logger = Mock()
        self.destroyed = False

    def declare_parameter(self, name, default):
        return SimpleNamespace(value=self.parameter_values.get(name, default))

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
        self.motors, self.arduino = object(), object()
        self.motor_updates, self.servo_updates = [], []
        self.loops, self.initializations = [], []
        self.fail_next_motor = False

        async def init_motors():
            self.initializations.append('motors')
            self.loops.append(asyncio.get_running_loop())
            return self.motors

        async def init_servos():
            self.initializations.append('servos')
            self.loops.append(asyncio.get_running_loop())
            return self.arduino

        async def update_wheel_velocities(velocities, motors, *, max_output_rps):
            self.loops.append(asyncio.get_running_loop())
            self.motor_updates.append((dict(velocities), motors, max_output_rps))
            if self.fail_next_motor:
                self.fail_next_motor = False
                raise RuntimeError('Fake motor command failure')

        async def update_servos(angle, arduino):
            self.loops.append(asyncio.get_running_loop())
            self.servo_updates.append((angle, arduino))

        self.interfaces = {
            'ralphee_hardware.ralphee_motor_interface': fake_module(
                'ralphee_hardware.ralphee_motor_interface',
                init_motors=init_motors, update_wheel_velocities=update_wheel_velocities),
            'ralphee_hardware.ralphee_servo_interface': fake_module(
                'ralphee_hardware.ralphee_servo_interface',
                init_servos=init_servos, update_servos=update_servos),
        }
        forbidden_ros = Mock(side_effect=AssertionError('Real ROS must not start.'))
        modules = {
            'rclpy': fake_module('rclpy', __path__=[], init=forbidden_ros,
                                 spin=forbidden_ros, shutdown=forbidden_ros),
            'rclpy.node': fake_module('rclpy.node', Node=FakeNode),
            'geometry_msgs': fake_module('geometry_msgs', __path__=[]),
            'geometry_msgs.msg': fake_module('geometry_msgs.msg', Twist=SimpleNamespace),
            'std_msgs': fake_module('std_msgs', __path__=[]),
            'std_msgs.msg': fake_module('std_msgs.msg', Float64MultiArray=SimpleNamespace,
                                        Int16=SimpleNamespace),
            'ralphee_hardware': fake_module('ralphee_hardware', __path__=[]),
        }
        self.blocked_imports = (
            'moteus', 'serial', 'usb', 'inputs', 'numpy', 'ralphee_remote_control',
            *self.interfaces)
        modules.update({name: None for name in self.blocked_imports})
        isolation = patch.dict(sys.modules, modules)
        isolation.start()
        self.addCleanup(isolation.stop)

        self.source = Path(__file__).resolve().parents[1] / 'ralphee_hardware'
        for name, filename in (
                ('ralphee_hardware.motion_geometry', 'motion_geometry.py'),
                ('controller_under_test', 'ralphee_hardware_controller.py')):
            spec = importlib.util.spec_from_file_location(name, self.source / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        self.geometry = sys.modules['ralphee_hardware.motion_geometry']
        self.module = sys.modules['controller_under_test']
        # Synthetic settings only, not measured physical steering or speed limits.
        self.reference = math.pi / 6
        self.limit = 2.0

    def make_node(self, hardware=False, reference=None, limit=None):
        FakeNode.parameter_values = {
            'hardware_output_enabled': hardware,
            'steering_reference_angle_rad': self.reference if reference is None else reference,
            'max_output_rps': self.limit if limit is None else limit,
        }
        with patch.dict(sys.modules, self.interfaces if hardware else {}):
            node = self.module.RalpheeHardwareController()
        self.addCleanup(node.destroy_node)
        return node

    def assert_stop(self):
        speeds, motors, _ = self.motor_updates[-1]
        self.assertEqual(set(speeds), set(self.geometry.MEASURED_GEOMETRY.wheel_positions_m))
        self.assertTrue(all(value == 0.0 for value in speeds.values()))
        self.assertIs(motors, self.motors)
        self.assertEqual(self.servo_updates[-1], (0.0, self.arduino))

    def test_default_disabled_node_never_imports_or_initializes_devices(self):
        FakeNode.parameter_values = {}
        node = self.module.RalpheeHardwareController()
        self.addCleanup(node.destroy_node)
        self.assertFalse(node.hardware_output_enabled)
        self.assertEqual(node.steering_reference_angle_rad, 0.0)
        self.assertEqual(node.max_output_rps, 0.0)
        self.assertEqual(self.initializations, [])
        self.assertIsNone(node.motors)
        for name in self.blocked_imports:
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(name)
        node.controller_angle_vel_listener_callback(SimpleNamespace(data=[0.4, 0.2]))
        self.assertEqual(self.motor_updates, [])
        self.assertEqual(self.servo_updates, [])
        node.logger.warning.assert_called()

    def test_disabled_configured_node_calculates_and_logs_without_devices(self):
        node = self.make_node()
        with patch.object(node, '_dispatch_command', wraps=node._dispatch_command) as dispatch:
            node.controller_angle_vel_listener_callback(SimpleNamespace(data=[0.5, 0.25]))
        speeds, steering = dispatch.call_args.args
        self.assertEqual(steering, 0.25)
        self.assertEqual(len(speeds), 6)
        self.assertNotEqual(speeds['front_left'], speeds['rear_left'])
        self.assertEqual(self.initializations, [])
        self.assertEqual(self.motor_updates, [])
        self.assertEqual(self.servo_updates, [])
        self.assertTrue(any('Calculated wheel velocities' in c.args[0]
                            for c in node.logger.info.call_args_list))

    def test_enabled_fake_initialization_preserves_handles_and_loop(self):
        node = self.make_node(hardware=True)
        self.assertEqual(self.initializations, ['motors', 'servos'])
        self.assertIs(node.motors, self.motors)
        self.assertIs(node.arduino, self.arduino)
        self.assertEqual(node.state, 1)
        self.assertIsNot(node.controller_sub, node.state_sub)
        self.assertFalse(node._loop.is_closed())
        self.assertTrue(all(loop is node._loop for loop in self.loops))

    def test_forward_and_reverse_straight_preserve_legacy_output_rps(self):
        node = self.make_node(hardware=True)
        circumference = math.tau * self.geometry.MEASURED_GEOMETRY.nominal_wheel_radius_m
        for base_rps in (0.75, -0.75, 1.5, -1.5):
            with self.subTest(base_rps=base_rps):
                node.controller_angle_vel_listener_callback(SimpleNamespace(data=[base_rps, 0.0]))
                speeds, motors, limit = self.motor_updates[-1]
                self.assertEqual(limit, self.limit)
                self.assertIs(motors, self.motors)
                for velocity in speeds.values():
                    self.assertAlmostEqual(velocity / circumference, base_rps)
                self.assertEqual(self.servo_updates[-1], (0.0, self.arduino))

    def test_curved_manual_commands_use_real_geometry_and_unchanged_steering(self):
        node = self.make_node(hardware=True)
        radius = self.geometry.MEASURED_GEOMETRY.nominal_wheel_radius_m
        limits = {name: (-self.reference, self.reference)
                  for name in ('front_left', 'front_right', 'rear_left', 'rear_right')}
        for speed, steering in ((0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)):
            with self.subTest(speed=speed, steering=steering):
                curvature = self.geometry.curvature_from_normalized_steering(
                    steering, reference_angle_rad=self.reference,
                    geometry=self.geometry.MEASURED_GEOMETRY)
                expected = self.geometry.solve_curvature(
                    speed * math.tau * radius, curvature,
                    geometry=self.geometry.MEASURED_GEOMETRY, steering_limits_rad=limits)
                node.controller_sub.callback(SimpleNamespace(data=[speed, steering]))
                speeds, _, _ = self.motor_updates[-1]
                self.assertEqual(speeds, {name: wheel.linear_velocity_mps
                                         for name, wheel in expected.items()})
                self.assertEqual(self.servo_updates[-1], (steering, self.arduino))
                self.assertTrue(all(loop is node._loop for loop in self.loops))

    def test_zero_and_invalid_manual_commands_request_stop_and_recover(self):
        node = self.make_node(hardware=True)
        commands = ([0.0, 0.8], [], [0.4], [0.4, 0.2, 0.1],
                    [math.nan, 0.1], [0.4, math.inf], [-math.inf, 0.1],
                    [0.4, math.nan], [0.4, 1.01], [0.4, -1.01],
                    [None, 0.1], [True, 0.1])
        for data in commands:
            with self.subTest(data=data):
                node.controller_angle_vel_listener_callback(SimpleNamespace(data=[0.4, 0.25]))
                node.controller_angle_vel_listener_callback(SimpleNamespace(data=data))
                self.assert_stop()
                node.controller_angle_vel_listener_callback(SimpleNamespace(data=[-0.3, 0.0]))
                self.assertLess(self.motor_updates[-1][0]['front_left'], 0.0)
                self.assertEqual(self.servo_updates[-1], (0.0, self.arduino))

    def test_excessive_outer_wheel_speed_stops_before_nonzero_outputs(self):
        node = self.make_node(hardware=True, limit=1.0)
        node.controller_angle_vel_listener_callback(SimpleNamespace(data=[0.9, 1.0]))
        self.assertEqual(len(self.motor_updates), 1)
        self.assert_stop()
        self.assertEqual(self.servo_updates, [(0.0, self.arduino)])

    def test_invalid_settings_fail_before_hardware_imports(self):
        # Interfaces remain blocked: an early device import would fail this check.
        for reference, limit in ((0.0, 2.0), (math.nan, 2.0), (math.pi / 2, 2.0),
                                 (self.reference, 0.0), (self.reference, math.inf)):
            with self.subTest(reference=reference, limit=limit):
                FakeNode.parameter_values = {
                    'hardware_output_enabled': True,
                    'steering_reference_angle_rad': reference, 'max_output_rps': limit}
                with self.assertRaises(ValueError):
                    self.module.RalpheeHardwareController()
        self.assertEqual(self.initializations, [])

    def test_mode_changes_stop_and_nav2_never_drives(self):
        node = self.make_node(hardware=True)
        command = SimpleNamespace(linear=SimpleNamespace(x=0.6), angular=SimpleNamespace(z=0.2))
        node.controller_angle_vel_listener_callback(SimpleNamespace(data=[0.4, 0.25]))
        count = len(self.motor_updates)
        node.nav2_listener_callback(command)  # Background Nav2 cannot interrupt manual.
        self.assertEqual(len(self.motor_updates), count)
        node.state_sub.callback(SimpleNamespace(data=2))
        self.assertEqual(node.state, 2)
        self.assert_stop()
        count = len(self.motor_updates)
        node.controller_angle_vel_listener_callback(SimpleNamespace(data=[0.4, 0.25]))
        self.assertEqual(len(self.motor_updates), count)
        node.nav2_listener_callback(command)
        self.assert_stop()
        node.state_sub.callback(SimpleNamespace(data=1))
        self.assertEqual(node.state, 1)
        self.assert_stop()
        node.state_sub.callback(SimpleNamespace(data=99))
        self.assert_stop()
        node.controller_angle_vel_listener_callback(SimpleNamespace(data=[0.4, 0.0]))
        self.assertGreater(self.motor_updates[-1][0]['front_left'], 0.0)

    def test_hardware_command_failure_requests_stop_and_recovery(self):
        node = self.make_node(hardware=True)
        self.fail_next_motor = True
        node.controller_angle_vel_listener_callback(SimpleNamespace(data=[0.4, 0.25]))
        node.logger.error.assert_called()
        self.assert_stop()
        node.controller_angle_vel_listener_callback(SimpleNamespace(data=[0.4, 0.0]))
        self.assertGreater(self.motor_updates[-1][0]['front_left'], 0.0)
        self.fail_next_motor = True
        node._request_stop('Fake failed stop')
        self.assertIn('Stop request failed', node.logger.error.call_args.args[0])

    def test_shutdown_requests_stop_and_closes_loop(self):
        node = self.make_node(hardware=True)
        node.controller_angle_vel_listener_callback(SimpleNamespace(data=[0.4, 0.25]))
        node.destroy_node()
        self.assert_stop()
        self.assertTrue(node._loop.is_closed())
        self.assertTrue(node.destroyed)

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
            with patch.multiple(self.interfaces['ralphee_hardware.ralphee_motor_interface'],
                                init_motors=waiting_motor_init):
                with patch.multiple(self.interfaces['ralphee_hardware.ralphee_servo_interface'],
                                    init_servos=failing_servo_init):
                    with self.assertRaisesRegex(RuntimeError, 'Fake initialization failure'):
                        self.make_node(hardware=True)
        self.assertEqual(cancelled, [True])
        self.assertTrue(loop.is_closed())


if __name__ == '__main__':
    unittest.main()
