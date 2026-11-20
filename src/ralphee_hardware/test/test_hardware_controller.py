"""Manual control checks using real geometry and fake ROS/device interfaces."""

import asyncio
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch, mock_open


class FakeNode:
    parameter_values = {}

    def __init__(self, name):
        self.subscriptions = {}
        self.publishers = {}
        self.logger = Mock()
        self.destroyed = False

    def declare_parameter(self, name, default):
        return SimpleNamespace(value=self.parameter_values.get(name, default))

    def create_subscription(self, message_type, topic, callback, depth):
        subscription = SimpleNamespace(message_type=message_type, callback=callback)
        self.subscriptions[topic] = subscription
        return subscription

    def create_timer(self, period, callback, *, clock=None):
        return SimpleNamespace(period=period, callback=callback, clock=clock)

    def create_publisher(self, message_type, topic, depth):
        publisher = Mock()
        self.publishers[topic] = publisher
        return publisher

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
        self.motors, self.arduino = object(), SimpleNamespace(close=Mock())
        self.motor_updates, self.servo_updates = [], []
        self.loops, self.initializations = [], []
        self.fail_next_motor = self.fail_next_stop = False
        self.motor_stops, self.watchdogs = [], []
        self.now = 100.0

        async def init_motors():
            self.initializations.append('motors')
            self.motor_stops.append(self.motors)  # init_motors acknowledges all six stops.
            self.loops.append(asyncio.get_running_loop())
            return self.motors

        async def init_servos(*, write_timeout_s):
            self.initializations.append('servos')
            self.loops.append(asyncio.get_running_loop())
            return self.arduino

        async def stop_motors(motors, *, timeout_s):
            self.motor_stops.append(motors)
            if self.fail_next_stop:
                self.fail_next_stop = False
                raise RuntimeError('Fake failed stop')

        async def update_wheel_velocities(
                velocities, motors, *, max_output_rps, watchdog_timeout_s):
            self.loops.append(asyncio.get_running_loop())
            self.motor_updates.append((dict(velocities), motors, max_output_rps))
            self.watchdogs.append(watchdog_timeout_s)
            if self.fail_next_motor:
                self.fail_next_motor = False
                raise RuntimeError('Fake motor command failure')

        async def update_servos(angle, arduino):
            self.loops.append(asyncio.get_running_loop())
            self.servo_updates.append((angle, arduino))

        self.interfaces = {
            'ralphee_hardware.ralphee_motor_interface': fake_module(
                'ralphee_hardware.ralphee_motor_interface',
                init_motors=init_motors, stop_motors=stop_motors,
                update_wheel_velocities=update_wheel_velocities),
            'ralphee_hardware.ralphee_servo_interface': fake_module(
                'ralphee_hardware.ralphee_servo_interface',
                init_servos=init_servos, update_servos=update_servos),
        }
        forbidden_ros = Mock(side_effect=AssertionError('Real ROS must not start.'))
        modules = {
            'rclpy': fake_module('rclpy', __path__=[], init=forbidden_ros,
                                 spin=forbidden_ros, shutdown=forbidden_ros),
            'rclpy.node': fake_module('rclpy.node', Node=FakeNode),
            'rclpy.clock': fake_module('rclpy.clock',
                                      Clock=lambda **kwargs: SimpleNamespace(**kwargs),
                                      ClockType=SimpleNamespace(STEADY_TIME='steady')),
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
        clock_patch = patch.object(self.module, 'monotonic', lambda: self.now)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        # Synthetic settings only, not measured physical steering or speed limits.
        self.reference = math.pi / 6
        self.limit = 2.0

    def load_motor_interface(self):
        sdk = fake_module(
            'moteus', Controller=object,
            Register=SimpleNamespace(MODE='mode', FAULT='fault'),
            Mode=SimpleNamespace(STOPPED=0, FAULT=1, POSITION=10, TIMEOUT=11))
        spec = importlib.util.spec_from_file_location(
            'motor_interface_under_test', self.source / 'ralphee_motor_interface.py')
        interface = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
                'moteus': sdk, 'usb': fake_module('usb'),
                'numpy': fake_module('numpy', ndarray=list, array=lambda values: values)}):
            spec.loader.exec_module(interface)
        return interface, sdk

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

    def arm(self, node, *, expected=True):
        node.state_sub.callback(SimpleNamespace(data=0))
        node.controller_sub.callback(SimpleNamespace(data=[0.0, 0.0]))
        node.state_sub.callback(SimpleNamespace(data=3))
        self.assertEqual(node.armed, expected)

    def assert_stop(self):
        self.assertIs(self.motor_stops[-1], self.motors)
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
        self.arm(node)
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
        self.arm(node)
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
        self.arm(node)
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

    def test_zero_and_invalid_commands_stop_and_invalid_requires_rearm(self):
        node = self.make_node(hardware=True)
        self.arm(node)
        node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.25]))
        node.controller_sub.callback(SimpleNamespace(data=[0.0, 0.8]))
        self.assert_stop()
        self.assertTrue(node.armed)  # Zero speed stops; held A still permits later motion.
        commands = ([], [0.4], [0.4, 0.2, 0.1],
                    [math.nan, 0.1], [0.4, math.inf], [-math.inf, 0.1],
                    [0.4, math.nan], [0.4, 1.01], [0.4, -1.01],
                    [None, 0.1], [True, 0.1])
        for data in commands:
            with self.subTest(data=data):
                self.arm(node)
                node.controller_sub.callback(SimpleNamespace(data=data))
                self.assert_stop()
                self.assertFalse(node.armed)
                count = len(self.motor_updates)
                node.controller_sub.callback(SimpleNamespace(data=[-0.3, 0.0]))
                self.assertEqual(len(self.motor_updates), count)
                self.arm(node)
                node.controller_sub.callback(SimpleNamespace(data=[-0.3, 0.0]))
                self.assertLess(self.motor_updates[-1][0]['front_left'], 0.0)

    def test_excessive_outer_wheel_speed_stops_before_nonzero_outputs(self):
        node = self.make_node(hardware=True, limit=1.0)
        self.arm(node)
        node.controller_angle_vel_listener_callback(SimpleNamespace(data=[0.9, 1.0]))
        self.assertEqual(self.motor_updates, [])
        self.assert_stop()
        self.assertFalse(node.armed)
        self.assertTrue(all(angle == 0.0 for angle, _ in self.servo_updates))

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
        self.arm(node)
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
        count = len(self.motor_updates)
        node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
        self.assertEqual(len(self.motor_updates), count)
        self.arm(node)
        node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
        self.assertGreater(self.motor_updates[-1][0]['front_left'], 0.0)

    def test_hardware_command_failure_requests_stop_and_latches_fault(self):
        node = self.make_node(hardware=True)
        self.arm(node)
        self.fail_next_motor = True
        node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.25]))
        node.logger.error.assert_called()
        self.assert_stop()
        self.assertFalse(node.armed)
        self.assertTrue(node._actuator_fault)
        count = len(self.motor_updates)
        self.arm(node, expected=False)
        node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
        self.assertEqual(len(self.motor_updates), count)


    def test_startup_held_enable_or_non_neutral_cannot_arm(self):
        node = self.make_node(hardware=True)
        self.assertFalse(node.armed)
        node.state_sub.callback(SimpleNamespace(data=3))
        node.controller_sub.callback(SimpleNamespace(data=[0.0, 0.0]))
        node.state_sub.callback(SimpleNamespace(data=3))
        node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
        self.assertEqual(self.motor_updates, [])
        node.state_sub.callback(SimpleNamespace(data=0))
        node.controller_sub.callback(SimpleNamespace(data=[0.0, 0.5]))
        node.state_sub.callback(SimpleNamespace(data=3))
        self.assertFalse(node.armed)
        self.arm(node)
        node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
        self.assertEqual(len(self.motor_updates), 1)
        node.state_sub.callback(SimpleNamespace(data=0))
        self.assert_stop()
        self.assertFalse(node.armed)

    def test_command_watchdog_disarms_and_returning_messages_cannot_resume(self):
        node = self.make_node(hardware=True)
        self.arm(node)
        node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
        self.now += 0.31
        node.command_watchdog.callback()
        self.assert_stop()
        self.assertFalse(node.armed)
        count = len(self.motor_updates)
        for data in ([0.4, 0.0], [0.0, 0.0], [0.4, 0.0]):
            node.controller_sub.callback(SimpleNamespace(data=data))
            node.state_sub.callback(SimpleNamespace(data=3))
        self.assertEqual(len(self.motor_updates), count)
        self.arm(node)
        node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
        self.assertEqual(len(self.motor_updates), count + 1)
        self.assertEqual(node.command_watchdog.clock.clock_type, 'steady')
        self.assertTrue(all(value == 0.5 for value in self.watchdogs))

    def test_command_and_deadman_heartbeats_expire_independently(self):
        for missing in ('command', 'deadman'):
            with self.subTest(missing=missing):
                node = self.make_node(hardware=True)
                self.arm(node)
                self.now += 0.15
                if missing == 'command':
                    node.state_sub.callback(SimpleNamespace(data=3))
                else:
                    node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
                self.now += 0.16
                # Receipt handlers must detect timeout before processing a late command,
                # even if the watchdog timer callback has not run.
                if missing == 'command':
                    node.state_sub.callback(SimpleNamespace(data=3))
                else:
                    node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
                self.assertFalse(node.armed)
                self.assert_stop()

    def test_stalled_actuator_update_times_out_and_latches_fault(self):
        node = self.make_node(hardware=True)
        self.arm(node)

        async def stalled(*args, **kwargs):
            await asyncio.Future()

        with patch.object(node, '_update_wheel_velocities', stalled):
            node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
        self.assertFalse(node.armed)
        self.assertTrue(node._actuator_fault)
        self.assert_stop()
        self.assertEqual(asyncio.all_tasks(node._loop), set())

    def test_shutdown_requests_stop_and_closes_loop(self):
        node = self.make_node(hardware=True)
        self.arm(node)
        node.controller_angle_vel_listener_callback(SimpleNamespace(data=[0.4, 0.25]))
        node.destroy_node()
        self.assert_stop()
        self.assertTrue(node._loop.is_closed())
        self.assertTrue(node.destroyed)

    def test_servo_initialization_failure_stops_already_initialized_motors(self):
        async def failing_servo_init(**kwargs):
            raise RuntimeError('Fake initialization failure')

        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        with patch.object(self.module.asyncio, 'new_event_loop', return_value=loop):
            with patch.multiple(self.interfaces['ralphee_hardware.ralphee_servo_interface'],
                                init_servos=failing_servo_init):
                with self.assertRaisesRegex(RuntimeError, 'Fake initialization failure'):
                    self.make_node(hardware=True)
        self.assertIs(self.motor_stops[-1], self.motors)
        self.assertEqual(self.motor_updates, [])
        self.assertTrue(loop.is_closed())

    def test_reader_polls_stationary_state_and_expires_stalls_or_disconnects(self):
        source = self.source.parents[1] / 'ralphee_remote_control' / 'ralphee_remote_control'
        fake_inputs = fake_module(
            'inputs', get_gamepad=Mock(side_effect=AssertionError('Blocking reader forbidden')),
            DeviceManager=lambda: SimpleNamespace(gamepads=[
                SimpleNamespace(get_char_device_path=lambda: '/mock/gamepad')]))
        spec = importlib.util.spec_from_file_location(
            'reader_under_test', source / 'remote_controller.py')
        reader_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'inputs': fake_inputs}):
            spec.loader.exec_module(reader_module)
        with patch.object(reader_module.threading, 'Thread'):
            reader = reader_module.Controller()
        reader._monitor_thread.is_alive.return_value = True

        class EndRead(BaseException):
            pass

        # Fixed state with no change events. These are synthetic Xbox-format values.
        def state_ioctl(device, request, buffer):
            if request == 0x80604518:
                keys = bytearray(96)
                keys[0x130 // 8] |= 1 << (0x130 % 8)  # Hold A.
                return bytes(keys)
            self.assertEqual(len(buffer), 24)
            value = {0x80184541: 16000, 0x80184543: 8000,
                     0x80184542: 0, 0x80184545: 0}[request]
            minimum, maximum = (-32768, 32767) if request in (
                0x80184541, 0x80184543) else (0, 255)
            return reader_module.struct.pack('=6i', value, minimum, maximum, 0, 0, 0)

        snapshots = []

        def ten_polls(seconds):
            self.assertEqual(seconds, 0.05)
            snapshots.append(reader.read_command())
            self.now += seconds
            if len(snapshots) == 10:
                raise EndRead()

        with patch.object(reader_module, 'monotonic', lambda: self.now), \
                patch.object(reader_module, 'ioctl', side_effect=state_ioctl), \
                patch('builtins.open', mock_open()), \
                patch.object(reader_module, 'sleep', side_effect=ten_polls):
            with self.assertRaises(EndRead):
                reader.update()
            # More than the 0.3 s freshness window, with identical held controls.
            self.assertFalse(snapshots[0][-1])  # Startup loss must be delivered once.
            self.assertTrue(all(sample[-1] for sample in snapshots[1:]))
            self.assertTrue(all(sample[0] > 0.0 and sample[2] for sample in snapshots[1:]))
            self.assertEqual(snapshots[1:], [snapshots[1]] * 9)
            fake_inputs.get_gamepad.assert_not_called()

            # A stalled reader cannot keep its previous nonzero snapshot healthy.
            self.now += 0.31
            self.assertEqual(reader.read_command(),
                             (0.0, 0.0, False, False, False, False))

        def end_after_poll(seconds):
            raise EndRead()

        with patch.object(reader_module, 'monotonic', lambda: self.now), \
                patch.object(reader_module, 'ioctl', side_effect=state_ioctl), \
                patch('builtins.open', mock_open()), \
                patch.object(reader_module, 'sleep', side_effect=end_after_poll):
            with self.assertRaises(EndRead):
                reader.update()
            self.assertTrue(reader.read_command()[-1])
            # Recovery between ROS ticks must still announce an input loss.
            self.now += 0.31
            with self.assertRaises(EndRead):
                reader.update()
            self.assertFalse(reader.read_command()[-1])
            self.assertTrue(reader.read_command()[-1])
            reader._monitor_thread.is_alive.return_value = False
            self.assertFalse(reader.read_command()[-1])
            reader._monitor_thread.is_alive.return_value = True

        def slow_ioctl(device, request, buffer):
            result = state_ioctl(device, request, buffer)
            if request == 0x80604518:
                self.now += 0.31
            return result

        for backend in (slow_ioctl, Mock(side_effect=OSError('Controller disconnected'))):
            with patch.object(reader_module, 'monotonic', lambda: self.now), \
                    patch.object(reader_module, 'ioctl', side_effect=backend), \
                    patch('builtins.open', mock_open()), \
                    patch.object(reader_module, 'sleep', side_effect=end_after_poll):
                with self.assertRaises(EndRead):
                    reader.update()
                self.assertIsNone(reader.last_update_time)
                self.assertEqual(reader.read_command(),
                                 (0.0, 0.0, False, False, False, False))

    def test_confirmed_stop_is_reused_but_drive_and_failed_stop_invalidate_it(self):
        node = self.make_node(hardware=True)
        initial_stops = len(self.motor_stops)
        for _ in range(5):
            node.controller_sub.callback(SimpleNamespace(data=[0.0, 0.5]))
        self.assertEqual(len(self.motor_stops), initial_stops)
        self.assertTrue(node._motors_stopped)

        self.arm(node)
        node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
        self.assertFalse(node._motors_stopped)
        node.controller_sub.callback(SimpleNamespace(data=[0.0, 0.0]))
        self.assertEqual(len(self.motor_stops), initial_stops + 1)
        self.assertTrue(node._motors_stopped)
        for _ in range(5):
            node.controller_sub.callback(SimpleNamespace(data=[0.0, 0.0]))
        self.assertEqual(len(self.motor_stops), initial_stops + 1)

        node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
        self.fail_next_stop = True
        node.controller_sub.callback(SimpleNamespace(data=[0.0, 0.0]))
        self.assertFalse(node._motors_stopped)
        self.assertFalse(node.armed)
        self.assertTrue(node._actuator_fault)
        self.assertIn('Stop request failed', node.logger.error.call_args.args[0])
        failed_count = len(self.motor_stops)
        node.controller_sub.callback(SimpleNamespace(data=[0.0, 0.0]))
        self.assertEqual(len(self.motor_stops), failed_count + 1)  # Retry unconfirmed stop.
        self.assertTrue(node._motors_stopped)
        self.arm(node, expected=False)  # Successful stop does not clear the fault latch.
        count = len(self.motor_stops)
        node.destroy_node()
        self.assertEqual(len(self.motor_stops), count + 1)  # Fresh final stop.

    def test_motor_stop_precedes_serial_write_and_still_centers_after_failure(self):
        node = self.make_node(hardware=True)
        order = []

        async def stop_motors(*args, **kwargs):
            order.append('motor batch sent')
            await asyncio.sleep(0)
            order.append('motor batch acknowledged')

        async def center(*args):
            order.append('steering center')

        async def failed_stop(*args, **kwargs):
            order.append('motor batch failed')
            raise OSError('Fake bus error')

        node._motors_stopped = False
        with patch.object(node, '_stop_motors', stop_motors), \
                patch.object(node, '_update_servos', center):
            node._stop_outputs()
        self.assertEqual(order, ['motor batch sent', 'motor batch acknowledged', 'steering center'])
        self.assertTrue(node._motors_stopped)
        order.clear()
        node._motors_stopped = False
        with patch.object(node, '_stop_motors', failed_stop), \
                patch.object(node, '_update_servos', center):
            node._stop_outputs()
        self.assertEqual(order, ['motor batch failed', 'steering center'])
        self.assertFalse(node._motors_stopped)
        self.assertTrue(node._actuator_fault)

    def test_motor_stop_batches_then_independently_verifies_with_one_deadline(self):
        interface, sdk = self.load_motor_interface()
        ids = [2, 5, 3, 0, 4, 1]
        controllers = [SimpleNamespace(
            id=motor_id,
            make_stop=Mock(return_value=SimpleNamespace(kind='stop', destination=motor_id)),
            make_query=Mock(return_value=SimpleNamespace(kind='query', destination=motor_id)))
            for motor_id in ids]
        groups = [controllers[:3], controllers[3:]]
        cycles = []
        query_round = 0
        scenario = 'delayed'

        async def cycle(commands):
            nonlocal query_round
            kind = commands[0].kind
            requested = [command.destination for command in commands]
            cycles.append((kind, requested))
            if kind == 'stop':
                return []  # Write-only stop batch, not a post-command status read.
            query_round += 1
            if scenario == 'stalled':
                await asyncio.Future()
            results = []
            for motor_id in requested:
                if scenario == 'missing' and motor_id == 2:
                    continue
                mode = 10 if scenario == 'never_stopped' else 0
                if scenario == 'delayed' and query_round == 1 and motor_id == 2:
                    mode = 10  # Stop staged, previous POSITION still visible.
                fault = 7 if scenario == 'fault' and motor_id == 5 else 0
                results.append(SimpleNamespace(id=motor_id, values={'mode': mode, 'fault': fault}))
            return results

        sdk.get_singleton_transport = lambda: SimpleNamespace(cycle=cycle)

        async def check():
            nonlocal scenario
            await interface.stop_motors(groups, timeout_s=0.1)
            self.assertEqual(cycles, [('stop', ids), ('query', ids), ('query', [2])])
            for controller in controllers:
                controller.make_stop.assert_called_once_with(query=False)

            for scenario, message in (
                    ('missing', 'Motor 2'), ('fault', 'Motor 5'),
                    ('never_stopped', 'state verification timed out'),
                    ('stalled', 'state verification timed out')):
                with self.subTest(scenario=scenario):
                    cycles.clear()
                    with patch.object(interface.asyncio, 'timeout',
                                      wraps=asyncio.timeout) as deadline:
                        with self.assertRaisesRegex(RuntimeError, message):
                            await interface.stop_motors(groups, timeout_s=0.1)
                        self.assertEqual(deadline.call_count, 1)
                        self.assertEqual(deadline.call_args.args, (0.1,))
                    self.assertEqual(cycles[:2], [('stop', ids), ('query', ids)])
                    self.assertTrue(all(kind == 'query' for kind, _ in cycles[1:]))
                    self.assertEqual(asyncio.all_tasks(), {asyncio.current_task()})

        asyncio.run(check())

    def test_motor_move_verifies_staged_modes_and_keeps_steady_updates_efficient(self):
        interface, sdk = self.load_motor_interface()
        ids = [2, 5, 3, 0, 4, 1]
        sent = []
        cycles = []
        initial_modes = dict.fromkeys(ids, 0)
        query_modes = [0, 2, 3, 4, 10]  # Previous mode, then firmware enabling/calibration.
        scenario = 'healthy'
        query_round = 0
        stop_sent = False

        def reply(motor_id, mode=10, fault=0):
            return SimpleNamespace(id=motor_id, values={'mode': mode, 'fault': fault})

        def controller(motor_id):
            async def set_position(**kwargs):
                sent.append((motor_id, kwargs))
                if scenario == 'missing_immediate' and motor_id == 2:
                    return None
                fault = 9 if scenario == 'immediate_fault' and motor_id == 2 else 0
                return reply(motor_id, initial_modes[motor_id], fault)

            return SimpleNamespace(
                id=motor_id, set_position=set_position,
                make_stop=lambda **kwargs: SimpleNamespace(kind='stop', destination=motor_id),
                make_query=lambda: SimpleNamespace(kind='query', destination=motor_id))

        motors = [controller(motor_id) for motor_id in ids]
        groups = [motors[:3], motors[3:]]
        speeds = {name: (index + 1) * 0.05
                  for index, name in enumerate(self.geometry.MEASURED_GEOMETRY.wheel_positions_m)}

        async def cycle(commands):
            nonlocal query_round, stop_sent
            kind = commands[0].kind
            requested = [command.destination for command in commands]
            cycles.append((kind, requested))
            if kind == 'stop':
                stop_sent = True
                return []
            # The first follow-up read must come after ALL six movement writes.
            self.assertEqual(len(sent), 6)
            query_round += 1
            if stop_sent:
                return [reply(motor_id, 0) for motor_id in requested]
            if scenario == 'stalled_query':
                await asyncio.Future()
            mode = query_modes[min(query_round - 1, len(query_modes) - 1)]
            results = [reply(motor_id, mode) for motor_id in requested]
            if scenario == 'missing_query':
                return results[1:]
            if scenario == 'query_fault':
                results[0].values['fault'] = 7
            if scenario == 'missing_fault':
                del results[0].values['fault']
            return results

        sdk.get_singleton_transport = lambda: SimpleNamespace(cycle=cycle)

        async def check():
            nonlocal scenario, query_round, query_modes
            await interface.update_wheel_velocities(
                speeds, groups, max_output_rps=2.0, watchdog_timeout_s=0.4, timeout_s=0.2)
            self.assertEqual([motor_id for motor_id, _ in sent], ids)
            self.assertEqual(cycles, [('query', ids)] * 5)
            wheel_order = ('front_left', 'middle_left', 'rear_left',
                          'front_right', 'middle_right', 'rear_right')
            for index, ((_, command), wheel) in enumerate(zip(sent, wheel_order)):
                expected = speeds[wheel] / (math.tau * 0.125) * (1 if index < 3 else -1)
                self.assertAlmostEqual(command['velocity'], expected)
                self.assertTrue(math.isnan(command['position']))
                self.assertTrue(command['query'])
                self.assertEqual(command['watchdog_timeout'], 0.4)

            sent.clear()
            cycles.clear()
            initial_modes.update(dict.fromkeys(ids, 10))
            await interface.update_wheel_velocities(speeds, groups, max_output_rps=2.0)
            self.assertEqual(len(sent), 6)
            self.assertEqual(cycles, [])  # No added reads or sleeps in steady POSITION mode.

            # Only a motor with an old/transitioning mode needs follow-up reads.
            sent.clear()
            initial_modes[2] = 0
            query_modes = [10]
            await interface.update_wheel_velocities(speeds, groups, max_output_rps=2.0)
            self.assertEqual(cycles, [('query', [2])])

            initial_modes.update(dict.fromkeys(ids, 0))
            for scenario, modes, message in (
                    ('immediate_fault', [10], 'FAULT=9'),
                    ('missing_immediate', [10], 'missing'),
                    ('missing_query', [10], 'missing'),
                    ('query_fault', [10], 'FAULT=7'),
                    ('missing_fault', [10], 'FAULT=None'),
                    ('healthy', [1], 'MODE=1'), ('healthy', [11], 'MODE=11'),
                    ('healthy', [0], 'timed out'), ('stalled_query', [10], 'timed out')):
                with self.subTest(scenario=scenario, modes=modes):
                    sent.clear()
                    cycles.clear()
                    query_round = 0
                    query_modes = modes
                    with self.assertRaisesRegex(RuntimeError, message):
                        await interface.update_wheel_velocities(
                            speeds, groups, max_output_rps=2.0, timeout_s=0.1)
                    self.assertEqual(asyncio.all_tasks(), {asyncio.current_task()})

            # Invalid speed validation still precedes every write.
            sent.clear()
            for bad in (math.nan, math.inf, 100.0):
                with self.assertRaises(ValueError):
                    await interface.update_wheel_velocities(
                        dict(speeds, rear_right=bad), groups, max_output_rps=2.0)
            self.assertEqual(sent, [])

        asyncio.run(check())

        # A real interface verification error must still drive the unchanged ROS
        # controller's stop/disarm/fault-latch path, with all device I/O faked.
        sent.clear()
        cycles.clear()
        scenario = 'query_fault'
        query_modes = [10]
        query_round = 0
        node = self.make_node(hardware=True)
        self.arm(node)
        node.motors = groups
        node._update_wheel_velocities = interface.update_wheel_velocities
        node._stop_motors = interface.stop_motors
        node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
        self.assertTrue(node._actuator_fault)
        self.assertFalse(node.armed)
        self.assertTrue(node._motors_stopped)
        self.assertEqual(cycles, [('query', ids), ('stop', ids), ('query', ids)])
        count = len(sent)
        self.arm(node, expected=False)
        node.controller_sub.callback(SimpleNamespace(data=[0.4, 0.0]))
        self.assertEqual(len(sent), count)

    def test_publisher_deadman_loss_neutral_and_mode_gating(self):
        source = self.source.parents[1] / 'ralphee_remote_control' / 'ralphee_remote_control'
        reader = Mock()
        package = fake_module('ralphee_remote_control', __path__=[])
        reader_module = fake_module('ralphee_remote_control.remote_controller',
                                   Controller=lambda **kwargs: reader)
        spec = importlib.util.spec_from_file_location(
            'publisher_under_test', source / 'remote_control_publisher.py')
        publisher_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'ralphee_remote_control': package,
                                     'ralphee_remote_control.remote_controller': reader_module}):
            spec.loader.exec_module(publisher_module)
        FakeNode.parameter_values = {}
        publisher = publisher_module.ControllerPublisher()

        def tick(speed=0.0, steering=0.0, enable=False, x=False, y=False, healthy=True):
            reader.read_command.return_value = (speed, steering, enable, x, y, healthy)
            publisher.timer_callback()
            return (publisher.cmd_vel_pub.publish.call_args.args[0].data,
                    publisher.state_pub.publish.call_args.args[0].data)

        self.assertEqual(tick(0.5, enable=True), ([0.0, 0.0], 0))
        tick()  # Released and neutral.
        self.assertEqual(tick(enable=True), ([0.0, 0.0], 3))
        self.assertEqual(tick(0.5, 0.2, enable=True), ([0.5, 0.2], 3))
        self.assertEqual(tick(healthy=False), ([0.0, 0.0], 0))
        self.assertEqual(tick(0.5, enable=True), ([0.0, 0.0], 0))
        tick(0.5)  # Released but not neutral.
        self.assertEqual(tick(enable=True), ([0.0, 0.0], 0))
        tick()
        self.assertEqual(tick(enable=True), ([0.0, 0.0], 3))
        self.assertEqual(tick(0.5, enable=True, y=True), ([0.0, 0.0], 2))
        self.assertEqual(tick(0.5, enable=True, x=True), ([0.0, 0.0], 0))
        reader.read_command.side_effect = RuntimeError('Input reader unavailable')
        publisher.timer_callback()
        self.assertEqual(publisher.cmd_vel_pub.publish.call_args.args[0].data, [0.0, 0.0])
        self.assertEqual(publisher.state_pub.publish.call_args.args[0].data, 0)


if __name__ == '__main__':
    unittest.main()
