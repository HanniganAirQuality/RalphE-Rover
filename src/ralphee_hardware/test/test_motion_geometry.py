"""Hardware-free checks of RalphE's measured, asymmetric motion geometry."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


class TestMotionGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocked_imports = (
            'rclpy', 'geometry_msgs', 'std_msgs', 'moteus', 'usb', 'serial',
            'inputs', 'numpy', 'ralphee_remote_control',
            'ralphee_hardware.ralphee_hardware_controller',
            'ralphee_hardware.ralphee_motor_interface',
            'ralphee_hardware.ralphee_servo_interface',
        )
        isolation = patch.dict(sys.modules, {name: None for name in cls.blocked_imports})
        isolation.start()
        cls.addClassCleanup(isolation.stop)
        source = Path(__file__).resolve().parents[1] / 'ralphee_hardware' / 'motion_geometry.py'
        spec = importlib.util.spec_from_file_location('motion_geometry_under_test', source)
        cls.motion = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.motion
        spec.loader.exec_module(cls.motion)
        cls.geometry = cls.motion.MEASURED_GEOMETRY
        # Synthetic test limits only: RalphE's physical wheel-angle limits are unknown.
        cls.test_angle_limit = math.radians(30)
        cls.limits = {
            name: (-cls.test_angle_limit, cls.test_angle_limit)
            for name in ('front_left', 'front_right', 'rear_left', 'rear_right')
        }

    def solve_curvature(self, velocity, curvature, **kwargs):
        return self.motion.solve_curvature(
            velocity, curvature, geometry=self.geometry,
            steering_limits_rad=kwargs.get('limits', self.limits))

    def solve_twist(self, velocity, yaw_rate):
        return self.motion.solve_twist(
            velocity, yaw_rate, geometry=self.geometry, steering_limits_rad=self.limits)

    def assert_finite_motion(self, solution):
        self.assertEqual(set(solution), set(self.geometry.wheel_positions_m))
        for wheel in solution.values():
            self.assertTrue(math.isfinite(wheel.steering_angle_rad))
            self.assertTrue(math.isfinite(wheel.linear_velocity_mps))

    def test_ros_and_device_imports_remain_blocked(self):
        for name in self.blocked_imports:
            with self.subTest(module=name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(name)

    def test_measured_coordinates_include_front_rear_asymmetry(self):
        self.assertEqual(dict(self.geometry.wheel_positions_m), {
            'front_left': (0.350, 0.290), 'front_right': (0.350, -0.290),
            'middle_left': (0.0, 0.290), 'middle_right': (0.0, -0.290),
            'rear_left': (-0.330, 0.290), 'rear_right': (-0.330, -0.290),
        })
        self.assertAlmostEqual(
            self.geometry.front_to_middle_m + self.geometry.middle_to_rear_m, 0.680)
        self.assertEqual(self.geometry.nominal_wheel_radius_m, 0.125)

    def test_straight_forward_and_reverse(self):
        for velocity in (0.5, -0.5):
            for solve in (self.solve_twist, self.solve_curvature):
                with self.subTest(velocity=velocity, solver=solve.__name__):
                    solution = solve(velocity, 0.0)
                    self.assert_finite_motion(solution)
                    for wheel in solution.values():
                        self.assertEqual(wheel.steering_angle_rad, 0.0)
                        self.assertEqual(wheel.linear_velocity_mps, velocity)

    def test_zero_commands_are_centered_and_finite(self):
        for solve in (self.solve_twist, self.solve_curvature):
            solution = solve(0.0, 0.0)
            self.assert_finite_motion(solution)
            for wheel in solution.values():
                self.assertEqual(wheel.steering_angle_rad, 0.0)
                self.assertEqual(wheel.linear_velocity_mps, 0.0)

    def test_zero_velocity_rejects_nonzero_curvature_and_pure_rotation(self):
        for value in (-0.5, 0.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'Zero linear velocity'):
                    self.solve_curvature(0.0, value)
                with self.assertRaisesRegex(ValueError, 'Pure rotation'):
                    self.solve_twist(0.0, value)

    def test_positive_forward_yaw_steers_front_left_rear_right(self):
        solution = self.solve_twist(0.5, 0.25)
        for side in ('left', 'right'):
            self.assertGreater(solution[f'front_{side}'].steering_angle_rad, 0.0)
            self.assertLess(solution[f'rear_{side}'].steering_angle_rad, 0.0)
        for axle in ('front', 'middle', 'rear'):
            self.assertGreater(
                solution[f'{axle}_right'].linear_velocity_mps,
                solution[f'{axle}_left'].linear_velocity_mps)

    def test_negative_forward_yaw_steers_front_right_rear_left(self):
        solution = self.solve_twist(0.5, -0.25)
        for side in ('left', 'right'):
            self.assertLess(solution[f'front_{side}'].steering_angle_rad, 0.0)
            self.assertGreater(solution[f'rear_{side}'].steering_angle_rad, 0.0)
        for axle in ('front', 'middle', 'rear'):
            self.assertGreater(
                solution[f'{axle}_left'].linear_velocity_mps,
                solution[f'{axle}_right'].linear_velocity_mps)

    def test_mirrored_turns_exchange_left_and_right_results(self):
        for velocity in (0.5, -0.5):
            left_turn = self.solve_curvature(velocity, 0.5)
            right_turn = self.solve_curvature(velocity, -0.5)
            for axle in ('front', 'middle', 'rear'):
                for side, opposite in (('left', 'right'), ('right', 'left')):
                    original = left_turn[f'{axle}_{side}']
                    mirrored = right_turn[f'{axle}_{opposite}']
                    self.assertAlmostEqual(
                        original.steering_angle_rad, -mirrored.steering_angle_rad)
                    self.assertAlmostEqual(
                        original.linear_velocity_mps, mirrored.linear_velocity_mps)

    def test_reverse_preserves_curvature_and_reverses_wheel_velocities(self):
        forward = self.solve_twist(0.5, 0.25)
        reverse = self.solve_twist(-0.5, -0.25)
        for name in forward:
            self.assertEqual(forward[name].steering_angle_rad, reverse[name].steering_angle_rad)
            self.assertEqual(forward[name].linear_velocity_mps, -reverse[name].linear_velocity_mps)

    def test_front_and_rear_angles_and_speeds_are_calculated_independently(self):
        for curvature in (-0.5, 0.5):
            solution = self.solve_curvature(0.5, curvature)
            for side in ('left', 'right'):
                front = solution[f'front_{side}']
                rear = solution[f'rear_{side}']
                self.assertGreater(abs(front.steering_angle_rad), abs(rear.steering_angle_rad))
                self.assertGreater(front.linear_velocity_mps, rear.linear_velocity_mps)
                self.assertEqual(solution[f'middle_{side}'].steering_angle_rad, 0.0)

    def test_wheel_rolling_vectors_match_the_requested_twist(self):
        for velocity in (0.5, -0.5):
            for yaw_rate in (-0.25, 0.0, 0.25):
                solution = self.solve_twist(velocity, yaw_rate)
                self.assert_finite_motion(solution)
                for name, (x, y) in self.geometry.wheel_positions_m.items():
                    wheel = solution[name]
                    self.assertAlmostEqual(
                        wheel.linear_velocity_mps * math.cos(wheel.steering_angle_rad),
                        velocity - yaw_rate * y, places=12)
                    self.assertAlmostEqual(
                        wheel.linear_velocity_mps * math.sin(wheel.steering_angle_rad),
                        yaw_rate * x, places=12)

    def test_motion_converges_smoothly_to_neutral_from_both_sides(self):
        for sign in (-1, 1):
            previous_angle_error = previous_speed_error = math.inf
            for magnitude in (1e-3, 1e-6, 1e-9):
                solution = self.solve_curvature(0.5, sign * magnitude)
                self.assert_finite_motion(solution)
                angle_error = max(abs(w.steering_angle_rad) for w in solution.values())
                speed_error = max(abs(w.linear_velocity_mps - 0.5) for w in solution.values())
                self.assertLess(angle_error, magnitude)
                self.assertLess(speed_error, magnitude)
                self.assertLess(angle_error, previous_angle_error)
                self.assertLess(speed_error, previous_speed_error)
                previous_angle_error, previous_speed_error = angle_error, speed_error

    def test_minimum_radius_boundary_uses_the_front_axle_distance(self):
        minimum_radius = 0.290 + 0.350 / math.tan(self.test_angle_limit)
        for sign, inner_side in ((1, 'left'), (-1, 'right')):
            curvature = sign / minimum_radius
            self.solve_curvature(0.5, curvature * (1 - 1e-6))
            boundary = self.solve_curvature(0.5, curvature)
            self.assertAlmostEqual(
                abs(boundary[f'front_{inner_side}'].steering_angle_rad), self.test_angle_limit)
            self.assertLess(
                abs(boundary[f'rear_{inner_side}'].steering_angle_rad), self.test_angle_limit)
            with self.assertRaises(ValueError):
                self.solve_curvature(0.5, curvature * (1 + 1e-6))

    def test_each_corner_and_both_limit_directions_are_enforced(self):
        for curvature in (-0.5, 0.5):
            solution = self.solve_curvature(0.5, curvature)
            for name in self.limits:
                with self.subTest(curvature=curvature, wheel=name):
                    limits = dict(self.limits)
                    angle = solution[name].steering_angle_rad
                    lower, upper = limits[name]
                    limits[name] = (lower, angle / 2) if angle > 0 else (angle / 2, upper)
                    with self.assertRaisesRegex(ValueError, name):
                        self.solve_curvature(0.5, curvature, limits=limits)

    def test_turning_center_on_or_inside_tracks_is_rejected(self):
        for curvature in (-4.0, -2 / 0.580, 2 / 0.580, 4.0):
            with self.subTest(curvature=curvature):
                with self.assertRaisesRegex(ValueError, 'outside the wheel tracks'):
                    self.solve_curvature(0.5, curvature)

    def test_invalid_geometry_is_rejected(self):
        for field in ('front_to_middle_m', 'middle_to_rear_m', 'track_width_m',
                      'nominal_wheel_radius_m'):
            for value in (0.0, -0.1, math.nan, math.inf, None, 'bad'):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.motion.Geometry(**{field: value})
        with self.assertRaises(ValueError):
            self.motion.Geometry(track_width_m=math.ulp(0.0))
        with self.assertRaises(ValueError):
            self.motion.solve_curvature(0.5, 0.0, geometry=None, steering_limits_rad=self.limits)

    def test_limits_are_required_and_invalid_limits_are_rejected(self):
        with self.assertRaises(TypeError):
            self.motion.solve_curvature(0.5, 0.0, geometry=self.geometry)
        with self.assertRaises(TypeError):
            self.motion.solve_twist(0.5, 0.0, geometry=self.geometry)
        invalid = [None, {}, {**self.limits, 'middle_left': (0.0, 0.0)}]
        for bounds in ((0.1, 0.2), (-0.2, -0.1), (0.2, -0.2),
                       (-math.pi / 2, 0.2), (-0.2, math.pi / 2),
                       (math.nan, 0.2), (-0.2, math.inf), (0.0,), None):
            invalid.append({**self.limits, 'front_left': bounds})
        for limits in invalid:
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError):
                    self.solve_curvature(0.5, 0.0, limits=limits)

    def test_nonfinite_and_malformed_motion_inputs_are_rejected(self):
        for solve in (self.solve_curvature, self.solve_twist):
            for value in (math.nan, math.inf, -math.inf, None, '0.5', True):
                with self.subTest(solver=solve.__name__, value=value):
                    with self.assertRaises(ValueError):
                        solve(value, 0.0)
                    with self.assertRaises(ValueError):
                        solve(0.5, value)

    def test_unrepresentable_calculations_are_rejected_without_changing_inputs(self):
        limits_before = dict(self.limits)
        with self.assertRaises(ValueError):
            self.solve_twist(1e-308, 1e308)
        with self.assertRaises(ValueError):
            self.solve_twist(1e308, 1e-308)
        with self.assertRaisesRegex(ValueError, 'rolling velocity is nonfinite'):
            self.solve_curvature(sys.float_info.max, 0.5)
        self.assertEqual(self.limits, limits_before)
        self.assert_finite_motion(self.solve_twist(0.5, 0.25))


if __name__ == '__main__':
    unittest.main()
