"""Pure counter-phase steering geometry, with no ROS or device dependencies.

The origin is the middle axle midpoint: x forward, y left, positive yaw
counterclockwise. Geometry defaults are physical measurements supplied for
RalphE. Steering limits are required wheel-heading angles, not servo commands.
Results are ideal wheel-center rolling velocities in m/s and angles in radians.
No motor polarity, gearing, speed scaling, or servo calibration is applied.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType


_CORNER_WHEELS = ('front_left', 'front_right', 'rear_left', 'rear_right')


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(f'{name} must be a finite number.')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a finite number.') from None
    if not math.isfinite(result):
        raise ValueError(f'{name} must be a finite number.')
    return result


@dataclass(frozen=True)
class Geometry:
    """Measured axle distances and track; radius is nominal, not load calibrated."""

    front_to_middle_m: float = 0.350
    middle_to_rear_m: float = 0.330
    track_width_m: float = 0.580
    nominal_wheel_radius_m: float = 0.125

    def __post_init__(self):
        for name in ('front_to_middle_m', 'middle_to_rear_m', 'track_width_m',
                     'nominal_wheel_radius_m'):
            value = _finite_float(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f'{name} must be positive.')
            object.__setattr__(self, name, value)
        if self.track_width_m / 2 == 0:
            raise ValueError('Half-track must be representable and positive.')

    @property
    def wheel_positions_m(self) -> Mapping[str, tuple[float, float]]:
        """Return all six wheel centers relative to the middle axle midpoint."""
        half_track = self.track_width_m / 2
        return MappingProxyType({
            'front_left': (self.front_to_middle_m, half_track),
            'front_right': (self.front_to_middle_m, -half_track),
            'middle_left': (0.0, half_track),
            'middle_right': (0.0, -half_track),
            'rear_left': (-self.middle_to_rear_m, half_track),
            'rear_right': (-self.middle_to_rear_m, -half_track),
        })


MEASURED_GEOMETRY = Geometry()


@dataclass(frozen=True)
class WheelMotion:
    """Wheel-heading angle and signed rolling speed, before hardware conversion."""

    steering_angle_rad: float
    linear_velocity_mps: float


def curvature_from_normalized_steering(
    normalized_steering: float,
    *,
    reference_angle_rad: float,
    geometry: Geometry,
) -> float:
    """Convert steering in [-1, 1] to signed curvature in inverse metres.

    Zero is straight; positive is left in ROS coordinates. The explicitly
    supplied reference angle is the front-inner wheel-heading magnitude at
    full steering, strictly between zero and pi/2. For angle = steering*reference,
    curvature = sin(angle)/(front_distance*cos(angle) + half_track*abs(sin(angle))).

    Invalid or unrepresentable requests raise ValueError. Check all four
    physical wheel-angle limits separately with solve_curvature().
    """
    steering = _finite_float(normalized_steering, 'Normalized steering')
    reference = _finite_float(reference_angle_rad, 'Steering reference angle')
    if not isinstance(geometry, Geometry):
        raise ValueError('geometry must be a Geometry instance.')
    if not 0 < reference < math.pi / 2:
        raise ValueError('Steering reference angle must lie strictly between zero and pi/2.')
    if abs(steering) > 1:
        raise ValueError('Normalized steering must lie in [-1, 1].')
    if steering == 0:
        return 0.0

    angle = steering * reference
    sine = math.sin(angle)
    denominator = (geometry.front_to_middle_m * math.cos(angle)
                   + geometry.track_width_m / 2 * abs(sine))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError('Steering curvature calculation is unrepresentable.')
    curvature = sine / denominator
    if (not math.isfinite(curvature) or curvature == 0
            or abs(curvature) >= 2.0 / geometry.track_width_m):
        raise ValueError('Steering curvature is unrepresentable outside the wheel tracks.')
    return curvature


def normalized_steering_from_curvature(
    curvature_inv_m: float,
    *,
    reference_angle_rad: float,
    geometry: Geometry,
) -> float:
    """Invert curvature_from_normalized_steering() within its supported range.

    The supported curvature interval is bounded by normalized steering +/-1
    with the same geometry and explicit reference angle. Positive is left;
    zero returns exactly zero. Out-of-range requests raise ValueError.
    """
    curvature = _finite_float(curvature_inv_m, 'Curvature')
    reference = _finite_float(reference_angle_rad, 'Steering reference angle')
    maximum = curvature_from_normalized_steering(
        1.0, reference_angle_rad=reference, geometry=geometry)
    if abs(curvature) > maximum:
        raise ValueError('Curvature exceeds the normalized steering range.')
    if curvature == 0:
        return 0.0
    # Return exact normalized endpoints after validating the curvature range.
    if abs(curvature) == maximum:
        return math.copysign(1.0, curvature)

    longitudinal = 1.0 - geometry.track_width_m / 2 * abs(curvature)
    lateral = geometry.front_to_middle_m * curvature
    if longitudinal <= 0 or not math.isfinite(lateral):
        raise ValueError('Steering angle calculation is unrepresentable.')
    steering = math.atan2(lateral, longitudinal) / reference
    if not math.isfinite(steering) or steering == 0 or abs(steering) > 1:
        raise ValueError('Normalized steering calculation is unrepresentable.')
    return steering


def _validated_limits(
    limits: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    if not isinstance(limits, Mapping) or set(limits) != set(_CORNER_WHEELS):
        raise ValueError('Steering limits must name exactly the four corner wheels.')
    result = {}
    for name in _CORNER_WHEELS:
        try:
            lower, upper = limits[name]
        except (TypeError, ValueError):
            raise ValueError(f'{name} limits must be a (minimum, maximum) pair.') from None
        lower = _finite_float(lower, f'{name} minimum angle')
        upper = _finite_float(upper, f'{name} maximum angle')
        if not -math.pi / 2 < lower <= 0 <= upper < math.pi / 2:
            raise ValueError(f'{name} limits must include zero and lie inside +/-pi/2.')
        result[name] = (lower, upper)
    return result


def _checked_angle(angle: float, limits: tuple[float, float], name: str) -> float:
    lower, upper = limits
    bounded = min(max(angle, lower), upper)
    # Round floating-point boundary error onto the configured endpoint.
    if abs(angle - bounded) > 4 * math.ulp(bounded):
        raise ValueError(f'{name} steering exceeds its configured wheel-angle limits.')
    return bounded


def solve_curvature(
    linear_velocity_mps: float,
    curvature_inv_m: float,
    *,
    geometry: Geometry,
    steering_limits_rad: Mapping[str, tuple[float, float]],
) -> Mapping[str, WheelMotion]:
    """Calculate six independent wheel motions for signed curvature.

    Positive curvature puts the turning center on the left. For each wheel,
    A = 1 - curvature*y, B = curvature*x, angle = atan2(B, A), and rolling
    speed = body_speed*hypot(A, B). The two middle wheels stay straight.

    Limits must be explicitly supplied for every corner wheel. The turning
    center must remain outside the tracks. Zero speed requires zero curvature
    and returns centered steering and six zero speeds. Invalid or infeasible
    requests raise ValueError; no motion is saturated or speed scaled.
    """
    velocity = _finite_float(linear_velocity_mps, 'Linear velocity')
    curvature = _finite_float(curvature_inv_m, 'Curvature')
    if not isinstance(geometry, Geometry):
        raise ValueError('geometry must be a Geometry instance.')
    limits = _validated_limits(steering_limits_rad)
    positions = geometry.wheel_positions_m

    if velocity == 0:
        if curvature != 0:
            raise ValueError('Zero linear velocity requires zero curvature.')
        return MappingProxyType({name: WheelMotion(0.0, 0.0) for name in positions})
    if abs(curvature) >= 2.0 / geometry.track_width_m:
        raise ValueError('The turning center must remain outside the wheel tracks.')

    result = {}
    for name, (x, y) in positions.items():
        longitudinal = 1.0 - curvature * y
        lateral = curvature * x
        if not math.isfinite(longitudinal) or not math.isfinite(lateral):
            raise ValueError(f'{name} motion calculation is nonfinite.')
        if longitudinal <= 0:
            raise ValueError('The turning center must remain outside the wheel tracks.')
        angle = math.atan2(lateral, longitudinal) if x != 0 and curvature != 0 else 0.0
        if name in limits:
            angle = _checked_angle(angle, limits[name], name)
        speed = velocity * math.hypot(longitudinal, lateral)
        if not math.isfinite(speed):
            raise ValueError(f'{name} rolling velocity is nonfinite.')
        result[name] = WheelMotion(angle, speed)
    return MappingProxyType(result)


def solve_twist(
    linear_velocity_mps: float,
    yaw_rate_rad_s: float,
    *,
    geometry: Geometry,
    steering_limits_rad: Mapping[str, tuple[float, float]],
) -> Mapping[str, WheelMotion]:
    """Calculate wheel motions for a twist referenced to the middle axle.

    Curvature is yaw_rate/linear_velocity, retaining the sign of reverse
    velocity. A zero twist returns centered zero motion. Pure rotation and
    unrepresentable curvature raise ValueError, as do infeasible wheel angles.
    """
    velocity = _finite_float(linear_velocity_mps, 'Linear velocity')
    yaw_rate = _finite_float(yaw_rate_rad_s, 'Yaw rate')
    if velocity == 0:
        if yaw_rate != 0:
            raise ValueError('Pure rotation is unsupported.')
        curvature = 0.0
    else:
        curvature = yaw_rate / velocity
        if yaw_rate != 0 and curvature == 0:
            raise ValueError('Twist curvature underflows floating-point representation.')
    return solve_curvature(
        velocity, curvature, geometry=geometry, steering_limits_rad=steering_limits_rad)
