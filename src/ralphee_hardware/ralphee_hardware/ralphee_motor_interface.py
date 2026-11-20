import asyncio
import moteus
import time
import math
import usb
import numpy as np
from collections.abc import Mapping
from ralphee_hardware.motion_geometry import MEASURED_GEOMETRY

# Can bus id, bisit mjbots moteus repo for more info: https://github.com/mjbots/moteus
# center motor id should be in center.
# 1 5 3
# 0 3
LEFT_MOTOR_IDS  = [2, 5, 3]
RIGHT_MOTOR_IDS = [0, 4, 1]

# Publisher runs at 10 Hz. This is a backup for a stopped ROS/Python process,
# not an input-health check. Verify latency and the deployed watchdog action.
DEFAULT_WATCHDOG_TIMEOUT_S = 0.5


def _motor_reply_mode(controller, result):
    """Require a reply from this motor, with mode/fault fields and no fault."""
    if result is None or result.id != controller.id:
        raise RuntimeError(f'Motor {controller.id}: missing or mismatched reply.')
    mode = result.values.get(moteus.Register.MODE)
    fault = result.values.get(moteus.Register.FAULT)
    if (fault != 0 or not isinstance(mode, int) or isinstance(mode, bool)
            or not 0 <= mode <= 15 or mode in (moteus.Mode.FAULT, moteus.Mode.TIMEOUT)):
        raise RuntimeError(f'Motor {controller.id}: MODE={mode}, FAULT={fault}.')
    return mode


async def _verify_motor_modes(controllers, expected_mode):
    """Query until every motor reaches the requested mode; caller bounds time."""
    pending = list(controllers)
    transport = moteus.get_singleton_transport()
    while pending:
        # Firmware stages register writes before applying them in Poll()/the ISR.
        # Yield before each independent read; the read, not this delay, proves mode.
        await asyncio.sleep(0.001)
        results = await transport.cycle([controller.make_query() for controller in pending])
        by_id = {result.id: result for result in results}
        pending = [
            controller for controller in pending
            if _motor_reply_mode(controller, by_id.get(controller.id)) != expected_mode]
        # Internal enabling/calibration modes (2/3/4) or the previous healthy mode
        # may persist briefly. They are never accepted as the requested final mode.


async def stop_motors(controller_groups, *, timeout_s=0.2):
    """Batch all six stops, then independently confirm stopped mode/fault=0."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('Stop timeout must be finite and positive seconds.')
    controllers = [controller for group in controller_groups for controller in group]
    if len(controllers) != 6 or len({controller.id for controller in controllers}) != 6:
        raise ValueError('Stopping requires six distinct motor controllers.')
    # Keep the existing default fdcanusb transport and all-six stop batch.
    # A combined write/query can report the PREVIOUS mode, so request no state
    # in the stop frame and verify all six with subsequent query-only frames.
    commands = [controller.make_stop(query=False) for controller in controllers]
    try:
        async with asyncio.timeout(timeout_s):
            await moteus.get_singleton_transport().cycle(commands)
            await _verify_motor_modes(controllers, moteus.Mode.STOPPED)
    except TimeoutError as error:
        raise RuntimeError(
            f'All-six motor stop batch/state verification timed out after {timeout_s} s; '
            'stop not confirmed.') from error


# [0] = Left Non Center
# [1] = Left Center
# [2] = Right Non Center
# [3] = Right Center
MOTOR_COORDS = np.array([[-0.3048,0.3556], [-0.3048,0.0],  # Left Side
                         [ 0.3048,0.3556], [ 0.3048,0.0]]) # Right Side

def get_motor_radiuses(radius: float) -> np.ndarray[float]:
    """
        Gets radius to each motor, used velocity calculation
        Args:
            radius:
                Radius from center over rover to turn point
        Return:
            Array of radius to each motor
    """
    global MOTOR_COORDS

    ROVER_CENTER = np.array([radius,0.0])
    radiuses = np.empty(6, dtype=float)

    for i, motor_coord in enumerate(MOTOR_COORDS):
        p = ROVER_CENTER - motor_coord
        sum_sq = np.dot(p, p)
        radiuses[i] = np.sqrt(sum_sq)

    return radiuses

async def init_motors() -> list[list[moteus.Controller]]:
    """
        Initializes moteus controllers!
        Return:
            Groups of moteus controllers! Each group has a different velocity sent to it.
    """
    print("[init_motors] resetting device file...")

    # Reset fdcamusb device
    # Without this step the program will get stuck at 'set_stop' on repeat runs.
    dev = usb.core.find(idVendor=0x0483, idProduct=0x5740)
    if dev is None:
        raise Exception('[init_motors] plug in motors!')

    dev.reset()

    print("[init_motors] reset usb!")
    print('[init_motors] create controller objects')
    qr = moteus.QueryResolution()
    qr.trajectory_complete = moteus.INT8

    # Create Motor Objects
    left_controllers = []
    for id in LEFT_MOTOR_IDS:
        left_controllers.append(moteus.Controller(id, query_resolution=qr))
    right_controllers = []
    for id in RIGHT_MOTOR_IDS:
        right_controllers.append(moteus.Controller(id, query_resolution=qr))

    # Reset motors incase of inpropper shutdown.
    print('[init_motors] set stop')
    await stop_motors([left_controllers, right_controllers])
    print('[init_motors] ready!')

    # Return groups
    return [left_controllers, right_controllers]

async def update_motors(velocity: float, radius: float, controller_groups: list[list[moteus.Controller]],
                        *, watchdog_timeout_s=DEFAULT_WATCHDOG_TIMEOUT_S):
    """
        Updates motor velocities. https://www.desmos.com/calculator/pwxu4jlu2f
        Args:
            velocity:
                Speed that the rover itself will travel at.
            radius:
                length from turning point to center of body
            controller_groups:
                Controller groups to be set.
    """
    if not math.isfinite(watchdog_timeout_s) or watchdog_timeout_s <= 0:
        raise ValueError('watchdog_timeout_s must be finite and positive.')
    coroutines = []

    if math.isfinite(radius):
        radiuses = get_motor_radiuses(radius) / radius
    else:
        radiuses = np.ones(4)
    if velocity > 1:
        velocity = 1

    # Gets the velocity for both sides.
    left_velocity         =  velocity*abs(radiuses[0])
    left_center_velocity  =  velocity*abs(radiuses[1])
    right_velocity        = -velocity*abs(radiuses[2])
    right_center_velocity = -velocity*abs(radiuses[3])

    LEFT_SIDE_INDEX  = 0
    RIGHT_SIDE_INDEX = 1

    # Start coroutines
    for i, c in enumerate(controller_groups[LEFT_SIDE_INDEX]):
        v = left_velocity if i != 1 else left_center_velocity
        coroutines.append(c.set_position(position=math.nan, velocity=v, query=True,
                                        watchdog_timeout=watchdog_timeout_s))

    for i, c in enumerate(controller_groups[RIGHT_SIDE_INDEX]):
        v = right_velocity if i != 1 else right_center_velocity
        coroutines.append(c.set_position(position=math.nan, velocity=v, query=True,
                                        watchdog_timeout=watchdog_timeout_s))

    # print(f'[update_motors] lv: {left_velocity}, lcv: {left_center_velocity}, rv: {right_velocity}, rcv: {right_center_velocity}')

    # Await coroutines
    for coroutine in coroutines:
        await coroutine


def wheel_velocity_mps_to_output_rps(velocity_mps: float) -> float:
    """Convert m/s using the nominal 0.125 m wheel radius from motion_geometry.

    Assumes moteus's configured output represents wheel revolutions after the
    gearbox reduction. No gearbox multiplier is applied here. Verify the
    deployed moteus configuration before enabling physical motor output.
    """
    try:
        if isinstance(velocity_mps, (str, bytes, bool)):
            raise ValueError
        velocity = float(velocity_mps)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Wheel velocity must be a finite number in m/s.') from None
    if not math.isfinite(velocity):
        raise ValueError('Wheel velocity must be a finite number in m/s.')
    output_rps = velocity / (math.tau * MEASURED_GEOMETRY.nominal_wheel_radius_m)
    if not math.isfinite(output_rps):
        raise ValueError('Converted output velocity must be finite.')
    return output_rps


async def update_wheel_velocities(
    wheel_velocities_mps: Mapping[str, float],
    controller_groups: list[list[moteus.Controller]],
    *,
    max_output_rps: float,
    watchdog_timeout_s: float = DEFAULT_WATCHDOG_TIMEOUT_S,
    timeout_s: float = 0.2,
):
    """Send six independent wheel velocities after validating the whole request.

    Input is a wheel-name-to-m/s mapping, using linear_velocity_mps from each
    motion_geometry result. Groups retain init_motors() ordering: left then
    right, each front/middle/rear, with IDs [2, 5, 3] and [0, 4, 1]. Existing
    command signs are preserved: left positive, right negative for forward.

    max_output_rps is a required positive magnitude limit for every wheel in
    either direction. Invalid requests raise ValueError before any command;
    speeds are never clipped. watchdog_timeout_s is the moteus backup timeout,
    in seconds; the ROS controller must also enforce fresh input and arming.
    timeout_s bounds all six writes and any post-command mode verification;
    the controller's existing outer I/O timeout remains in force as well.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('Motor update timeout must be finite and positive seconds.')
    wheel_names = (
        ('front_left', 'middle_left', 'rear_left'),
        ('front_right', 'middle_right', 'rear_right'),
    )
    expected = {name for side in wheel_names for name in side}
    if not isinstance(wheel_velocities_mps, Mapping) or set(wheel_velocities_mps) != expected:
        raise ValueError('Velocities must name exactly the six wheels.')
    try:
        if isinstance(max_output_rps, (str, bytes, bool)):
            raise ValueError
        limit = float(max_output_rps)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('max_output_rps must be a finite positive number.') from None
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('max_output_rps must be a finite positive number.')
    if (isinstance(watchdog_timeout_s, (str, bytes, bool))
            or not math.isfinite(watchdog_timeout_s) or watchdog_timeout_s <= 0):
        raise ValueError('watchdog_timeout_s must be finite and positive.')
    try:
        valid_groups = len(controller_groups) == 2 and all(
            len(group) == 3 for group in controller_groups)
    except TypeError:
        valid_groups = False
    if not valid_groups:
        raise ValueError('Expected left and right controller groups of three motors each.')

    commands = []
    for side, names in enumerate(wheel_names):
        direction = 1.0 if side == 0 else -1.0
        for index, name in enumerate(names):
            controller = controller_groups[side][index]
            if not callable(getattr(controller, 'set_position', None)):
                raise ValueError(f'{name} controller must provide set_position().')
            output_rps = wheel_velocity_mps_to_output_rps(wheel_velocities_mps[name])
            if abs(output_rps) > limit:
                raise ValueError(f'{name} exceeds max_output_rps.')
            commands.append((controller, direction * output_rps))

    try:
        async with asyncio.timeout(timeout_s):
            pending = []
            for controller, output_rps in commands:
                result = await controller.set_position(
                    position=math.nan, velocity=output_rps, query=True,
                    watchdog_timeout=watchdog_timeout_s)
                mode = _motor_reply_mode(controller, result)
                if mode != moteus.Mode.POSITION:
                    pending.append(controller)
            # Do not wait for each motor to enable separately. Verify transitions
            # together after all six commands. Steady POSITION replies add no I/O.
            if pending:
                await _verify_motor_modes(pending, moteus.Mode.POSITION)
    except TimeoutError as error:
        raise RuntimeError(
            f'Motor update/state verification timed out after {timeout_s} s.') from error
