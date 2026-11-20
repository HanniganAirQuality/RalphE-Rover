"""Poll complete Linux gamepad state; expire any missing reader progress.

EVIOCG* reads work with stationary controls and fail when Linux removes the
input device. A wireless receiver must also expose controller link loss through
its driver. Verify C6 power-off with the dongle left plugged in before driving;
successful kernel-state queries alone cannot prove a receiver's radio link.
"""

import inputs
import threading
import math
import struct
from time import sleep, monotonic
from types import SimpleNamespace

try:
    from fcntl import ioctl
except ImportError:
    ioctl = None  # Target is Linux; unsupported platforms remain disarmed.

DEADZONE = 0.025
DEADZONE_INV = 1-DEADZONE

MAX_SPEED = 0.75

# Used as reference
# https://stackoverflow.com/questions/46506850/how-can-i-get-input-from-an-xbox-one-controller-in-python
class Controller(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self, input_timeout_s=0.3):
        self.input_timeout_s = input_timeout_s
        self.velocity: float = 0.0
        self.angle: float = 0.0

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._lock = threading.Lock()
        self.last_update_time = None
        self._seen_controls = set()
        self._input_loss_pending = True

        self._monitor_thread = threading.Thread(target=self.update, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def update(self):
        while True:
            try:
                if ioctl is None:
                    raise RuntimeError('Live gamepad state polling requires Linux.')
                gamepads = inputs.DeviceManager().gamepads
                if not gamepads:
                    raise RuntimeError('No gamepad detected.')
                path = gamepads[0].get_char_device_path()
                print(f'[update_inputs] polling gamepad state: {path}')
                with open(path, 'rb', buffering=0) as device:
                    while True:
                        started = monotonic()
                        events = self._read_linux_state(device)
                        if monotonic() - started >= self.input_timeout_s:
                            raise TimeoutError('Gamepad state poll exceeded freshness timeout.')
                        with self._lock:
                            if (self.last_update_time is not None
                                    and started - self.last_update_time >= self.input_timeout_s):
                                # Report a gap even if this poll recovered before ROS noticed.
                                self._input_loss_pending = True
                            self.update_inputs(events)
                            self.update_velocity()
                            self.update_angle()
                            # Advance only after a full successful poll. Use its start
                            # time so time spent inside a slow query counts toward age.
                            self.last_update_time = started
                        sleep(0.05)
            except Exception as error:
                with self._lock:
                    self._invalidate_input()
                print(f'[update_inputs] input unavailable: {error}')
                sleep(1)

    def _read_linux_state(self, device):
        """Read the existing axes/buttons without blocking for a change event."""
        events = []
        # Linux input.h: EVIOCGABS(code), 24-byte struct input_absinfo.
        # These ioctl encodings are shared by Jetson ARM64 and Linux x86-64.
        for name, code in (('ABS_Y', 0x01), ('ABS_RX', 0x03),
                           ('ABS_Z', 0x02), ('ABS_RZ', 0x05)):
            value, minimum, maximum, _, _, _ = struct.unpack(
                '=6i', ioctl(device, 0x80184540 + code, bytes(24)))
            if name in ('ABS_Y', 'ABS_RX') and not (
                    -self.MAX_JOY_VAL <= minimum < 0 < maximum <= self.MAX_JOY_VAL):
                raise ValueError('Unsupported drive-axis range; expected signed Xbox axes.')
            events.append(SimpleNamespace(code=name, state=value))
        # EVIOCGKEY(96): current key bitmap, including all BTN_GAMEPAD codes.
        keys = ioctl(device, 0x80604518, bytes(96))
        for name, code in (('BTN_SOUTH', 0x130), ('BTN_NORTH', 0x133),
                           ('BTN_WEST', 0x134), ('BTN_TL', 0x136), ('BTN_TR', 0x137)):
            events.append(SimpleNamespace(
                code=name, state=int(bool(keys[code // 8] & (1 << (code % 8))))))
        return events

    def _invalidate_input(self):
        # Called under the lock. Never reuse cached axes/buttons after a gap.
        self.velocity = self.angle = 0.0
        self.LeftJoystickY = self.RightJoystickX = 0
        self.LeftTrigger = self.RightTrigger = 0
        self.LeftBumper = self.RightBumper = 0
        self.A = self.X = self.Y = 0
        self.last_update_time = None
        self._seen_controls.clear()
        self._input_loss_pending = True

    def read_command(self):
        """Return an atomic (speed, steering, A, X, Y, healthy) snapshot.

        Only complete state polls refresh health. A fault/gap is delivered at
        least once, so recovery cannot hide the need to release A and rearm.
        """
        with self._lock:
            if (self.last_update_time is None
                    or not self._monitor_thread.is_alive()
                    or monotonic() - self.last_update_time >= self.input_timeout_s):
                self._invalidate_input()
            if self._input_loss_pending:
                self._input_loss_pending = False
                return 0.0, 0.0, False, False, False, False
            healthy = {'ABS_Y', 'ABS_RX', 'BTN_SOUTH'} <= self._seen_controls
            if not healthy:
                return 0.0, 0.0, False, False, False, False
            return self.velocity, self.angle, bool(self.A), bool(self.X), bool(self.Y), True

    def update_inputs(self, events):
        for event in events:
            if event.code == 'SYN_DROPPED':
                raise ValueError('Gamepad events were lost; current input is unknown.')
            if not math.isfinite(event.state):
                raise ValueError('Nonfinite gamepad report.')
            if event.code in ('ABS_Y', 'ABS_RX'):
                if abs(event.state) > self.MAX_JOY_VAL:
                    raise ValueError('Invalid drive-axis report.')
            if event.code in ('BTN_SOUTH', 'BTN_WEST', 'BTN_NORTH'):
                if event.state not in (0, 1):
                    raise ValueError('Invalid enable/mode button report.')
            self._seen_controls.add(event.code)
            if event.code == 'ABS_Y':
                self.LeftJoystickY = event.state / Controller.MAX_JOY_VAL # normalize between -1 and 1
            elif event.code == 'ABS_X':
                self.LeftJoystickX = event.state / Controller.MAX_JOY_VAL # normalize between -1 and 1
            elif event.code == 'ABS_RY':
                self.RightJoystickY = event.state / Controller.MAX_JOY_VAL # normalize between -1 and 1
            elif event.code == 'ABS_RX':
                self.RightJoystickX = event.state / Controller.MAX_JOY_VAL # normalize between -1 and 1
            elif event.code == 'ABS_Z':
                self.LeftTrigger = event.state / Controller.MAX_TRIG_VAL # normalize between 0 and 1
            elif event.code == 'ABS_RZ':
                self.RightTrigger = event.state / Controller.MAX_TRIG_VAL # normalize between 0 and 1
            elif event.code == 'BTN_TL':
                self.LeftBumper = event.state
            elif event.code == 'BTN_TR':
                self.RightBumper = event.state
            elif event.code == 'BTN_SOUTH':
                self.A = event.state
            elif event.code == 'BTN_NORTH':
                self.Y = event.state #previously switched with X
            elif event.code == 'BTN_WEST':
                self.X = event.state #previously switched with Y
            elif event.code == 'BTN_EAST':
                self.B = event.state
            elif event.code == 'BTN_THUMBL':
                self.LeftThumb = event.state
            elif event.code == 'BTN_THUMBR':
                self.RightThumb = event.state
            elif event.code == 'BTN_SELECT':
                self.Back = event.state
            elif event.code == 'BTN_START':
                self.Start = event.state
            elif event.code == 'BTN_TRIGGER_HAPPY1':
                self.LeftDPad = event.state
            elif event.code == 'BTN_TRIGGER_HAPPY2':
                self.RightDPad = event.state
            elif event.code == 'BTN_TRIGGER_HAPPY3':
                self.UpDPad = event.state
            elif event.code == 'BTN_TRIGGER_HAPPY4':
                self.DownDPad = event.state

    def update_velocity(self):
        # Equation https://www.desmos.com/calculator/hmsh6km1tl
        if abs(self.LeftJoystickY) < DEADZONE:
            self.velocity = 0.0
            return
        # Keeps joystick in 'square' zone
        if self.LeftJoystickY != 0.0:
            sign = abs(self.LeftJoystickY)/self.LeftJoystickY
        else:
            sign = 1.0
        vel = (self.LeftJoystickY-DEADZONE*sign) / (DEADZONE_INV)

        speed_multiplier = MAX_SPEED
        if self.LeftTrigger:
            speed_multiplier *= 2
        if self.LeftBumper:
            speed_multiplier *= 0.5

        self.velocity = max(-1, min(1, vel)) * speed_multiplier

    def update_angle(self):
        # Equation https://www.desmos.com/calculator/hmsh6km1tl
        
        if abs(self.RightJoystickX) < DEADZONE:
            self.angle = 0.0
            return
        # Keeps joystick in 'square' zone
        if self.RightJoystickX != 0.0:
            sign = abs(self.RightJoystickX)/self.RightJoystickX
        else:
            sign = 1.0
        angle = (self.RightJoystickX - DEADZONE*sign) / (DEADZONE_INV)
        angle_multiplier = 0.75
        if self.RightTrigger:
            angle_multiplier = 1
        if self.RightBumper:
            angle_multiplier = 0.5
        self.angle = max(-1, min(1, angle*angle_multiplier))
