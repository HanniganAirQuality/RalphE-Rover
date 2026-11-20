import serial
import time
import os.path

async def init_servos(*, write_timeout_s=0.2) -> serial.Serial:
    print('[init_servos] create arduino object')
    # Check for existence of file
    if not os.path.exists('/dev/ttyACM0'):
        raise Exception('[init_servos] arduino not plugged in!')
    
    # Creates arduino object
    # A stuck write must not hold up the controller's stop/watchdog handling.
    arduino = serial.Serial('/dev/ttyACM0', 9600, write_timeout=write_timeout_s)

    # Sets up usb port
    print('[init_servos] setup serial port for arduino')
    arduino.setDTR(False)
    time.sleep(0.1) # Initially set to 1
    arduino.flushInput()
    arduino.setDTR(True)
    time.sleep(0.1) # Initially set to 2
    print('[init_servos] arduino ready!')
    return arduino

async def update_servos(angle: float, arduino: serial.Serial):
    command = (str(angle) + '\r').encode()
    if arduino.write(command) != len(command):
        raise IOError('Incomplete Arduino steering write.')
    # print(f'[update_servos] t: {angle}')
