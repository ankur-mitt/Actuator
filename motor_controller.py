from pyfirmata import Arduino, Pin


class MotorController:
    __board: Arduino
    __dir_pin: Pin
    __stp_pin: Pin
    __enb_pin: Pin
    __max_stp: int = 1600
    __rev_deg: int = 360
    __theta: float = 0

    def __set_clockwise(self):
        self.__dir_pin.write(True)

    def __set_anticlockwise(self):
        self.__dir_pin.write(False)

    def __enable_motor(self):
        self.__enb_pin.write(False)

    def __disable_motor(self):
        self.__enb_pin.write(True)

    def __apply_voltage(self):
        self.__stp_pin.write(True)

    def __remove_voltage(self):
        self.__stp_pin.write(False)

    def __give_single_step(self):
        self.__apply_voltage()
        self.__remove_voltage()

    def __setup_defaults(self):
        self.__remove_voltage()
        self.__set_clockwise()
        self.__enable_motor()

    def __rotate(self, __delta: float):
        self.__set_clockwise() if __delta >= 0 else self.__set_anticlockwise()
        __steps_count = round(abs(__delta) * self.__max_stp / self.__rev_deg)
        [self.__give_single_step() for _ in range(__steps_count)]

    def __restore(self):
        self.__rotate(self.__theta * -1)

    def __cleanup(self):
        self.__remove_voltage()
        self.__set_clockwise()
        self.__disable_motor()

    def __init__(self, __board: Arduino, __dir_pin: int, __stp_pin: int, __enb_pin: int):
        self.__board = __board
        self.__dir_pin = self.__board.get_pin(f"d:{__dir_pin}:o")
        self.__stp_pin = self.__board.get_pin(f"d:{__stp_pin}:o")
        self.__enb_pin = self.__board.get_pin(f"d:{__enb_pin}:o")
        self.__setup_defaults()

    def __del__(self):
        self.__restore()
        self.__cleanup()

    def get_position(self):
        return self.__theta

    def set_position(self, __theta: float):
        __delta = __theta - self.__theta
        self.__rotate(__delta)
        self.__theta = __theta


def connect_arduino():
    arduino_port = "COM5"
    arduino_board = Arduino(arduino_port)
    l_controller = MotorController(arduino_board, 9, 10, 11)
    r_controller = MotorController(arduino_board, 3, 4, 5)
    return l_controller, r_controller
