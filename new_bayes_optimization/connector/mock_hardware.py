import numpy as np
from sympy.physics.units import current

from new_bayes_optimization.simulate import get_response


class MockHardware:
    def __init__(self, DAC_SIZE, wavelength):
        self.DAC_SIZE = DAC_SIZE
        self.current_volts = np.zeros(DAC_SIZE)
        self.wavelength = wavelength

    def apply_voltage(self, new_volts: np.ndarray):
        self.current_volts = new_volts
        print(f"电压已更新：{self.current_volts}")

    def read_voltage(self):
        return self.current_volts

    def get_simulated_response(self,):
        return get_response(self.wavelength, self.current_volts)

if __name__ == "__main__":
    hardware = MockHardware(4, np.linspace(0,1,11))
    hardware.apply_voltage(np.array([1,0,0]))
    print(hardware.get_simulated_response())
