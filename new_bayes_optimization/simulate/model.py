import numpy as np

def get_response(wavelength, input_volts):
    wavelength = np.asarray(wavelength)
    input_volts = np.asarray(input_volts)
    w_min, w_max = wavelength.min(), wavelength.max()
    if w_max == w_min:
        raise ValueError("wavelength 所有元素相同，无法做线性映射。")

    x = (wavelength - w_min) / (w_max - w_min) * (2*np.pi) - np.pi

    i = np.arange(1, input_volts.size + 1, dtype=float)[:, None]
    S = np.cos(i * x)
    response = (input_volts[:, None] * S).sum(axis=0)
    return response

if __name__=="__main__":
    wavelengths = np.linspace(0, 1, 11)
    volts = np.array([1,0,0])
    resp = response_on_wavelength(wavelengths, volts)
    print(resp)