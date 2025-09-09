import numpy as np

from OCCS.connector import MockHardware
from OCCS.optimizer.objective import create_hardware_objective
from OCCS.simulate import get_response


def _save_two_row_csv(path, x, y):
    data = np.vstack([x, y])
    np.savetxt(path, data, delimiter=",")


def test_hardware_objective_matches_target(tmp_path):
    wavelengths = np.linspace(0.0, 1.0, 50)
    volts = np.array([0.5, -0.3])
    target = get_response(wavelengths, volts)
    csv_path = tmp_path / "ideal.csv"
    _save_two_row_csv(csv_path, wavelengths, target)

    hw = MockHardware(
        dac_size=volts.size,
        wavelength=wavelengths,
        noise_std=0.0,
        rng=np.random.default_rng(0),
    )
    obj = create_hardware_objective(hw, csv_path, M=50)

    loss, diag = obj(volts)
    assert loss < 1e-5
    assert abs(diag["delta_nm"]) < 1e-2

    worse_loss, _ = obj(volts + 0.1)
    assert worse_loss > loss


def test_hardware_objective_with_noise(tmp_path):
    wavelengths = np.linspace(0.0, 1.0, 50)
    volts = np.array([0.5, -0.3])
    target = get_response(wavelengths, volts)
    csv_path = tmp_path / "ideal.csv"
    _save_two_row_csv(csv_path, wavelengths, target)

    hw = MockHardware(
        dac_size=volts.size,
        wavelength=wavelengths,
        noise_std=0.05,
        rng=np.random.default_rng(0),
    )
    obj = create_hardware_objective(hw, csv_path, M=50)

    loss1, _ = obj(volts)
    loss2, _ = obj(volts)
    assert loss1 != loss2
