import numpy as np

from new_bayes_optimization.simulate import get_response


def test_zero_noise_equals_deterministic():
    wavelengths = np.linspace(0.0, 1.0, 10)
    volts = np.array([0.5, -0.2, 0.3])
    resp_det = get_response(wavelengths, volts)
    resp_zero = get_response(wavelengths, volts, noise_std=0.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(resp_det, resp_zero)


def test_reproducible_with_seed():
    wavelengths = np.linspace(0.0, 1.0, 20)
    volts = np.array([0.1, 0.2, 0.3])
    noise = np.array([0.01, 0.02, 0.03])
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    resp1 = get_response(wavelengths, volts, noise_std=noise, rng=rng1)
    resp2 = get_response(wavelengths, volts, noise_std=noise, rng=rng2)
    np.testing.assert_allclose(resp1, resp2)


def test_noise_statistics_match_expectation():
    wavelengths = np.linspace(0.0, 1.0, 50)
    volts = np.zeros(3)
    noise_std = np.array([0.1, 0.2, 0.3])
    rng = np.random.default_rng(0)
    samples = np.array([
        get_response(wavelengths, volts, noise_std=noise_std, rng=rng)
        for _ in range(1000)
    ])
    sample_mean = samples.mean(axis=0)
    sample_var = samples.var(axis=0)

    # Analytic variance of the linear combination of modes
    w_min, w_max = wavelengths.min(), wavelengths.max()
    x = (wavelengths - w_min) / (w_max - w_min) * (2 * np.pi) - np.pi
    i = np.arange(1, len(volts) + 1)[:, None]
    basis = np.cos(i * x)
    expected_var = ((noise_std[:, None] ** 2) * (basis ** 2)).sum(axis=0)

    np.testing.assert_allclose(sample_mean, 0.0, atol=5e-2)
    np.testing.assert_allclose(sample_var, expected_var, rtol=0.2)
