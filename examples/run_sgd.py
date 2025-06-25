"""SGD 优化示例，与 ``run_evolution.py`` 参数一致"""

import numpy as np
from opt_sim.components import (
    tunable_mzi_in,
    tunable_mzi_out,
    phase_shifter_matrix,
)
from opt_sim.reference import create_reference_box_filter
from opt_sim.simulation import optical_simulation
from opt_sim.optimization import optimize_params_sgd


if __name__ == "__main__":
    n_ku = 3
    m_kl = 2

    f_center = 193.1e12
    fsr = 100e9
    s = 10
    t = 0.979888

    theta_i = np.pi / 2
    theta_o = np.pi / 2
    phi_t = 0.0
    phi_b = 0.0

    w1 = -20 * np.pi
    w2 = 20 * np.pi
    dw = 0.006285
    w_range = np.arange(w1, w2, dw)
    len_w = len(w_range)
    frequency_f = np.linspace(f_center - s * fsr, f_center + s * fsr, len_w)

    Hi = tunable_mzi_in(theta_i)
    Hp = phase_shifter_matrix(phi_t, phi_b)
    H1 = Hi @ Hp
    H3 = tunable_mzi_out(theta_o)

    target_spectrum = create_reference_box_filter(
        frequency_array=frequency_f,
        center_freq=f_center,
        fsr=fsr,
        bandwidth=50e9,
        passband_level_db=0,
        stopband_level_db=-40,
    )

    total_params = n_ku + m_kl
    bounds = [(0, 1)] * total_params

    args = (target_spectrum, frequency_f, t, w_range, H1, H3, n_ku, m_kl)

    result = optimize_params_sgd(bounds, args)
    best_params = result.x
    final_spectrum = optical_simulation(best_params, t, w_range, H1, H3, n_ku, m_kl)

    print("最优参数:")
    for i in range(n_ku):
        print(f"  ku{i+1} = {best_params[i]:.4f}")
    for i in range(m_kl):
        print(f"  kl{i+1} = {best_params[n_ku + i]:.4f}")

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(
        frequency_f,
        target_spectrum,
        label="Ideal Reference Box-Filter",
        linestyle="--",
        linewidth=2.5,
        color="red",
    )
    plt.plot(frequency_f, final_spectrum, label="Final Spectrum (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Target vs Final Spectrum")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

