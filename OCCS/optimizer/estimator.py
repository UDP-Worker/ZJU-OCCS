from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
import numpy as np

def make_gp_base_estimator(dimensions, noise_floor=1e-6, n_restarts=10):
    d = len(dimensions)
    ranges = np.array([hi - lo for (lo, hi) in dimensions], dtype=float)

    # ARD 长度尺度的初值与边界（按每维搜索区间尺度设定）
    ls0  = 0.2 * ranges
    ls_lo = 0.05 * ranges
    ls_hi = 2.0  * ranges

    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=ls0,
                                          length_scale_bounds=np.vstack([ls_lo, ls_hi]).T,
                                          nu=2.5) \
             + WhiteKernel(noise_level=noise_floor,
                           noise_level_bounds=(noise_floor, noise_floor * 100.0))

    # 注意：既然用了 WhiteKernel 作为噪声下限，这里 alpha 建议用 0 或极小抖动，避免双计噪声
    return GaussianProcessRegressor(kernel=kernel,
                                    alpha=0.0,
                                    normalize_y=True,
                                    n_restarts_optimizer=n_restarts,
                                    random_state=42)