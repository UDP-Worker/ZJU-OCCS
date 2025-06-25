"""基于差分进化的参数优化算法"""

import numpy as np
from types import SimpleNamespace

try:
    from scipy.optimize import differential_evolution
except Exception:  # pragma: no cover - SciPy may be unavailable
    differential_evolution = None

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None

from .simulation import optical_simulation


def objective_function(
        params: np.ndarray,
        target_spectrum_db: np.ndarray,
        frequency_f: np.ndarray,
        t: float,
        w_range: np.ndarray,
        H1: np.ndarray,
        H3: np.ndarray,
        n_ku: int,
        m_kl: int,
) -> float:
    """
    自定义损失函数：
    ─ 通带：加权均方误差 (MSE)
    ─ 阻带：对 -20 dB 阈值以上的“超出部分”施加对数障碍惩罚
    """
    # 得到被优化结构的实际响应（dB）
    simulated_db = optical_simulation(params, t, w_range, H1, H3, n_ku, m_kl)
    simulated_db = np.clip(simulated_db, -200.0, 200.0)

    # 按 “目标响应的最大值” 自动划分通带 / 阻带
    #    这里假设目标在通带全部取同一峰值（典型设置 0 dB）
    pb_level = np.max(target_spectrum_db)
    passband_mask = np.isclose(target_spectrum_db, pb_level, atol=1e-6)
    stopband_mask = ~passband_mask

    # ---------- 通带部分：均方误差 ----------
    mse_passband = np.mean(
        (simulated_db[passband_mask] - target_spectrum_db[passband_mask]) ** 2
    ) if np.any(passband_mask) else 0.0

    # ---------- 阻带部分：对数障碍 ----------
    # 只惩罚高于 -20 dB 的“漏光”点，保留 2 dB 缓冲空间
    leakage_threshold_db = -22.0
    excess = simulated_db[stopband_mask] - leakage_threshold_db        # 正值才违规
    excess_pos = np.maximum(excess, 0.0)                               # 所有负值置零
    barrier = np.log1p(excess_pos) ** 2
    penalty_stopband = np.mean(barrier) if barrier.size else 0.0

    # ③ 组合损失（权重可按需要调整）
    passband_weight = 10.0
    stopband_weight = 1.0
    loss = passband_weight * mse_passband + stopband_weight * penalty_stopband

    return float(loss)

def optimize_params(
    bounds,
    args,
    maxiter: int = 300,
    popsize: int = 20,
) -> differential_evolution:
    """运行差分进化优化器并返回结果."""
    result = differential_evolution(
        objective_function,
        bounds,
        args=args,
        maxiter=maxiter,
        popsize=popsize,
        disp=True,
    )
    return result


def _ensure_torch():
    if torch is None:
        raise ImportError("PyTorch is required for SGD optimization")


def mrr_transfer_function_torch(w: "torch.Tensor", t: float, k: "torch.Tensor", phi_offset: float) -> "torch.Tensor":
    j = 1j
    numerator = torch.sqrt(1 - k) - t ** 2 * torch.exp(-j * (2 * w + phi_offset))
    denominator = 1 - t ** 2 * torch.sqrt(1 - k) * torch.exp(-j * (2 * w + phi_offset))
    denominator = denominator + 1e-12  # 避免分母为零
    return numerator / denominator


def delay_line_torch(w: "torch.Tensor", t: float, delay: float, phi_c: float) -> "torch.Tensor":
    j = 1j
    return t * torch.exp(-j * w * delay - j * phi_c)


def optical_simulation_torch(
    params: "torch.Tensor",
    t: float,
    w_range: np.ndarray,
    H1: np.ndarray,
    H3: np.ndarray,
    n_ku: int,
    m_kl: int,
) -> "torch.Tensor":
    device = params.device
    dtype_c = torch.complex128
    w = torch.tensor(w_range, dtype=torch.float64, device=device)
    H1_t = torch.tensor(H1, dtype=dtype_c, device=device)
    H3_t = torch.tensor(H3, dtype=dtype_c, device=device)

    ku_params = params[:n_ku]
    kl_params = params[n_ku:]

    if n_ku > 0:
        au_list = [mrr_transfer_function_torch(w, t, k, phi_offset=np.pi) for k in ku_params]
        Au = torch.prod(torch.stack(au_list), dim=0)
    else:
        Au = torch.ones_like(w, dtype=dtype_c, device=device)

    if m_kl > 0:
        al_list = [mrr_transfer_function_torch(w, t, k, phi_offset=np.pi) for k in kl_params]
        Al_mrr = torch.prod(torch.stack(al_list), dim=0)
    else:
        Al_mrr = torch.ones_like(w, dtype=dtype_c, device=device)

    Al = Al_mrr * delay_line_torch(w, t, delay=1.0, phi_c=0.0)

    H2_stack = torch.zeros((w.shape[0], 2, 2), dtype=dtype_c, device=device)
    H2_stack[:, 0, 0] = Au
    H2_stack[:, 1, 1] = Al

    H_final = H1_t.unsqueeze(0).matmul(H2_stack).matmul(H3_t.unsqueeze(0))
    H11 = H_final[:, 0, 0]

    magnitude = torch.abs(H11)
    magnitude = torch.clamp(magnitude, min=1e-12)
    return 20 * torch.log10(magnitude)


def objective_function_torch(
    params: "torch.Tensor",
    target_spectrum_db: np.ndarray,
    frequency_f: np.ndarray,
    t: float,
    w_range: np.ndarray,
    H1: np.ndarray,
    H3: np.ndarray,
    n_ku: int,
    m_kl: int,
) -> "torch.Tensor":
    target = torch.tensor(target_spectrum_db, dtype=torch.float64, device=params.device)
    simulated_db = optical_simulation_torch(params, t, w_range, H1, H3, n_ku, m_kl)
    simulated_db = torch.clamp(simulated_db, min=-200.0, max=200.0)

    pb_level = torch.max(target)
    passband_mask = torch.isclose(target, pb_level, atol=1e-6)
    stopband_mask = ~passband_mask

    if torch.any(passband_mask):
        mse_passband = torch.mean((simulated_db[passband_mask] - target[passband_mask]) ** 2)
    else:
        mse_passband = torch.tensor(0.0, device=params.device)

    leakage_threshold_db = -22.0
    excess = simulated_db[stopband_mask] - leakage_threshold_db
    excess_pos = torch.clamp(excess, min=0.0)
    barrier = torch.log1p(excess_pos) ** 2
    penalty_stopband = torch.mean(barrier) if barrier.numel() else torch.tensor(0.0, device=params.device)

    passband_weight = 10.0
    stopband_weight = 1.0
    loss = passband_weight * mse_passband + stopband_weight * penalty_stopband

    return loss


def optimize_params_sgd(
    bounds,
    args,
    maxiter: int = 300,
    popsize: int = 20,
):
    """基于 Adam 的梯度下降优化参数，接口与 :func:`optimize_params` 相同."""
    _ensure_torch()

    (target_spectrum, frequency_f, t, w_range, H1, H3, n_ku, m_kl) = args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 在合理区间附近初始化，避免初始值过于极端
    init = 0.5 + 0.1 * torch.randn(len(bounds), dtype=torch.float64, device=device)
    params = init.clamp(0.0, 1.0).detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([params], lr=0.05)

    def _progress_bar(iter_idx: int) -> None:
        bar_len = 30
        filled = int(bar_len * (iter_idx + 1) / maxiter)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r[{bar}] {iter_idx + 1}/{maxiter}", end="")

    best_loss = float("inf")
    best_params = params.detach().clone()

    for i in range(maxiter):
        optimizer.zero_grad()
        loss = objective_function_torch(
            params,
            target_spectrum,
            frequency_f,
            t,
            w_range,
            H1,
            H3,
            n_ku,
            m_kl,
        )
        if torch.isnan(loss):
            break
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for idx, (low, high) in enumerate(bounds):
                params[idx].clamp_(low, high)  # 投影回参数范围

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = params.detach().clone()

        _progress_bar(i)

        if best_loss < 1e-8:
            break
    print()
    return SimpleNamespace(x=best_params.cpu().numpy())
