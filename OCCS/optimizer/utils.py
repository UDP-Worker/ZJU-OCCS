# optimizer/utils.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Iterable

def load_two_row_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取仅两行的CSV或纯文本：第1行=波长，第2行=响应。
    自动适配逗号/空白分隔。返回 (lambda_array, signal_array) ，均为 float64 1D。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ideal waveform file not found: {p}")

    # 尝试逗号分隔
    arr = np.genfromtxt(p.as_posix(), delimiter=",")
    if arr.ndim != 2 or arr.shape[0] < 2:
        # 再尝试空白分隔
        arr = np.genfromtxt(p.as_posix())
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"Expected 2-row data in {p}, got shape {arr.shape}")

    lam = np.asarray(arr[0, :], dtype=np.float64).ravel()
    sig = np.asarray(arr[1, :], dtype=np.float64).ravel()
    if lam.size != sig.size:
        raise ValueError("Lambda and signal lengths differ in the CSV.")
    return lam, sig

def parse_two_row_array(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将硬件返回的数组解析为 (lambda, signal)。
    支持 (2, N) 或 (N, 2)，其他形状报错。
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")

    if a.shape[0] == 2:
        lam, sig = a[0, :], a[1, :]
    elif a.shape[1] == 2:
        lam, sig = a[:, 0], a[:, 1]
    else:
        raise ValueError(f"Expected shape (2,N) or (N,2), got {a.shape}")
    lam = np.asarray(lam, dtype=np.float64).ravel()
    sig = np.asarray(sig, dtype=np.float64).ravel()
    if lam.size != sig.size:
        raise ValueError("Lambda and signal lengths differ.")
    return lam, sig

def resample_to_ref(lambda_raw: np.ndarray,
                    s_raw: np.ndarray,
                    lambda_ref: np.ndarray) -> np.ndarray:
    """
    用一维线性插值把 (lambda_raw, s_raw) 重采样到 lambda_ref。
    """
    lambda_raw = np.asarray(lambda_raw, dtype=np.float64).ravel()
    s_raw = np.asarray(s_raw, dtype=np.float64).ravel()
    if lambda_raw.size != s_raw.size:
        raise ValueError("lambda_raw and s_raw must have the same length.")
    return np.interp(lambda_ref, lambda_raw, s_raw)

def normalize_shape(s: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    形状归一：去中位基线 + L2 归一，使度量更关注形状。
    """
    s = np.asarray(s, dtype=np.float64).ravel()
    s0 = s - np.median(s)
    nrm = np.linalg.norm(s0)
    return s0 / (nrm + eps)

def huber_loss(e: np.ndarray, kappa: float) -> np.ndarray:
    """
    Huber 损失的逐点值；kappa 为二/一次转折点（在归一尺度下建议 0.5~1.0）。
    """
    a = np.abs(e)
    quad = 0.5 * (e ** 2)
    lin = kappa * (a - 0.5 * kappa)
    return np.where(a <= kappa, quad, lin)

def make_band_weights(lambda_ref: np.ndarray,
                      passband: Optional[Tuple[float, float]] = None,
                      transition: Optional[Tuple[float, float]] = None,
                      stopband: Optional[Tuple[float, float]] = None,
                      weights: Tuple[float, float, float] = (3.0, 2.0, 1.0)) -> np.ndarray:
    """
    根据波长区间给出分区权重：通带>过渡带>阻带。
    任意一个区间可以留空（则其权重不会被赋值）。
    """
    lam = np.asarray(lambda_ref, dtype=np.float64).ravel()
    w = np.ones_like(lam, dtype=np.float64)

    if passband is not None:
        m = (lam >= passband[0]) & (lam <= passband[1])
        w[m] = weights[0]
    if transition is not None:
        m = (lam >= transition[0]) & (lam <= transition[1])
        w[m] = weights[1]
    if stopband is not None:
        m = (lam >= stopband[0]) & (lam <= stopband[1])
        w[m] = weights[2]
    return w
