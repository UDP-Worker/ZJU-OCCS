"""Hardware factory and lightweight adapter utilities for the web service.

Notes
-----
- Both MockHardware and RealHardware expose a `get_response()` method name.
  RealHardware remains a placeholder (NotImplemented) at this stage. The
  factory will reject attempts to construct a real backend unless explicitly
  enabled by the caller.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Any
import os
import numpy as np

from OCCS.connector import MockHardware, RealHardware  # type: ignore


def _broadcast_bounds(
    bounds: Optional[Sequence[Tuple[float, float]] | Tuple[float, float]],
    dac_size: int,
) -> Optional[List[Tuple[float, float]]]:
    if bounds is None:
        return None
    # Allow a single pair to be broadcast to all channels
    if isinstance(bounds, tuple) and len(bounds) == 2 and not any(
        isinstance(b, (list, tuple)) and len(b) == 2  # type: ignore[truthy-bool]
        for b in bounds
    ):
        lo, hi = float(bounds[0]), float(bounds[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError("Invalid bounds: expected finite low < high.")
        return [(lo, hi) for _ in range(int(dac_size))]

    try:
        seq = list(bounds)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("bounds must be (low, high) or sequence of such") from exc
    if len(seq) != int(dac_size):
        raise ValueError(
            f"bounds length mismatch: expected {int(dac_size)}, got {len(seq)}"
        )
    norm: List[Tuple[float, float]] = []
    for i, pair in enumerate(seq):
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(f"bounds[{i}] must be a (low, high) pair")
        lo, hi = float(pair[0]), float(pair[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError(f"Invalid bounds at index {i}: low < high and finite required")
        norm.append((lo, hi))
    return norm


def list_backends() -> List[Dict[str, Any]]:
    """List available backend types for selection in the UI.

    Returns a list of dicts with `name` and `available` flags. Real hardware
    availability can be toggled via environment variable `OCCS_REAL_AVAILABLE=1`.
    """
    real_available = os.environ.get("OCCS_REAL_AVAILABLE", "0") in {"1", "true", "True"}
    return [
        {"name": "mock", "available": True},
        {"name": "real", "available": bool(real_available)},
    ]


def make_hardware(
    backend: str,
    *,
    dac_size: int,
    wavelength: Iterable[float],
    bounds: Optional[Sequence[Tuple[float, float]] | Tuple[float, float]] = None,
    noise_std: Optional[Iterable[float] | float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Any:
    """Construct a hardware instance by backend name.

    Parameters
    ----------
    backend: "mock" | "real"
    dac_size: Number of channels.
    wavelength: Wavelength grid used by the hardware.
    bounds: Optional per-channel bounds. Single pair is broadcast to all.
    noise_std: Optional noise std (mock only).
    rng: Optional RNG (mock only).
    """
    name = str(backend).strip().lower()
    if name == "mock":
        return MockHardware(
            dac_size=int(dac_size),
            wavelength=wavelength,
            noise_std=noise_std,
            rng=rng,
            voltage_bounds=_broadcast_bounds(bounds, int(dac_size)),
        )
    if name == "real":
        real_available = os.environ.get("OCCS_REAL_AVAILABLE", "0") in {"1", "true", "True"}
        if not real_available:
            raise NotImplementedError("Real hardware backend is disabled or not available")
        # The RealHardware placeholder raises NotImplemented in its __init__ by design.
        return RealHardware(
            dac_size=int(dac_size),
            wavelength=wavelength,
            voltage_bounds=_broadcast_bounds(bounds, int(dac_size)),
        )
    raise ValueError(f"Unknown backend: {backend!r}")


__all__ = ["list_backends", "make_hardware"]

