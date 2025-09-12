"""Lightweight schema helpers for the service layer.

Avoids hard dependency on Pydantic for Phase 1. Provides simple normalisers
for wavelength and bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Any
import numpy as np


@dataclass
class WavelengthSpec:
    """Either an explicit array or a linspace spec."""
    array: Optional[Iterable[float]] = None
    start: Optional[float] = None
    stop: Optional[float] = None
    M: Optional[int] = None

    def resolve(self) -> np.ndarray:
        if self.array is not None:
            return np.asarray(self.array, dtype=float).ravel()
        if None in (self.start, self.stop, self.M):
            raise ValueError("WavelengthSpec requires either array or start/stop/M")
        start = float(self.start)  # type: ignore[arg-type]
        stop = float(self.stop)  # type: ignore[arg-type]
        M = int(self.M)  # type: ignore[arg-type]
        if M <= 1 or not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            raise ValueError("Invalid linspace spec for wavelength")
        return np.linspace(start, stop, M)


def normalise_bounds(
    bounds: Optional[Sequence[Tuple[float, float]] | Tuple[float, float]],
    dac_size: int,
) -> Optional[List[Tuple[float, float]]]:
    if bounds is None:
        return None
    # Treat a 2-element sequence of scalars (tuple or list) as a single pair to broadcast.
    try:
        seq0 = list(bounds)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("bounds must be (low, high) or sequence of such") from exc
    if len(seq0) == 2 and not (
        (isinstance(seq0[0], (list, tuple)) and len(seq0[0]) == 2)
        or (isinstance(seq0[1], (list, tuple)) and len(seq0[1]) == 2)
    ):
        lo, hi = float(seq0[0]), float(seq0[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError("Invalid bounds: expected finite low < high.")
        return [(lo, hi) for _ in range(int(dac_size))]

    # Otherwise, expect a sequence of (low, high) pairs with length == dac_size
    if len(seq0) != int(dac_size):
        raise ValueError(
            f"bounds length mismatch: expected {int(dac_size)}, got {len(seq0)}"
        )
    norm: List[Tuple[float, float]] = []
    for i, pair in enumerate(seq0):
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(f"bounds[{i}] must be a (low, high) pair")
        lo, hi = float(pair[0]), float(pair[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError(f"Invalid bounds at index {i}")
        norm.append((lo, hi))
    return norm


__all__ = ["WavelengthSpec", "normalise_bounds"]
