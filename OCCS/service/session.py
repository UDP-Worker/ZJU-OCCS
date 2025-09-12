"""Optimization session management for the web service.

Phase 1 scope: construct hardware + objective + optimizer, provide basic
methods to apply manual voltages and fetch current waveform. The iterative
optimisation loop and streaming live updates will be added in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, List, Any, Dict, Iterable
from pathlib import Path
import numpy as np

from OCCS.service.hardware import make_hardware
from OCCS.optimizer.objective import create_hardware_objective, HardwareObjective
from OCCS.optimizer.optimizer import BayesianOptimizer


@dataclass
class OptimizerSession:
    backend: str
    dac_size: int
    wavelength: np.ndarray
    bounds: Optional[List[Tuple[float, float]]] = None
    target_csv_path: Optional[Path] = None
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Initialized fields
    hardware: Any = field(init=False)
    hw_objective: HardwareObjective = field(init=False)
    optimizer: BayesianOptimizer = field(init=False)
    history: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # Create hardware
        self.hardware = make_hardware(
            self.backend,
            dac_size=int(self.dac_size),
            wavelength=self.wavelength,
            bounds=self.bounds,
        )

        # Objective: if not provided, fall back to built-in ideal waveform
        target_csv = (
            Path(self.target_csv_path)
            if self.target_csv_path is not None
            else Path(__file__).resolve().parent.parent / "data" / "ideal_waveform.csv"
        )

        # Use same number of points as hardware wavelength for reference grid
        self.hw_objective = create_hardware_objective(
            self.hardware,
            target_csv_path=target_csv,
            M=int(self.wavelength.size),
        )

        # Construct optimizer with bounds from hardware if available
        dims = getattr(self.hardware, "skopt_dimensions", None)
        dimensions = dims if dims is not None else self.bounds
        self.optimizer = BayesianOptimizer(
            self.hw_objective,
            dimensions=dimensions,
            **dict(self.optimizer_kwargs),
        )

    # ---- Basic operations (Phase 1) ----
    def apply_manual(self, volts: Iterable[float]) -> None:
        arr = np.asarray(list(volts), dtype=float).ravel()
        if arr.size != int(self.dac_size):
            raise ValueError(
                f"Expected {self.dac_size} voltage values, got {arr.size}"
            )
        self.hardware.apply_voltage(arr)

    def read_waveform(self) -> Dict[str, Any]:
        signal = self.hardware.get_response()
        lam = np.asarray(self.wavelength, dtype=float)
        # Resample target to wavelength grid via the objective itself
        # (Call with identical grid to get aligned s_ref and use target from diag)
        _, diag = self.hw_objective.curve_obj(lam, signal)
        target = np.asarray(diag.get("target_norm", diag.get("s_ref", [])), dtype=float)
        return {
            "lambda": lam,
            "signal": np.asarray(signal, dtype=float),
            "target": target if target.size == lam.size else np.asarray([], dtype=float),
        }

    def status(self) -> Dict[str, Any]:
        best_loss = (
            float(np.min([h.get("loss", np.inf) for h in self.history]))
            if self.history
            else None
        )
        return {
            "running": False,
            "iter": len(self.history),
            "best_loss": best_loss,
        }

    def close(self) -> None:
        # Placeholder for resource cleanup (if needed for real hardware)
        pass


__all__ = ["OptimizerSession"]

