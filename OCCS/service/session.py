"""Optimization session management for the web service.

Phase 1 scope: construct hardware + objective + optimizer, provide basic
methods to apply manual voltages and fetch current waveform. The iterative
optimisation loop and streaming live updates will be added in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, List, Any, Dict, Iterable
import threading
import asyncio
from pathlib import Path
import numpy as np

from OCCS.service.hardware import make_hardware
from OCCS.optimizer.objective import create_hardware_objective, HardwareObjective
from OCCS.optimizer.optimizer import BayesianOptimizer
from OCCS.optimizer.estimator import make_gp_base_estimator


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
    optimizer_config: Dict[str, Any] = field(default_factory=dict, init=False)
    history: List[Dict[str, Any]] = field(default_factory=list, init=False)
    running: bool = field(default=False, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _stop_evt: threading.Event = field(default_factory=threading.Event, init=False)
    _subscribers: List[Tuple[asyncio.AbstractEventLoop, "asyncio.Queue[Dict[str, Any]]"]] = field(
        default_factory=list, init=False
    )
    best_loss: Optional[float] = field(default=None, init=False)
    best_x: Optional[np.ndarray] = field(default=None, init=False)

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
        # 目标函数：默认完全采用 CSV 的原始波长网格（更稳妥，避免插值/外推造成的异常）
        self.hw_objective = create_hardware_objective(
            self.hardware,
            target_csv_path=target_csv,
            M=None,
        )

        # Optimizer configuration mirrors the integration tests defaults
        self.optimizer_config = dict(self.optimizer_kwargs)
        self.optimizer_config.setdefault("acq_func", "gp_hedge")
        self.optimizer_config.setdefault("random_state", 42)
        self._build_optimizer()

    def _resolve_dimensions(self) -> List[Tuple[float, float]]:
        dims = getattr(self.hardware, "skopt_dimensions", None)
        if dims is not None:
            return [tuple(map(float, d)) for d in dims]
        if self.bounds is not None:
            return [tuple(map(float, b)) for b in self.bounds]
        return [(-1.0, 1.0) for _ in range(int(self.dac_size))]

    def _build_optimizer(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        dimensions = self._resolve_dimensions()
        config = dict(self.optimizer_config)
        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    config[key] = value
        builder_kwargs = dict(config)
        try:
            builder_kwargs["base_estimator"] = make_gp_base_estimator(
                dimensions=dimensions,
                noise_floor=1e-6,
                n_restarts=10,
            )
        except Exception:
            pass
        self.optimizer = BayesianOptimizer(
            self.hw_objective,
            dimensions=dimensions,
            **builder_kwargs,
        )
        # Keep non-callable configuration for future rebuilds (avoid storing estimator instance)
        config.pop("base_estimator", None)
        self.optimizer_config = config

    # ---- Basic operations (Phase 1) ----
    def apply_manual(self, volts: Iterable[float]) -> None:
        arr = np.asarray(list(volts), dtype=float).ravel()
        if arr.size != int(self.dac_size):
            raise ValueError(
                f"Expected {self.dac_size} voltage values, got {arr.size}"
            )
        self.hardware.apply_voltage(arr)

    def read_waveform(self) -> Dict[str, Any]:
        # 获取当前硬件波形，并通过目标函数在参考网格上对齐/重采样
        raw_signal = self.hardware.get_response()
        lam_in = np.asarray(self.wavelength, dtype=float)
        # 使用曲线目标的调用来得到对齐后的诊断（含 lambda_ref/s_ref/target_norm）
        _, diag = self.hw_objective.curve_obj(lam_in, raw_signal)
        lam_ref = np.asarray(diag.get("lambda_ref", lam_in), dtype=float)
        s_ref = np.asarray(diag.get("s_ref", raw_signal), dtype=float)
        t_ref = np.asarray(diag.get("target_norm", []), dtype=float)
        # 仅当目标长度匹配参考网格时返回目标，以避免前端绘图异常
        target = t_ref if t_ref.size == lam_ref.size else np.asarray([], dtype=float)
        return {"lambda": lam_ref, "signal": s_ref, "target": target}

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
        self.stop_optimize()

    # ---- Realtime streaming helpers ----
    def add_subscriber(self, loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[Dict[str, Any]]") -> None:
        self._subscribers.append((loop, queue))

    def remove_subscriber(self, queue: "asyncio.Queue[Dict[str, Any]]") -> None:
        self._subscribers = [(lp, q) for (lp, q) in self._subscribers if q is not queue]

    def _emit(self, event: Dict[str, Any]) -> None:
        # Thread-safe enqueue into each subscriber's asyncio.Queue
        for loop, q in list(self._subscribers):
            try:
                asyncio.run_coroutine_threadsafe(q.put(event), loop)
            except Exception:
                # Drop subscriber on error
                try:
                    self._subscribers.remove((loop, q))
                except ValueError:
                    pass

    # ---- Optimization control (Phase 2) ----
    def start_optimize(
        self,
        *,
        n_calls: int,
        x0: Optional[Iterable[float]] = None,
        acq_func: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if self.running:
            raise RuntimeError("Optimization already running for this session")

        overrides: Dict[str, Any] = {}
        if acq_func is not None:
            overrides["acq_func"] = acq_func
        if random_state is not None:
            overrides["random_state"] = int(random_state)
        self._build_optimizer(overrides if overrides else None)

        # Reset session history to mirror the integration test environment
        self.history.clear()
        self.best_loss = None
        self.best_x = None
        self._stop_evt.clear()
        self.running = True

        # Pre-emit status so subscribers know a run has started
        self._emit({
            "type": "status",
            "running": True,
            "iter": 0,
            "best_loss": None,
        })

        def _run_loop() -> None:
            try:
                if x0 is None:
                    x0_arr = np.zeros(int(self.dac_size), dtype=float)
                else:
                    x0_arr = np.asarray(list(x0), dtype=float).ravel()
                if x0_arr.size != int(self.dac_size):
                    raise ValueError(f"Expected {self.dac_size} initial voltage values, got {x0_arr.size}")

                local_best = float("inf")

                def _handle_step(info: Dict[str, Any]) -> bool:
                    nonlocal local_best

                    diag = info.get("diag", {})
                    x_vec = np.asarray(info.get("x", []), dtype=float).ravel()
                    if x_vec.size != int(self.dac_size):
                        x_vec = np.asarray(diag.get("volts", x_vec), dtype=float).ravel()
                    loss_val = float(info.get("loss", float("inf")))
                    self.history.append({"x": x_vec, "loss": loss_val, "diag": diag})

                    best_loss_payload = info.get("best_loss")
                    if best_loss_payload is not None and np.isfinite(best_loss_payload):
                        local_best = float(best_loss_payload)
                        if self.best_loss is None or local_best < float(self.best_loss):
                            self.best_loss = local_best
                    else:
                        local_best = min(local_best, loss_val)
                        if not np.isfinite(local_best):
                            local_best = float(loss_val)

                    best_x_payload = info.get("best_x")
                    if best_x_payload is not None:
                        self.best_x = np.asarray(best_x_payload, dtype=float)

                    lam = np.asarray(diag.get("lambda_ref", self.wavelength), dtype=float)
                    s_ref = np.asarray(diag.get("s_ref", []), dtype=float)
                    t_ref = np.asarray(diag.get("target_norm", []), dtype=float)
                    self._emit({
                        "type": "waveform",
                        "lambda": lam.tolist(),
                        "signal": s_ref.tolist() if s_ref.size else [],
                        "target": t_ref.tolist() if t_ref.size else [],
                    })

                    progress_payload = {
                        "type": "progress",
                        "iter": len(self.history),
                        "loss": float(loss_val),
                        "running_min": local_best if np.isfinite(local_best) else None,
                        "xi": float(diag.get("xi", np.nan)) if "xi" in diag else None,
                        "kappa": float(diag.get("kappa", np.nan)) if "kappa" in diag else None,
                        "gp_max_std": float(diag.get("gp_max_std", np.nan)) if "gp_max_std" in diag else None,
                        "x": list(map(float, x_vec)) if x_vec.size else [],
                        "best_x": list(map(float, np.asarray(self.best_x))) if self.best_x is not None else None,
                    }
                    self._emit(progress_payload)
                    return not self._stop_evt.is_set()

                result = self.optimizer.run(
                    n_calls=int(n_calls),
                    x0=x0_arr,
                    callback=_handle_step,
                )

                res_best_loss = result.get("best_loss")
                if res_best_loss is not None and np.isfinite(res_best_loss):
                    best_loss_val = float(res_best_loss)
                    if self.best_loss is None or best_loss_val < float(self.best_loss):
                        self.best_loss = best_loss_val
                res_best_x = result.get("best_x")
                if res_best_x is not None:
                    self.best_x = np.asarray(res_best_x, dtype=float)

                final_status = {
                    "type": "status",
                    "running": False,
                    "iter": len(self.history),
                    "best_loss": (self.best_loss if self.best_loss is not None else None),
                }
                if self.best_x is not None:
                    final_status["x"] = list(map(float, np.asarray(self.best_x)))
                self._emit(final_status)
                self._emit({"type": "done", "best_loss": (self.best_loss if self.best_loss is not None else None)})
            except Exception as e:
                self._emit({"type": "error", "message": str(e)})
            finally:
                self.running = False

        # Heuristic: for tiny jobs and no subscribers, run inline to be deterministic in tests
        if int(n_calls) <= 2 and len(self._subscribers) == 0:
            _run_loop()
        else:
            self._thread = threading.Thread(target=_run_loop, name="opt-runner", daemon=True)
            self._thread.start()

    def stop_optimize(self) -> None:
        if not self.running:
            return
        self._stop_evt.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)
        self.running = False


__all__ = ["OptimizerSession"]
