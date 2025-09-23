"""Bayesian optimization wrapper around scikit-optimize for OCCS hardware."""

import logging
from typing import Optional, Sequence, Tuple, Any, Dict, List, Callable

import numpy as np
from skopt import Optimizer

from OCCS.optimizer.objective import HardwareObjective

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """Manage suggest-observe loop and basic diagnostics.

    Wraps a ``skopt.Optimizer`` configured with voltage bounds and offers
    convenience helpers to run multiple iterations while tracking simple
    diagnostics like an estimate of the model's maximum uncertainty.
    """

    # ------------------ Adaptive exploration defaults (minimal & local) ------------------
    _XI_MIN: float = 1e-3         # lower bound for xi (or mapped to kappa if LCB/UCB)
    _XI_MAX: float = 5e-1         # upper bound for xi
    _C_INIT: float = 5e-2         # base schedule xi_base = max(xi_min, C / sqrt(t))
    _STALL_K: int = 3             # consecutive no-improve rounds to trigger boost
    _BOOST_BETA: float = 3.0      # stagnation boost factor for exploration
    _COOL_GAMMA: float = 0.8      # shrink factor after an improvement
    _DELTA_TOL: float = 1e-12     # rebuild tolerance for xi/kappa changes

    def __init__(
            self,
            hardware_objective: HardwareObjective,
            dimensions: Optional[Sequence[Tuple[float, float]]] = None,
            **skopt_kwargs: Any,
    ) -> None:
        """
        Thin wrapper holding a skopt.Optimizer configured with hardware bounds.

        Parameters
        ----------
        hardware_objective:
            Coupled objective providing access to hardware and loss function.
        dimensions:
            skopt-compatible bounds. If None, tries to read from
            ``hardware_objective.hardware.voltage_bounds``.
        skopt_kwargs:
            Additional kwargs forwarded to ``skopt.Optimizer``.
        """
        self.hardware_objective = hardware_objective
        if dimensions is None:
            dimensions = getattr(hardware_objective.hardware, "voltage_bounds", None)
        self.dimensions = dimensions

        # Keep a copy of constructor kwargs so we can "soft rebuild" Optimizer with updated
        # acq_func_kwargs while preserving all other settings (base_estimator, seeds, etc).
        self._skopt_kwargs: Dict[str, Any] = dict(skopt_kwargs)  # shallow copy is enough
        # Canonicalize acquisition function name to what skopt expects
        self._acq_func: str = self._canonicalize_acq_func(self._skopt_kwargs.get("acq_func", "EI"))
        # Ensure kwargs carry canonical form to keep behaviour stable across (re)builds
        self._skopt_kwargs["acq_func"] = self._acq_func

        self._opt: Optional[Optimizer] = None
        if self.dimensions is not None:
            # Ensure acq_func is explicitly set (keeps behaviour stable across soft rebuilds)
            if "acq_func" not in self._skopt_kwargs:
                self._skopt_kwargs["acq_func"] = self._acq_func or "EI"
            self._opt = Optimizer(dimensions=self.dimensions, **self._skopt_kwargs)

        # ---- State for incremental (step-by-step) adaptive exploration ----
        self._t_seen: int = 0
        self._best_loss_seen: float = float("inf")
        self._no_improve: int = 0
        self._param_value_cache: Optional[float] = None  # last used xi/kappa

    # ------------------ Small helpers for soft rebuilding & exploration param ------------
    def _get_acq_kwargs(self) -> Dict[str, Any]:
        """Return a shallow copy of current acq_func_kwargs (may be absent)."""
        ak = self._skopt_kwargs.get("acq_func_kwargs", None)
        return dict(ak) if isinstance(ak, dict) else {}

    def _canonicalize_acq_func(self, acq: Any) -> str:
        """Map various spellings/aliases to skopt-accepted acquisition names.

        Skopt accepted values (as of tests):
        - "gp_hedge" (lowercase only)
        - "EI", "PI", "LCB", "MES", "PVRS"
        - "EIps", "PIps"
        Also treat "UCB" as alias for "LCB" to select the kappa branch.
        """
        s = str(acq)
        sl = s.lower()
        if sl == "gp_hedge":
            return "gp_hedge"
        if sl == "eips":
            return "EIps"
        if sl == "pips":
            return "PIps"
        su = s.upper()
        if su == "UCB":
            return "LCB"
        if su in {"EI", "PI", "LCB", "MES", "PVRS"}:
            return su
        return s

    def _choose_param_name(self) -> str:
        """
        Decide which exploration parameter to adapt.
        - EI/PI/gp_hedge: adapt 'xi'
        - LCB/UCB: adapt 'kappa'
        """
        if self._acq_func in ("LCB",):
            return "kappa"
        # For EI / PI / gp_hedge → use xi
        return "xi"

    def _default_param_value(self, name: str) -> float:
        """Provide a gentle default if the user didn't set one."""
        if name == "kappa":
            # Common UCB default in skopt is 1.96; keep it gentle.
            return 1.96
        # xi for EI/PI/gp_hedge
        return 0.05

    def _build_optimizer_with(self, new_acq_kwargs: Dict[str, Any]) -> Optimizer:
        """Create a fresh Optimizer with updated acq_func_kwargs, keeping other kwargs."""
        kwargs = dict(self._skopt_kwargs)
        merged = self._get_acq_kwargs()
        merged.update(new_acq_kwargs or {})
        kwargs["acq_func_kwargs"] = merged
        kwargs["acq_func"] = self._acq_func  # keep stable across rebuilds
        # We are backfilling full history via tell(...), so skip skopt's initial design
        # to avoid repeating the same initial samples after each soft rebuild.
        kwargs["n_initial_points"] = 0
        # Keep our copy in sync so subsequent reads reflect the latest value
        self._skopt_kwargs["acq_func_kwargs"] = dict(merged)
        return Optimizer(dimensions=self.dimensions, **kwargs)

    @property
    def optimizer(self) -> Optimizer:
        """Return the underlying ``skopt.Optimizer`` instance."""
        if self._opt is None:
            raise RuntimeError("Optimizer not initialised: missing dimensions/bounds.")
        return self._opt

    def suggest(self) -> np.ndarray:
        """Ask skopt for the next candidate voltage vector."""
        x = self.optimizer.ask()
        return np.asarray(x, dtype=float)

    def observe(self, x: Sequence[float], y: float) -> None:
        """Report observation back to skopt."""
        self.optimizer.tell(list(map(float, x)), float(y))

    def step(self, x: Optional[Sequence[float]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the objective at ``x`` (or at a suggested point if None),
        report it to skopt and return (loss, diag).
        """
        # Decide which parameter we are adapting
        param_name = self._choose_param_name()  # 'xi' or 'kappa'

        # Snapshot history before this observation (to backfill on rebuild)
        try:
            prev_Xi = list(getattr(self.optimizer, "Xi", []))
            prev_yi = list(getattr(self.optimizer, "yi", []))
        except Exception:
            prev_Xi, prev_yi = [], []
        prior_best = float(np.min(prev_yi)) if len(prev_yi) > 0 else float("inf")

        # Evaluate objective (at suggested x if not provided) and report to skopt
        if x is None:
            x = self.suggest()
        x = np.asarray(x, dtype=float)
        loss, diag = self.hardware_objective(x)
        self.observe(x, loss)

        # Update counters
        try:
            self._t_seen = int(len(prev_yi) + 1)
        except Exception:
            self._t_seen = max(1, int(self._t_seen) + 1)

        # Enrich diagnostics to match what run() provides
        try:
            # Estimate current maximum predictive std/var over the space
            max_std, max_var = self._compute_gp_max_uncertainty(n_samples=1024, seed=12345)
        except Exception:
            max_std, max_var = float("nan"), float("nan")
        diag["gp_max_std"] = max_std
        diag["gp_max_var"] = max_var

        # Current exploration parameter value used during this iteration
        if self._param_value_cache is None:
            ak0 = self._get_acq_kwargs()
            self._param_value_cache = float(ak0.get(param_name, self._default_param_value(param_name)))
        param_value_now = float(self._param_value_cache)
        diag[param_name] = param_value_now

        # Adaptive exploration update for the NEXT iteration
        improved = (float(prior_best) - float(loss)) > max(0.01 * abs(float(prior_best)), 1e-12)
        if improved:
            self._best_loss_seen = float(loss)
            self._no_improve = 0
            new_val = max(self._XI_MIN, param_value_now * self._COOL_GAMMA)
        else:
            self._no_improve += 1
            # base schedule at t (1-based)
            base = max(self._XI_MIN, self._C_INIT / (self._t_seen ** 0.5))
            if self._no_improve >= self._STALL_K:
                new_val = min(self._BOOST_BETA * base, self._XI_MAX)
            else:
                new_val = base

        # Rebuild optimizer with updated exploration parameter if it changed materially
        if abs(new_val - param_value_now) > self._DELTA_TOL:
            try:
                ak = self._get_acq_kwargs()
                ak[param_name] = float(new_val)
                if self._acq_func == "gp_hedge":
                    # keep both present for hedge mixtures
                    ak.setdefault("xi", float(new_val))
                    ak.setdefault("kappa", 1.96)
                new_opt = self._build_optimizer_with(ak)
                # Backfill with history BEFORE this observation (keeps parity with run())
                if prev_Xi and prev_yi:
                    new_opt.tell(prev_Xi, prev_yi)
                self._opt = new_opt
                self._param_value_cache = float(new_val)
            except Exception:
                # If rebuild fails, keep current optimizer and cache unchanged
                pass

        diag["no_improve"] = int(self._no_improve)

        return loss, diag

    def _sample_points(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` random points uniformly from the bounded search space.

        Returns an array of shape (n, d). If dimensions are not available,
        returns an empty array.
        """
        if not self.dimensions:
            return np.empty((0, 0))
        dims = list(self.dimensions)
        d = len(dims)
        x_mat = np.empty((n, d), dtype=float)
        for j, (low, high) in enumerate(dims):
            x_mat[:, j] = rng.uniform(low, high, size=n)
        return x_mat

    def _compute_gp_max_uncertainty(
            self, n_samples: int = 1024, seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """Estimate the GP's maximum predictive uncertainty over the space.

        Uses random sampling over the bounded space to approximate the maximum
        posterior standard deviation and variance of the current surrogate
        model. Returns (max_std, max_var). If no model is available, returns
        (nan, nan).
        """
        try:
            from typing import cast

            models_obj = getattr(self.optimizer, "models", None)
            if not models_obj:
                return float("nan"), float("nan")
            models_list = cast(list, models_obj)
            model = models_list[-1]
        except Exception:
            return float("nan"), float("nan")

        # If there is no bounded space, cannot sample meaningfully
        if not self.dimensions:
            return float("nan"), float("nan")

        rng = np.random.default_rng(seed)
        x_samples = self._sample_points(int(n_samples), rng)
        if x_samples.size == 0:
            return float("nan"), float("nan")

        # GaussianProcessRegressor supports return_std=True
        try:
            _, std = model.predict(x_samples, return_std=True)
        except Exception:
            return float("nan"), float("nan")
        max_std = float(np.max(std)) if std.size else float("nan")
        return max_std, float(max_std ** 2)

    def run(
            self,
            n_calls: int,
            x0: Optional[Sequence[float]] = None,
            callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Run ``n_calls`` evaluations. If ``x0`` is provided, evaluate it first.

        Returns a dict with keys: best_x, best_loss, history (list of steps).
        Each history item contains keys: x, loss, diag.
        When ``callback`` is provided it is invoked after each observation with
        a payload describing the step. Returning ``False`` from the callback
        stops the run early while keeping accumulated history.
        """

        class _StopRun(Exception):
            """Internal control-flow exception to abort the optimisation loop."""

        history: List[Dict[str, Any]] = []
        X_hist: List[List[float]] = []
        y_hist: List[float] = []

        def record(x: np.ndarray, loss: float, diag: Dict[str, Any]) -> np.ndarray:
            arr = np.asarray(x, dtype=float)
            history.append({"x": arr, "loss": float(loss), "diag": diag})
            X_hist.append(list(map(float, arr)))
            y_hist.append(float(loss))
            return arr

        # ---- Initialise adaptive controller state ----
        param_name = self._choose_param_name()  # 'xi' or 'kappa'
        acq_kwargs_now = self._get_acq_kwargs()
        param_value = float(acq_kwargs_now.get(param_name, self._default_param_value(param_name)))
        no_improve = 0
        best_loss = float("inf")
        best_x_arr: Optional[np.ndarray] = None

        def xi_base(t: int) -> float:
            return max(self._XI_MIN, self._C_INIT / (t ** 0.5))

        def emit_callback(is_initial: bool, no_improve_val: int) -> None:
            if callback is None:
                return
            step_info = history[-1]
            payload = {
                "iteration": len(history),
                "is_initial": bool(is_initial),
                "x": step_info["x"],
                "loss": step_info["loss"],
                "diag": step_info["diag"],
                "best_loss": best_loss if np.isfinite(best_loss) else float("inf"),
                "best_x": None if best_x_arr is None else np.asarray(best_x_arr, dtype=float),
                "no_improve": int(no_improve_val),
            }
            if callback(payload) is False:
                raise _StopRun()

        # ---- Optional x0 seeding ----
        t_seen = 0  # number of *observed* points (for schedule)
        if x0 is not None:
            x0 = np.asarray(x0, dtype=float)
            loss0, diag0 = self.hardware_objective(x0)
            self.observe(x0, loss0)
            t_seen += 1
            if float(loss0) < best_loss:
                best_loss = float(loss0)
                best_x_arr = np.asarray(x0, dtype=float)
            max_std0, max_var0 = self._compute_gp_max_uncertainty(n_samples=1024, seed=12345)
            diag0["gp_max_std"] = max_std0
            diag0["gp_max_var"] = max_var0
            if param_name == "xi":
                diag0["xi"] = float(param_value)
            elif param_name == "kappa":
                diag0["kappa"] = float(param_value)
            diag0["no_improve"] = int(no_improve)
            try:
                logger.info(
                    "Init | loss=%.6g | %s=%.4g | GP max std=%.6g (var=%.6g)",
                    float(loss0), param_name, param_value, max_std0, max_var0,
                )
            except Exception:
                print(
                    f"Init | loss={float(loss0):.6g} | {param_name}={param_value:.4g} "
                    f"| GP max std={max_std0:.6g} (var={max_var0:.6g})"
                )
            record(x0, loss0, diag0)
            try:
                emit_callback(is_initial=True, no_improve_val=no_improve)
            except _StopRun:
                self._param_value_cache = float(param_value)
                self._t_seen = int(t_seen)
                self._best_loss_seen = float(best_loss)
                self._no_improve = int(no_improve)
                return {
                    "best_x": history[-1]["x"],
                    "best_loss": history[-1]["loss"],
                    "history": history,
                }

        # ---- Main optimisation loop ----
        try:
            for it in range(1, int(n_calls) + 1):
                x = self.suggest()
                loss, diag = self.hardware_objective(x)
                self.observe(x, loss)
                t_seen += 1

                max_std, max_var = self._compute_gp_max_uncertainty(
                    n_samples=1024, seed=12345 + it
                )
                diag["gp_max_std"] = max_std
                diag["gp_max_var"] = max_var
                if param_name == "xi":
                    diag["xi"] = float(param_value)
                elif param_name == "kappa":
                    diag["kappa"] = float(param_value)

                improved = (float(best_loss) - float(loss)) > max(0.01 * abs(float(best_loss)), 1e-12)
                if improved:
                    best_loss = float(loss)
                    best_x_arr = np.asarray(x, dtype=float)
                    no_improve = 0
                    new_val = max(self._XI_MIN, param_value * self._COOL_GAMMA)
                else:
                    no_improve += 1
                    base = xi_base(t_seen)
                    if no_improve >= self._STALL_K:
                        new_val = min(self._BOOST_BETA * base, self._XI_MAX)
                    else:
                        new_val = base

                if abs(new_val - param_value) > self._DELTA_TOL:
                    try:
                        ak = self._get_acq_kwargs()
                        ak[param_name] = float(new_val)
                        if self._acq_func == "gp_hedge":
                            ak.setdefault("xi", float(new_val))
                            ak.setdefault("kappa", 1.96)
                        new_opt = self._build_optimizer_with(ak)
                        if X_hist:
                            new_opt.tell(X_hist, y_hist)
                        self._opt = new_opt
                        old_val = float(param_value)
                        param_value = float(new_val)
                        try:
                            logger.info(
                                "Rebuild | iter=%d | %s: %.4g → %.4g | best=%.6g | no_improve=%d",
                                it, param_name, old_val, param_value, best_loss, no_improve,
                            )
                        except Exception:
                            print(
                                f"Rebuild | iter={it} | {param_name}: {old_val:.4g} → {param_value:.4g} "
                                f"| best={best_loss:.6g} | no_improve={no_improve}"
                            )
                    except Exception as e:
                        logger.warning("Adaptive %s rebuild skipped due to error: %r", param_name, e)

                try:
                    logger.info(
                        "Iter %d | loss=%.6g | %s=%.4g | GP max std=%.6g (var=%.6g)",
                        it, float(loss), param_name, param_value, max_std, max_var,
                    )
                except Exception:
                    print(
                        f"Iter {it} | loss={float(loss):.6g} | {param_name}={param_value:.4g} "
                        f"| GP max std={max_std:.6g} (var={max_var:.6g})"
                    )
                diag["no_improve"] = int(no_improve)
                record(x, loss, diag)
                emit_callback(is_initial=False, no_improve_val=no_improve)
        except _StopRun:
            pass

        if history and best_x_arr is None:
            # If no improvement logic triggered (e.g. single point), fall back to argmin history
            best_idx_local = int(np.argmin([h["loss"] for h in history]))
            best_loss = float(history[best_idx_local]["loss"])
            best_x_arr = np.asarray(history[best_idx_local]["x"], dtype=float)

        # Keep step-oriented state in sync for subsequent incremental usage
        self._param_value_cache = float(param_value)
        self._t_seen = int(t_seen)
        self._best_loss_seen = float(best_loss)
        self._no_improve = int(no_improve)

        best_idx = int(np.argmin([h["loss"] for h in history])) if history else -1
        best = history[best_idx] if history else {"x": None, "loss": np.inf}
        return {
            "best_x": best["x"],
            "best_loss": best["loss"],
            "history": history,
        }
