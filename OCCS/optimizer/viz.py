"""Visualization helpers for Bayesian optimisation runs.

This module provides lightweight plotting utilities that work with the
result dict returned by :class:`OCCS.optimizer.optimizer.BayesianOptimizer.run`.

Matplotlib is imported lazily to avoid an import-time dependency when plots
are not used (e.g. during headless or minimal test runs).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import csv
import numpy as np


def _require_matplotlib():
    """Import matplotlib lazily and return pyplot and matplotlib modules.

    Raises ImportError if matplotlib is not installed.
    """
    import matplotlib  # type: ignore
    # Ensure a non-interactive backend for headless environments
    try:
        backend = matplotlib.get_backend().lower()
    except Exception:
        backend = ""
    if "agg" not in backend:
        try:
            matplotlib.use("Agg", force=False)
        except Exception:
            # Fallback silently; pyplot import may still select a usable backend
            pass
    import matplotlib.pyplot as plt  # type: ignore
    return matplotlib, plt


def _extract_losses(history_or_result: Dict[str, Any] | Iterable[Dict[str, Any]]) -> List[float]:
    if isinstance(history_or_result, dict) and "history" in history_or_result:
        history = history_or_result.get("history", [])
    else:
        history = history_or_result  # type: ignore[assignment]
    losses: List[float] = [float(h["loss"]) for h in history]
    return losses


def _extract_diag_series(
    history_or_result: Dict[str, Any] | Iterable[Dict[str, Any]], key: str
) -> List[float]:
    if isinstance(history_or_result, dict) and "history" in history_or_result:
        history = history_or_result.get("history", [])
    else:
        history = history_or_result  # type: ignore[assignment]
    vals: List[float] = []
    for h in history:
        d = h.get("diag", {}) or {}
        v = d.get(key)
        try:
            vals.append(float(v))
        except Exception:
            vals.append(float("nan"))
    return vals


def _auto_dpi(n_points: int, *, base: int = 160, per_point: float = 2.0, max_dpi: int = 360) -> int:
    """Choose a higher DPI for long histories to keep PNG crisp.

    Simple linear rule with clamping: dpi = base + per_point * n_points.
    """
    try:
        n = int(n_points)
    except Exception:
        n = 0
    dpi = int(base + per_point * max(0, n))
    return max(base, min(max_dpi, dpi))


def plot_loss_history(
    history_or_result: Dict[str, Any] | Iterable[Dict[str, Any]],
    *,
    ax=None,
    show_running_min: bool = True,
    title: Optional[str] = None,
    xlabel: str = "Iteration",
    ylabel: str = "Objective loss",
):
    """Plot loss vs. iteration from a BO run.

    Parameters
    ----------
    history_or_result:
        Either the ``result`` dict returned by ``BayesianOptimizer.run`` or a
        list-like of history items (each with a ``"loss"`` key).
    ax:
        Optional matplotlib Axes to draw on. If None, creates a new figure.
    show_running_min:
        Also show the running minimum of the loss for visualizing convergence.
    title/xlabel/ylabel:
        Plot labels.

    Returns
    -------
    (fig, ax):
        The matplotlib Figure and Axes objects used for the plot.
    """
    _, plt = _require_matplotlib()

    losses = _extract_losses(history_or_result)
    iters = list(range(1, len(losses) + 1))

    if ax is None:
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure

    ax.plot(iters, losses, marker="o", ms=3.0, lw=1.2, label="loss")
    if show_running_min and losses:
        running = []
        cur = float("inf")
        for v in losses:
            cur = v if v < cur else cur
            running.append(cur)
        ax.plot(iters, running, ls="--", lw=1.2, color="#d55e00", label="running min")

    # Overlay only xi on the right: normalize by its min/max across the run and show raw xi ticks
    xi_series = _extract_diag_series(history_or_result, "xi")
    has_xi = any(np.isfinite(v) for v in xi_series if v is not None)
    ax2 = None
    if has_xi and xi_series:
        # Compute finite min/max for normalization
        finite_xi = np.asarray([v for v in xi_series if np.isfinite(v)], dtype=float)
        xi_min = float(np.min(finite_xi)) if finite_xi.size else 0.0
        xi_max = float(np.max(finite_xi)) if finite_xi.size else 1.0
        if not np.isfinite(xi_min) or not np.isfinite(xi_max) or xi_max <= xi_min:
            xi_min, xi_max = 0.0, 1.0

        # Normalize series into [0,1]
        denom = max(1e-12, (xi_max - xi_min))
        xi_norm = []
        for v in xi_series:
            try:
                xi_norm.append((float(v) - xi_min) / denom)
            except Exception:
                xi_norm.append(np.nan)

        # Right axis hosts normalized data but hides its own ticks
        ax2 = ax.twinx()
        ax2.plot(iters, xi_norm, color="#009e73", alpha=0.9, lw=1.2, label="xi (norm)")
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(False)
        ax2.set_yticks([])
        ax2.set_ylabel("")

        # Add a secondary right axis that shows raw xi values corresponding to normalized [0,1]
        try:
            def norm_to_xi(y: float) -> float:
                return xi_min + y * (xi_max - xi_min)

            def xi_to_norm(x: float) -> float:
                return (x - xi_min) / max(1e-12, (xi_max - xi_min))

            secax = ax2.secondary_yaxis("right", functions=(norm_to_xi, xi_to_norm))
            secax.set_ylabel("xi")
        except Exception:
            pass

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ax2 is not None:
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", framealpha=0.8)
    else:
        ax.legend(loc="best", framealpha=0.8)
    fig.tight_layout()
    return fig, ax


def plot_uncertainty_history(
    history_or_result: Dict[str, Any] | Iterable[Dict[str, Any]],
    *,
    metric: str = "std",  # "std" or "var"
    ax=None,
    title: Optional[str] = None,
    xlabel: str = "Iteration",
    ylabel_std: str = "GP max std",
    ylabel_var: str = "GP max var",
):
    """Plot GP max uncertainty vs. iteration from a BO run.

    Parameters
    ----------
    metric:
        Which metric to plot: "std" (posterior standard deviation) or
        "var" (posterior variance).
    """
    _, plt = _require_matplotlib()
    key = "gp_max_std" if metric.lower() == "std" else "gp_max_var"
    series = _extract_diag_series(history_or_result, key)
    iters = list(range(1, len(series) + 1))

    if ax is None:
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure

    ax.plot(iters, series, marker="o", ms=3.0, lw=1.2, label=key)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_std if key == "gp_max_std" else ylabel_var)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.8)
    fig.tight_layout()
    return fig, ax


def save_loss_history_plot(
    history_or_result: Dict[str, Any] | Iterable[Dict[str, Any]],
    path: str,
    dpi: Optional[int] = None,
    **plot_kwargs: Any,
) -> str:
    """Convenience function: plot loss history and save to file.

    Returns the path for convenience.
    """
    _, plt = _require_matplotlib()
    fig, _ = plot_loss_history(history_or_result, **plot_kwargs)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Choose a higher DPI automatically when many points are present
    if dpi is None:
        # Use number of loss points as heuristic
        n = len(_extract_losses(history_or_result))
        dpi = _auto_dpi(n)
    fig.savefig(p.as_posix(), dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return p.as_posix()


def save_uncertainty_history_plot(
    history_or_result: Dict[str, Any] | Iterable[Dict[str, Any]],
    path: str,
    dpi: Optional[int] = None,
    **plot_kwargs: Any,
) -> str:
    """Plot GP uncertainty history and save to file."""
    _, plt = _require_matplotlib()
    fig, _ = plot_uncertainty_history(history_or_result, **plot_kwargs)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if dpi is None:
        series = _extract_diag_series(history_or_result, "gp_max_std")
        dpi = _auto_dpi(len(series))
    fig.savefig(p.as_posix(), dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return p.as_posix()


def plot_exploration_param_history(
    history_or_result: Dict[str, Any] | Iterable[Dict[str, Any]],
    *,
    ax=None,
    title: Optional[str] = None,
    xlabel: str = "Iteration",
):
    """Plot exploration parameter history (xi and/or kappa).

    Draws whichever series are present in diag: "xi" (EI/PI/gp_hedge) and/or
    "kappa" (LCB/UCB).
    """
    _, plt = _require_matplotlib()

    xi_series = _extract_diag_series(history_or_result, "xi")
    ka_series = _extract_diag_series(history_or_result, "kappa")
    n = max(len(xi_series), len(ka_series))
    iters = list(range(1, n + 1))

    if ax is None:
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure

    has_xi = any(np.isfinite(v) for v in xi_series if v is not None)
    has_k = any(np.isfinite(v) for v in ka_series if v is not None)
    ax.set_xlabel(xlabel)

    if has_xi and has_k:
        # Two axes: left for xi with known bounds, right for kappa autoscaled
        XI_MIN_DEFAULT, XI_MAX_DEFAULT = 1e-3, 5e-1
        finite_xi = np.asarray([v for v in xi_series if np.isfinite(v)], dtype=float)
        xi_lo = float(np.min(finite_xi)) if finite_xi.size else XI_MIN_DEFAULT
        xi_hi = float(np.max(finite_xi)) if finite_xi.size else XI_MAX_DEFAULT
        # Ensure bounds cover defaults
        xi_lo = min(xi_lo, XI_MIN_DEFAULT)
        xi_hi = max(xi_hi, XI_MAX_DEFAULT)
        ax.plot(iters, xi_series, color="#009e73", lw=1.4, label="xi")
        ax.set_ylabel("xi", color="#009e73")
        ax.tick_params(axis='y', labelcolor="#009e73")
        ax.set_ylim(xi_lo, xi_hi)

        ax2 = ax.twinx()
        ax2.plot(iters, ka_series, color="#cc79a7", lw=1.2, label="kappa")
        ax2.set_ylabel("kappa", color="#cc79a7")
        ax2.tick_params(axis='y', labelcolor="#cc79a7")
        # Legends combined
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", framealpha=0.8)
    else:
        # Single axis case
        if has_xi:
            ax.plot(iters, xi_series, color="#009e73", lw=1.4, label="xi")
            ax.set_ylabel("xi")
        if has_k:
            ax.plot(iters, ka_series, color="#cc79a7", lw=1.2, label="kappa")
            ax.set_ylabel("kappa")
        ax.legend(loc="best", framealpha=0.8)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def save_exploration_param_history_plot(
    history_or_result: Dict[str, Any] | Iterable[Dict[str, Any]],
    path: str,
    dpi: Optional[int] = None,
    **plot_kwargs: Any,
) -> str:
    """Plot exploration parameter (xi/kappa) history and save to file."""
    _, plt = _require_matplotlib()
    fig, _ = plot_exploration_param_history(history_or_result, **plot_kwargs)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if dpi is None:
        n = len(_extract_losses(history_or_result))
        dpi = _auto_dpi(n)
    fig.savefig(p.as_posix(), dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return p.as_posix()


def save_history_csv(
    history_or_result: Dict[str, Any] | Iterable[Dict[str, Any]],
    path: str,
    include_running_min: bool = True,
    diag_keys: Tuple[str, ...] = ("delta_nm", "gp_max_std", "gp_max_var", "xi", "kappa"),
) -> str:
    """Save optimisation history to CSV.

    Columns: iteration, loss, [running_min], v0..v{n-1}, and any requested
    diag_keys that are present and scalar.
    """
    # Extract data
    if isinstance(history_or_result, dict) and "history" in history_or_result:
        history = history_or_result.get("history", [])
    else:
        history = list(history_or_result)  # type: ignore[arg-type]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        # Create an empty file with only header
        with p.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "loss"])  # minimal header
        return p.as_posix()

    # Determine vector length from first x
    first_x = history[0].get("x")
    try:
        n = len(first_x)  # type: ignore[arg-type]
    except Exception:
        n = 0

    # Build header
    header = ["iteration", "loss"]
    if include_running_min:
        header.append("running_min")
    header += [f"v{i}" for i in range(n)]
    # Only include diag columns that exist and are scalar in at least one item
    present_diag_keys: List[str] = []
    for k in diag_keys:
        for h in history:
            d = h.get("diag", {}) or {}
            v = d.get(k)
            if v is not None and not hasattr(v, "__len__"):
                present_diag_keys.append(k)
                break
    header += present_diag_keys

    # Write rows
    running = float("inf")
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, h in enumerate(history, start=1):
            loss = float(h.get("loss", float("nan")))
            running = min(running, loss)
            x = h.get("x")
            row: List[Any] = [i, loss]
            if include_running_min:
                row.append(running)
            # Voltages
            if x is not None:
                try:
                    row += [float(val) for val in x]
                except Exception:
                    row += [""] * n
            else:
                row += [""] * n
            # Diag scalar fields
            d = h.get("diag", {}) or {}
            for k in present_diag_keys:
                v = d.get(k)
                row.append(v if (v is None or not hasattr(v, "__len__")) else "")
            writer.writerow(row)

    return p.as_posix()


__all__ = [
    "plot_loss_history",
    "save_loss_history_plot",
    "plot_uncertainty_history",
    "save_uncertainty_history_plot",
    "plot_exploration_param_history",
    "save_exploration_param_history_plot",
    "save_history_csv",
]
