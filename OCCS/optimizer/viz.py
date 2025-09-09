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

    # If uncertainty info present, overlay on a twin axis for visibility
    std_series = _extract_diag_series(history_or_result, "gp_max_std")
    has_std = any(np.isfinite(v) for v in std_series if v is not None)
    ax2 = None
    if has_std:
        if std_series:
            ax2 = ax.twinx()
            ax2.plot(iters, std_series, color="#0072b2", alpha=0.6, lw=1.2, label="GP max std")
            ax2.set_ylabel("GP max std")

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
    **plot_kwargs: Any,
) -> str:
    """Convenience function: plot loss history and save to file.

    Returns the path for convenience.
    """
    _, plt = _require_matplotlib()
    fig, _ = plot_loss_history(history_or_result, **plot_kwargs)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.as_posix(), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p.as_posix()


def save_uncertainty_history_plot(
    history_or_result: Dict[str, Any] | Iterable[Dict[str, Any]],
    path: str,
    **plot_kwargs: Any,
) -> str:
    """Plot GP uncertainty history and save to file."""
    _, plt = _require_matplotlib()
    fig, _ = plot_uncertainty_history(history_or_result, **plot_kwargs)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.as_posix(), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p.as_posix()


def save_history_csv(
    history_or_result: Dict[str, Any] | Iterable[Dict[str, Any]],
    path: str,
    include_running_min: bool = True,
    diag_keys: Tuple[str, ...] = ("delta_nm", "gp_max_std", "gp_max_var"),
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
    "save_history_csv",
]
