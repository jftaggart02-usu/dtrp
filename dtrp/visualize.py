"""Visualisation utilities for DTRP solutions."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

Circle = tuple[float, float, float]


def plot_solution(
    circles: Sequence[Circle],
    result: dict,
    title: str = "DTRP Solution",
    ax: "matplotlib.axes.Axes | None" = None,
) -> tuple:
    """Plot the DTRP trajectory on top of the visit regions.

    Parameters
    ----------
    circles:
        Original list of ``(cx, cy, r)`` circles.
    result:
        Dictionary returned by :meth:`DTRPSolver.solve`.
    title:
        Figure title.
    ax:
        Existing Axes to draw into.  A new figure is created when ``None``.

    Returns
    -------
    ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    else:
        fig = ax.get_figure()

    cmap = plt.get_cmap("tab10")

    visit_order = result.get("visit_order", list(range(len(circles))))
    visit_set = set(visit_order)

    # Draw circles --------------------------------------------------------
    for idx, (cx, cy, r) in enumerate(circles):
        color = cmap(visit_order.index(idx) % 10) if idx in visit_set else "grey"
        patch = mpatches.Circle(
            (cx, cy), r, color=color, alpha=0.25, linewidth=0
        )
        border = mpatches.Circle(
            (cx, cy), r, fill=False, edgecolor=color, linewidth=2
        )
        ax.add_patch(patch)
        ax.add_patch(border)
        ax.text(
            cx, cy,
            f"{idx}",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=color,
        )

    # Draw path -----------------------------------------------------------
    if result.get("x") is not None and len(result["x"]) > 0:
        x, y, theta = result["x"], result["y"], result["theta"]

        ax.plot(x, y, "k-", linewidth=2, label="Path", zorder=3)
        ax.plot(x[0], y[0], "go", markersize=10, label="Start", zorder=4)
        ax.plot(x[-1], y[-1], "r^", markersize=10, label="End", zorder=4)

        # Heading arrows (every ~5 % of the path)
        step = max(1, len(x) // 20)
        ax.quiver(
            x[::step], y[::step],
            np.cos(theta[::step]), np.sin(theta[::step]),
            scale=20, width=0.003, color="darkred", alpha=0.6, zorder=5,
        )

        total = result.get("total_length", 0.0)
        status = "✓" if result.get("success") else "!"
        ax.set_title(
            f"{title}  [{status}  length = {total:.4g}]", fontsize=12
        )
    else:
        ax.set_title(title, fontsize=12)

    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax
