#!/usr/bin/env python3
"""Visualize set-valued propagation for a linear system with zonotope noise.

The disturbance set is the box W = [-0.1, 0.1]^2 = Z(0, 0.1 I).  Run with:
    uv run --python .venv/bin/python zlti.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Polygon


# Match slti.py: A_d is the discrete update used in the stochastic example.
A = np.array([[-1.0, 0.0], [-1.0, -0.8]])
A_DISCRETE = np.eye(2) + 0.1 * A
N_STEPS = 50
W_GENERATORS = 0.1 * np.eye(2)  # W = Z(0, 0.1 I) = [-0.1, 0.1]^2


def step_field(ax: plt.Axes, x_limit: float, y_limit: float) -> None:
    """Draw the deterministic increment A_d x - x."""
    x_grid = np.linspace(-x_limit, x_limit, 17)
    y_grid = np.linspace(-y_limit, y_limit, 17)
    x1, x2 = np.meshgrid(x_grid, y_grid)
    states = np.stack((x1, x2), axis=-1)
    increments = states @ A_DISCRETE.T - states
    ax.quiver(
        x1, x2, increments[..., 0], increments[..., 1],
        color=sns.color_palette("deep")[0], alpha=0.58, angles="xy",
        scale_units="xy", scale=1.2, width=0.004,
    )
    ax.axhline(0, color="0.35", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.35", linewidth=0.8, zorder=0)
    ax.set(xlim=(-x_limit, x_limit), ylim=(-y_limit, y_limit))
    ax.set_aspect("equal", adjustable="box")


def zonotope_vertices(center: np.ndarray, generators: np.ndarray) -> np.ndarray:
    """Return the boundary vertices of a 2-D zonotope in counter-clockwise order.

    A zonotope is a Minkowski sum of line segments.  Sorting their directions
    yields its boundary directly, so this avoids sampling particles or a grid.
    """
    if generators.size == 0:
        return center[None, :]

    # A segment is unchanged by negating its generator; use angles in [0, pi).
    oriented = generators.copy()
    flip = (oriented[:, 1] < 0) | ((oriented[:, 1] == 0) & (oriented[:, 0] < 0))
    oriented[flip] *= -1
    angles = np.arctan2(oriented[:, 1], oriented[:, 0])
    ordered = oriented[np.argsort(angles)]

    first = center - ordered.sum(axis=0)
    forward = first + np.cumsum(2 * ordered, axis=0)
    backward = forward[-1] - np.cumsum(2 * ordered, axis=0)
    return np.vstack((first, forward, backward[:-1]))


def propagate_zonotopes(x0: np.ndarray, steps: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Propagate X[t+1] = A_d X[t] (+) W, starting at the point x0."""
    zonotopes = [(x0, np.empty((0, 2)))]
    center, generators = x0, np.empty((0, 2))
    for _ in range(steps):
        center = A_DISCRETE @ center
        generators = np.vstack((generators @ A_DISCRETE.T, W_GENERATORS))
        zonotopes.append((center, generators))
    return zonotopes


def draw_zonotope(ax: plt.Axes, center: np.ndarray, generators: np.ndarray,
                  color: tuple[float, float, float], alpha: float) -> None:
    """Add a filled exact zonotope boundary to an axes."""
    vertices = zonotope_vertices(center, generators)
    ax.add_patch(Polygon(vertices, closed=True, facecolor=color, edgecolor=color,
                         alpha=alpha, linewidth=1.4, zorder=2))


def make_figure(x0: np.ndarray, steps: int) -> plt.Figure:
    sns.set_theme(style="whitegrid", context="talk", palette="deep", rc={"font.family": "Roboto"})
    zonotopes = propagate_zonotopes(x0, steps)
    # Match the fixed view and proportions used by slti.py.
    x_limit, y_limit = 5.0, 4.0
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), constrained_layout=True,
                             gridspec_kw={"wspace": 0.14}, sharex=True, sharey=True)

    step_field(axes[0], x_limit, y_limit)
    axes[0].scatter(*x0, s=150, color="#26734d", edgecolor="white", linewidth=1.4, zorder=3)
    axes[0].set_title("Initialization")

    step_field(axes[1], x_limit, y_limit)
    zonotope_color = "#5b3f8c"  # Same purple accent as the rollout panel in slti.py.
    for t, (center, generators) in enumerate(zonotopes[1:], start=1):
        # Earlier reachable sets remain visible while the final set receives emphasis.
        draw_zonotope(axes[1], center, generators, zonotope_color, alpha=0.08 if t < steps else 0.34)
    final_center, final_generators = zonotopes[-1]
    final_vertices = zonotope_vertices(final_center, final_generators)
    axes[1].plot(*np.vstack((final_vertices, final_vertices[0])).T, color=zonotope_color, linewidth=2.4, zorder=3)
    axes[1].scatter(*final_center, s=42, color=zonotope_color, edgecolor="white", linewidth=0.8, zorder=4)
    axes[1].set_title(f"Zonotopes through $t={steps}$")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x0", type=float, nargs=2, metavar=("X1", "X2"), default=(-4.0, 2.0),
                        help="initial state coordinates (default: -4 2)")
    parser.add_argument("--steps", type=int, default=N_STEPS, help="number of set propagations (default: 10)")
    parser.add_argument("--output", type=Path, default=Path("zlti.png"), help="output image path")
    parser.add_argument("--show", action="store_true", help="also open the figure interactively")
    args = parser.parse_args()

    print(f"A_d = I + 0.1 A =\n{A_DISCRETE}")
    print(f"Propagating W = [-0.1, 0.1]^2 for {args.steps} steps")
    figure = make_figure(np.asarray(args.x0), args.steps)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved visualization to {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
