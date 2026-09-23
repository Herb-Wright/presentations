#!/usr/bin/env python3
"""Visualize a stable stochastic linear time-invariant system.

Run with the project virtual environment:
    uv run --python .venv/bin/python slti.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# The user-specified continuous-time-style matrix and its discrete update.
A = np.array([[-1.0, 0.0], [-1.0, -0.5]])
A_DISCRETE = np.eye(2) + 0.1 * A
N_STEPS = 500
N_SAMPLES = 500
NOISE_SCALE = 0.2
SEED = 5170


def step_field(ax: plt.Axes, x_limit: float, y_limit: float) -> None:
    """Draw the deterministic discrete-time increment A' x - x."""
    x_grid = np.linspace(-x_limit, x_limit, 17)
    y_grid = np.linspace(-y_limit, y_limit, 17)
    x1, x2 = np.meshgrid(x_grid, y_grid)
    states = np.stack((x1, x2), axis=-1)
    increments = states @ A_DISCRETE.T - states

    ax.quiver(
        x1,
        x2,
        increments[..., 0],
        increments[..., 1],
        color=sns.color_palette("deep")[0],
        alpha=0.72,
        angles="xy",
        scale_units="xy",
        scale=1.2,
        width=0.004,
    )
    ax.axhline(0, color="0.35", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.35", linewidth=0.8, zorder=0)
    ax.set(xlim=(-x_limit, x_limit), ylim=(-y_limit, y_limit))
    ax.set_aspect("equal", adjustable="box")


def stochastic_rollouts(samples: int, steps: int, rng: np.random.Generator) -> np.ndarray:
    """Return independent final states after the requested number of noisy steps."""
    states = np.zeros((samples, 2))
    for _ in range(steps):
        states = states @ A_DISCRETE.T + NOISE_SCALE * rng.standard_normal(states.shape)
    return states


def make_figure(x0: np.ndarray) -> plt.Figure:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="deep",
        rc={"font.family": "Roboto"},
    )
    final_states = stochastic_rollouts(N_SAMPLES, N_STEPS, np.random.default_rng(SEED))

    # Both panels deliberately use the same fixed axis ranges.
    x_limit, y_limit = 5.0, 4.0
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.8),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.14},
        sharex=True,
        sharey=True,
    )

    step_field(axes[0], x_limit, y_limit)
    axes[0].scatter(
        *x0,
        s=150,
        color="#26734d",
        edgecolor="white",
        linewidth=1.4,
        zorder=3,
    )
    axes[0].set_title("Initialization")

    step_field(axes[1], x_limit, y_limit)
    axes[1].scatter(
        final_states[:, 0],
        final_states[:, 1],
        s=36,
        color="#5b3f8c",
        alpha=0.92,
        edgecolor="white",
        linewidth=0.2,
    )
    axes[1].set_title(f"After t={N_STEPS} iterations")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--x0",
        type=float,
        nargs=2,
        metavar=("X1", "X2"),
        default=(-4.0, 3.0),
        help="initial particle coordinates (default: -4 3)",
    )
    parser.add_argument("--output", type=Path, default=Path("slti.png"), help="output image path")
    parser.add_argument("--show", action="store_true", help="also open the figure interactively")
    args = parser.parse_args()

    print(f"A' = I + 0.1 A =\n{A_DISCRETE}")
    print(f"Eigenvalues of A': {np.linalg.eigvals(A_DISCRETE)}")
    figure = make_figure(np.asarray(args.x0))
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved visualization to {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
