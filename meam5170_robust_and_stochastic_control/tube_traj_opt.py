#!/usr/bin/env python3
"""Solve and visualize a small tube-trajectory optimization problem.

The plant is a planar, discrete-time double integrator.  A nominal trajectory
is optimized while a fixed PD error-feedback controller bounds the effect of a
small additive disturbance.  The resulting position projection of the tube is
drawn as a disk at every sample.

Run from any directory with::

    python tube_traj_opt.py

The default output is written next to this script as ``tube_traj_opt.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import minimize


# Reproducible problem data -------------------------------------------------
DT = 0.10
HORIZON = 40
TOTAL_TIME = DT * HORIZON
RW = 0.012  # radius of the additive disturbance ball in R^4
GAMMA = 0.70
SEED = 5170
Y_AMPLITUDE = 2.0
Y_SHAPE_PEAK = 0.0037190381436


def double_integrator_matrices(dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
    """Return exact-ZOH matrices for the planar double integrator.

    The model is linear, so linearizing about every point of the polynomial
    reference gives these same constant A and B matrices: they are the exact
    trajectory linearization used by the tube problem.
    """
    eye = np.eye(2)
    a = np.block([[eye, dt * eye], [np.zeros((2, 2)), eye]])
    b = np.vstack((0.5 * dt**2 * eye, dt * eye))
    return a, b


def minimum_jerk_reference(
    horizon: int = HORIZON, dt: float = DT, final_height: float = 10.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a rest-to-rest polynomial path and its acceleration.

    The x coordinate uses the usual quintic minimum-jerk profile. The y
    coordinate is a normalized degree-seven S-shape with zero position,
    velocity, and acceleration at both endpoints.
    """
    time = np.arange(horizon + 1, dtype=float) * dt
    tau = np.clip(time / (horizon * dt), 0.0, 1.0)
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    ds = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / (horizon * dt)
    d2s = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / (horizon * dt) ** 2
    # g(tau)=tau^3(1-tau)^3(2*tau-1), normalized to a +/-2 excursion.
    lateral = Y_AMPLITUDE * (
        -tau**3 + 5.0 * tau**4 - 9.0 * tau**5 + 7.0 * tau**6 - 2.0 * tau**7
    ) / Y_SHAPE_PEAK
    lateral_d1 = Y_AMPLITUDE * (
        -3.0 * tau**2 + 20.0 * tau**3 - 45.0 * tau**4
        + 42.0 * tau**5 - 14.0 * tau**6
    ) / (horizon * dt * Y_SHAPE_PEAK)
    lateral_d2 = Y_AMPLITUDE * (
        -6.0 * tau + 60.0 * tau**2 - 180.0 * tau**3
        + 210.0 * tau**4 - 84.0 * tau**5
    ) / ((horizon * dt) ** 2 * Y_SHAPE_PEAK)
    x_ref = np.column_stack(
        (final_height * s, lateral, final_height * ds, lateral_d1)
    )
    u_ref = np.column_stack((final_height * d2s[:-1], lateral_d2[:-1]))
    return time, x_ref, u_ref


def make_pd_gain() -> np.ndarray:
    """Fixed negative-feedback PD gain, duplicated for x and y axes."""
    kp, kd = 0.8597, 2.8055
    return np.array([[-kp, 0.0, -kd, 0.0], [0.0, -kp, 0.0, -kd]])


def rollout(x0: np.ndarray, controls: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Roll out the exact discrete dynamics for a control sequence."""
    states = np.empty((controls.shape[0] + 1, 4), dtype=float)
    states[0] = x0
    for t, control in enumerate(controls):
        states[t + 1] = a @ states[t] + b @ control
    return states


def feasible_initial_guess(
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    target: np.ndarray,
    rw: float,
    contraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a dynamics-feasible, terminally-correct starting point.

    Midpoint samples of both reference accelerations give a good warm start. A
    least-squares correction to both input channels then makes the terminal
    position and velocity exactly equal to the target.
    """
    horizon = u_ref.shape[0]
    x0 = np.zeros(4)

    midpoint = (np.arange(horizon, dtype=float) + 0.5) * DT
    tau = midpoint / (horizon * DT)
    x_accel_mid = 10.0 * (
        60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3
    ) / (horizon * DT) ** 2
    y_accel_mid = Y_AMPLITUDE * (
        -6.0 * tau + 60.0 * tau**2 - 180.0 * tau**3
        + 210.0 * tau**4 - 84.0 * tau**5
    ) / ((horizon * DT) ** 2 * Y_SHAPE_PEAK)
    controls = np.column_stack((x_accel_mid, y_accel_mid))

    # Terminal response matrix from each acceleration to [px, py, vx, vy].
    response = np.zeros((4, 2 * horizon))
    for j in range(horizon):
        impulse = np.zeros((horizon, 2))
        impulse[j, 0] = 1.0
        terminal = rollout(x0, impulse, a, b)[-1]
        response[:, 2 * j] = terminal[[0, 1, 2, 3]]
        impulse[j, 0] = 0.0
        impulse[j, 1] = 1.0
        terminal = rollout(x0, impulse, a, b)[-1]
        response[:, 2 * j + 1] = terminal[[0, 1, 2, 3]]
    error = target - rollout(x0, controls, a, b)[-1]
    correction = np.linalg.lstsq(response, error, rcond=None)[0]
    controls += correction.reshape(horizon, 2)
    states = rollout(x0, controls, a, b)

    alpha = np.empty(horizon + 1)
    alpha[0] = 0.0
    for t in range(horizon):
        alpha[t + 1] = contraction * alpha[t] + rw
    return states, controls, alpha


def solve_tube_problem() -> dict[str, np.ndarray | float | object]:
    """Set up and solve the nonlinear program from the tube-TO slide."""
    a, b = double_integrator_matrices()
    k = make_pd_gain()
    closed_loop = a + b @ k
    contraction = float(np.linalg.norm(closed_loop, 2))
    time, x_ref, u_ref = minimum_jerk_reference()
    x0 = np.zeros(4)
    target = np.array([10.0, 0.0, 0.0, 0.0])
    x_guess, u_guess, alpha_guess = feasible_initial_guess(
        x_ref, u_ref, a, b, target, RW, contraction
    )

    n_x = (HORIZON + 1) * 4
    n_u = HORIZON * 2

    def unpack(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = z[:n_x].reshape(HORIZON + 1, 4)
        u = z[n_x : n_x + n_u].reshape(HORIZON, 2)
        alpha = z[n_x + n_u :]
        return x, u, alpha

    q = np.diag([4.0, 12.0, 0.25, 0.25])
    r = np.diag([0.025, 0.025])

    def objective(z: np.ndarray) -> float:
        x, u, alpha = unpack(z)
        tracking = x[:-1] - x_ref[:-1]
        return float(
            np.einsum("ti,ij,tj->", tracking, q, tracking)
            + np.einsum("ti,ij,tj->", u, r, u)
            + GAMMA * np.sum(alpha[1:])
        )

    def equalities(z: np.ndarray) -> np.ndarray:
        x, u, alpha = unpack(z)
        dynamics = x[1:] - (x[:-1] @ a.T + u @ b.T)
        return np.concatenate(
            (x[0] - x0, dynamics.ravel(), x[-1] - target, np.array([alpha[0]]))
        )

    def tube_slack(z: np.ndarray) -> np.ndarray:
        _, _, alpha = unpack(z)
        return alpha[1:] - contraction * alpha[:-1] - RW

    z0 = np.concatenate((x_guess.ravel(), u_guess.ravel(), alpha_guess))
    bounds = [(None, None)] * (n_x + n_u) + [(0.0, None)] * (HORIZON + 1)
    result = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints=(
            {"type": "eq", "fun": equalities},
            {"type": "ineq", "fun": tube_slack},
        ),
        options={"ftol": 1e-9, "maxiter": 1200, "disp": False},
    )
    x_opt, u_opt, alpha_opt = unpack(result.x)
    dynamics_error = x_opt[1:] - (x_opt[:-1] @ a.T + u_opt @ b.T)
    print(f"Solver: SLSQP | success={result.success} | status={result.status}")
    print(f"Message: {result.message}")
    print(f"Objective: {result.fun:.6f} | iterations: {result.nit}")
    print(f"Max dynamics residual: {np.max(np.abs(dynamics_error)):.3e}")
    print(f"Minimum tube slack: {np.min(tube_slack(result.x)):.3e}")
    print(f"Final state: {x_opt[-1]} | alpha_T: {alpha_opt[-1]:.4f}")
    if not result.success or np.max(np.abs(equalities(result.x))) > 2e-5:
        raise RuntimeError("Tube trajectory optimization did not satisfy its constraints")
    return {
        "time": time,
        "x_ref": x_ref,
        "u_ref": u_ref,
        "x": x_opt,
        "u": u_opt,
        "alpha": alpha_opt,
        "A": a,
        "B": b,
        "K": k,
        "contraction": contraction,
        "result": result,
    }


def sample_noisy_rollouts(data: dict[str, np.ndarray | float | object], count: int = 4) -> list[np.ndarray]:
    """Generate restrained, deterministic closed-loop disturbances for the plot."""
    a = data["A"]
    b = data["B"]
    k = data["K"]
    x_nom = data["x"]
    u_nom = data["u"]
    rng = np.random.default_rng(SEED)
    trajectories: list[np.ndarray] = []
    for _ in range(count):
        x = np.zeros(4)
        path = np.empty_like(x_nom)
        path[0] = x
        for t in range(HORIZON):
            direction = rng.normal(size=4)
            direction /= np.linalg.norm(direction)
            radius = RW * rng.random() ** 0.25 * 0.78
            disturbance = radius * direction
            control = u_nom[t] + k @ (x - x_nom[t])
            x = a @ x + b @ control + disturbance
            path[t + 1] = x
        trajectories.append(path)
    return trajectories


def make_figure(data: dict[str, np.ndarray | float | object]) -> plt.Figure:
    """Draw a minimal 3-D lifted tube around the planar nominal trajectory."""
    plt.rcParams.update({"font.family": "Roboto", "font.size": 10})
    x_ref = data["x_ref"]
    x_opt = data["x"]
    alpha = data["alpha"]
    trajectories = sample_noisy_rollouts(data, count=3)

    fig = plt.figure(figsize=(12.0, 6.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    blue, orange, teal, red = "#2468a2", "#d8873d", "#2a9d8f", "#c96b6b"

    # Cross-sections use the planar normal and a visual z direction. z is not
    # a dynamical state; it simply exposes the unit-ball tube in the figure.
    # The optimizer stores [p_y, p_x, v_y, v_x], so swap the first two entries
    # here to display physical (p_x, p_y) coordinates.
    centers = np.column_stack((x_opt[:, 1], x_opt[:, 0], np.zeros(len(x_opt))))
    tangent_xy = np.gradient(centers[:, :2], axis=0)
    tangent_xy /= np.linalg.norm(tangent_xy, axis=1, keepdims=True)
    normal = np.column_stack((-tangent_xy[:, 1], tangent_xy[:, 0], np.zeros(len(x_opt))))
    binormal = np.tile(np.array([0.0, 0.0, 1.0]), (len(x_opt), 1))
    theta = np.linspace(0.0, 2.0 * np.pi, 28)
    ring = np.cos(theta)[None, :, None] * normal[:, None, :]
    ring += np.sin(theta)[None, :, None] * binormal[:, None, :]
    surface = centers[:, None, :] + alpha[:, None, None] * ring
    ax.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2],
                    color=teal, alpha=0.38, linewidth=0.0, antialiased=True, shade=True)

    for path in trajectories:
        ax.plot(path[:, 1], path[:, 0], np.zeros(len(path)), color=red,
                linewidth=0.8, alpha=0.34, zorder=2)
    ax.plot(x_ref[:, 1], x_ref[:, 0], np.zeros(len(x_ref)), "--",
            color=orange, linewidth=1.6, alpha=0.78, zorder=4)
    ax.plot(x_opt[:, 1], x_opt[:, 0], np.zeros(len(x_opt)), color=blue,
            linewidth=2.6, zorder=5)
    ax.scatter([x_opt[0, 1]], [x_opt[0, 0]], [0.0], s=28, color="#202b35", zorder=6)
    ax.scatter([x_opt[-1, 1]], [x_opt[-1, 0]], [0.0], s=36, color=orange,
               edgecolor="white", linewidth=0.7, zorder=6)
    ax.text(x_opt[0, 1] - 0.25, x_opt[0, 0] - 0.55, 0.0, "start", color="#202b35",
            fontsize=8, zorder=7)
    ax.text(x_opt[-1, 1] - 0.2, x_opt[-1, 0] - 0.55, 0.0, "goal", color="#202b35",
            fontsize=8, zorder=7)

    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-0.8, 10.8)
    ax.set_zlim(-0.42, 0.42)
    ax.set_box_aspect((11.0, 5.6, 1.8))
    ax.view_init(elev=34.0, azim=-64.0)
    ax.set_axis_off()
    ax.legend(handles=[
        Line2D([], [], color=orange, linestyle="--", linewidth=1.6, label="reference"),
        Line2D([], [], color=blue, linewidth=2.6, label="optimized nominal"),
        Line2D([], [], color=teal, linewidth=7.0, alpha=0.28, label="tube"),
        Line2D([], [], color=red, linewidth=0.8, alpha=0.45, label="noisy rollouts"),
    ], loc="upper left", bbox_to_anchor=(0.025, 0.96), frameon=False, fontsize=9)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(__file__).resolve().with_name("tube_traj_opt.png")
    parser.add_argument("--output", type=Path, default=default_output,
                        help="output image path (relative paths are next to this script)")
    parser.add_argument("--show", action="store_true", help="also open the figure interactively")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else Path(__file__).resolve().parent / args.output
    data = solve_tube_problem()
    figure = make_figure(data)
    figure.savefig(output, dpi=220, facecolor="white")
    print(f"Saved visualization to {output} (2640 x 1320 px, 2:1)")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
