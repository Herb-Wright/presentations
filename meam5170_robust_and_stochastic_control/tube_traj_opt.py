#!/usr/bin/env python3
"""Solve and visualize a small tube-trajectory optimization problem.

The plant is a planar, discrete-time double integrator.  A nominal trajectory
is optimized while a fixed PD error-feedback controller bounds the effect of a
small additive disturbance.  The resulting position projection of the tube is
drawn as a disk at every sample.

Run from this project directory with::

    uv run --python .venv/bin/python tube_traj_opt.py

The default output is written next to this script as ``tube_traj_opt.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.optimize import minimize


# Reproducible problem data -------------------------------------------------
DT = 0.10
HORIZON = 40
TOTAL_TIME = DT * HORIZON
Q_TARGET = (1.0 / 3.0) ** (1.0 / 20.0)
RW_PHYSICAL = 0.045846943848  # raw-state disturbance-ball radius
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
    s = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
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
    kp = 4.0
    kd = (1.0 + 0.5 * DT**2 * kp - Q_TARGET**2) / DT
    return np.array([[-kp, 0.0, -kd, 0.0], [0.0, -kp, 0.0, -kd]])


def make_tube_metric(a: np.ndarray, b: np.ndarray, k: np.ndarray) -> tuple[
    np.ndarray, float, float, float
]:
    """Return T, q, transformed noise radius, and position projection scale.

    The unit ball is defined in transformed error coordinates z = T e.  For
    the identical x/y axes, T is the same 2-by-2 position/velocity metric on
    each axis.  Its first two rows' inverse determine the isotropic physical
    position radius of alpha * E.
    """
    f = a + b @ k
    f_axis = f[np.ix_([0, 2], [0, 2])]
    eigvals, eigvecs = np.linalg.eig(f_axis)
    complex_index = int(np.argmax(eigvals.imag))
    v = eigvecs[:, complex_index]
    basis = np.column_stack((v.real, -v.imag))
    t_axis = np.linalg.inv(basis)
    t = np.kron(t_axis, np.eye(2))
    t_inv = np.linalg.inv(t)
    q = float(np.linalg.norm(t @ f @ t_inv, 2))
    transformed_noise_radius = float(np.linalg.norm(t, 2) * RW_PHYSICAL)
    position_projection_scale = float(np.linalg.norm(t_inv[:2, :], 2))
    return t, q, transformed_noise_radius, position_projection_scale


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
    """Set up and solve the nonlinear program from the tube-TO slide.

    The additive disturbance is a raw-state Euclidean ball of radius
    ``RW_PHYSICAL``.  In z = T e coordinates it is conservatively bounded by
    ``RW_Z = ||T||_2 RW_PHYSICAL`` and the tube recursion uses that radius.
    """
    a, b = double_integrator_matrices()
    k = make_pd_gain()
    closed_loop = a + b @ k
    metric, contraction, rw_z, position_scale = make_tube_metric(a, b, k)
    time, x_ref, u_ref = minimum_jerk_reference()
    x0 = np.zeros(4)
    target = np.array([10.0, 0.0, 0.0, 0.0])
    x_guess, u_guess, alpha_guess = feasible_initial_guess(
        x_ref, u_ref, a, b, target, rw_z, contraction
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
        return alpha[1:] - contraction * alpha[:-1] - rw_z

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
    midpoint_ratio = alpha_opt[HORIZON // 2] / alpha_opt[-1]
    print(f"Metric q = ||T(A+BK)T^-1||_2: {contraction:.7f}")
    print(f"Noise radius: raw={RW_PHYSICAL:.6f}, transformed={rw_z:.6f}")
    print(f"Position projection scale: {position_scale:.6f}")
    print(f"Physical tube radii: alpha_20={alpha_opt[HORIZON // 2] * position_scale:.4f}, "
          f"alpha_T={alpha_opt[-1] * position_scale:.4f}, midpoint/end={midpoint_ratio:.4f}")
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
        "T": metric,
        "contraction": contraction,
        "rw_z": rw_z,
        "position_scale": position_scale,
        "rw_physical": RW_PHYSICAL,
        "result": result,
    }


def make_figure(data: dict[str, np.ndarray | float | object]) -> plt.Figure:
    """Draw the planar reference, nominal, and physical position tube disks."""
    plt.rcParams.update({"font.family": "Roboto", "font.size": 10})
    x_ref = data["x_ref"]
    x_opt = data["x"]
    alpha = data["alpha"] * data["position_scale"]
    fig, ax = plt.subplots(figsize=(12.0, 6.0), facecolor="white")
    blue, orange, teal = "#2468a2", "#d8873d", "#2a9d8f"

    # Each patch is the exact position projection of alpha_t E. Keeping the
    # disks independent avoids interpolating across the sampled tube.
    for (center_x, center_y), radius in zip(x_opt[:, :2], alpha):
        ax.add_patch(Circle((center_x, center_y), float(radius),
                            facecolor=teal, edgecolor=teal, linewidth=0.75,
                            alpha=0.12, zorder=1))
    ax.plot(x_ref[:, 0], x_ref[:, 1], "--", color=orange, linewidth=1.8,
            alpha=0.80, label="reference", zorder=4)
    ax.plot(x_opt[:, 0], x_opt[:, 1], color=blue, linewidth=2.5,
            label="optimized nominal", zorder=5)
    ax.scatter([0.0, 10.0], [0.0, 0.0], s=[30, 36], color="#202b35", zorder=6)
    ax.text(-0.60, -0.95, "start", color="#202b35", fontsize=11)
    ax.text(9.45, -0.95, "goal", color="#202b35", fontsize=11)
    ax.set_xlim(-0.9, 10.9)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.05, top=0.93)
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
