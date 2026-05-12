#!/usr/bin/env python3


from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

import pysindy as ps


def lorenz_rhs(t, x, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Lorenz-63 right-hand side."""
    return np.array(
        [
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ],
        dtype=float,
    )


def generate_lorenz_trajectory(x0, t):
    """Generate one clean Lorenz trajectory and exact x_dot values."""
    sol = solve_ivp(
        lorenz_rhs,
        (float(t[0]), float(t[-1])),
        np.asarray(x0, dtype=float),
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    x = sol.y.T
    x_dot = np.array([lorenz_rhs(ti, xi) for ti, xi in zip(t, x)], dtype=float)
    return x, x_dot



def print_matrix(title, matrix, row_names=None, col_names=None, precision=4):
    """Small dependency-free table printer."""
    print(f"\n{title}")
    arr = np.asarray(matrix)
    if row_names is None:
        row_names = [f"row{i}" for i in range(arr.shape[0])]
    if col_names is None:
        col_names = [f"col{j}" for j in range(arr.shape[1])]

    width = 14
    header = " " * 14 + "".join(f"{name:>{width}}" for name in col_names)
    print(header)
    for name, row in zip(row_names, arr):
        vals = "".join(f"{v:>{width}.{precision}g}" for v in row)
        print(f"{name:<14}{vals}")


def main():
    rng = np.random.default_rng(123)

    # Short time window keeps the demo quick and avoids unnecessary chaos issues.
    t = np.linspace(0.0, 1.0, 201)

    initial_conditions = [
        [-8.0, 8.0, 27.0],
        [-6.0, 7.0, 25.0],
        [-10.0, 9.0, 30.0],
        [-7.5, 6.5, 28.0],
    ]

    noise_levels = np.array([0.0, 1.0e-4, 5.0e-4, 1.0e-3], dtype=float)

    x_list = []
    x_dot_list = []
    clean_x_list = []

    for x0, noise in zip(initial_conditions, noise_levels, strict=True):
        x_clean, x_dot = generate_lorenz_trajectory(x0, t)
        x_noisy = x_clean + noise * rng.standard_normal(size=x_clean.shape)

        clean_x_list.append(x_clean)
        x_list.append(x_noisy)
        x_dot_list.append(x_dot)

    sigma_x_for_model = float(np.max(noise_levels))

    model = ps.BINDy(
        sigma_x=sigma_x_for_model,
        shared_support=True,
    )

    print("\nFitting BINDy shared-support model...")
    print("Actual state-noise levels per trajectory:", noise_levels)
    print("sigma_x passed to BINDy:", sigma_x_for_model)

    model.fit(x_list, t=[t] * len(x_list), x_dot=x_dot_list)

    print("\n=== Shared-support diagnostics ===")
    print("shared_support_:", getattr(model.optimizer, "shared_support_", None))
    print("multi_trajectory_mode_:", getattr(model.optimizer, "multi_trajectory_mode_", None))
    print("Public pooled coef shape:", model.optimizer.coef_.shape)
    print("Shared support shape:", model.optimizer.ind_.shape)
    print("Trajectory coef shape:", model.optimizer.coef_trajectories_.shape)
    print("Trajectory lengths:", model.optimizer.trajectory_lengths_)
    print("Trajectory sigma2s:", getattr(model.optimizer, "trajectory_sigma2s_", None))

    feature_names = model.feature_library.get_feature_names(
        input_features=["x", "y", "z"]
    )
    target_names = ["x_dot", "y_dot", "z_dot"]

    print("\nFeature names:")
    for i, name in enumerate(feature_names):
        print(f"{i:2d}: {name}")

    print("\nRecovered public pooled model:")
    model.print()

    print_matrix(
        "Public pooled coefficients, shape (n_targets, n_features)",
        model.optimizer.coef_,
        row_names=target_names,
        col_names=feature_names,
        precision=4,
    )

    print("\nActive terms from shared support mask:")
    for target_idx, target_name in enumerate(target_names):
        active_idx = np.where(model.optimizer.ind_[target_idx])[0]
        active_terms = [feature_names[j] for j in active_idx]
        print(f"  {target_name}: {active_terms}")

    print("\nTrajectory-specific coefficient diagnostics:")
    coef_traj = model.optimizer.coef_trajectories_
    for i in range(coef_traj.shape[0]):
        diff_norm = np.linalg.norm(coef_traj[i] - model.optimizer.coef_)
        coef_norm = np.linalg.norm(coef_traj[i])
        print(
            f"  trajectory {i}: "
            f"coef_norm={coef_norm:.6g}, "
            f"norm(coef_traj - pooled)={diff_norm:.6g}"
        )

    # Print one trajectory-specific matrix to show the storage convention.
    print_matrix(
        "Trajectory 0 coefficients, shape (n_targets, n_features)",
        coef_traj[0],
        row_names=target_names,
        col_names=feature_names,
        precision=4,
    )


if __name__ == "__main__":
    main()
