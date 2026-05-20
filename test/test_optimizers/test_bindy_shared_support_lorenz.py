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
        ]
    )


def generate_lorenz_trajectory(x0, t):
    """Generate one clean Lorenz trajectory and exact x_dot values."""
    sol = solve_ivp(
        lorenz_rhs,
        (t[0], t[-1]),
        np.asarray(x0, dtype=float),
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    x = sol.y.T
    x_dot = np.array([lorenz_rhs(ti, xi) for ti, xi in zip(t, x)])
    return x, x_dot


def test_bindy_shared_support_lorenz_multiple_trajectories():
    """BINDy should fit Lorenz data using shared-support multi-trajectory mode.
        TODO: different coef & diff magnitude of noise (pass in list of sigmaX for each), 
        stay in a similar range X drastic, initial condition (can be variant)
        taking in inputs coef as inputs, length of trajectory vary it. randomize 
    """
    t = np.linspace(0.0, 1.0, 201)

    initial_conditions = [
        [-8.0, 8.0, 27.0],
        [-6.0, 7.0, 25.0],
        [-10.0, 9.0, 30.0],
    ]

    x_list = []
    x_dot_list = []

    for x0 in initial_conditions:
        x, x_dot = generate_lorenz_trajectory(x0, t)
        x_list.append(x)
        x_dot_list.append(x_dot)

    optimizer = ps.EvidenceGreedy(
        alpha=1.0,
        normalize_columns=True,
        unbias=False,
        max_iter=None,
    )

    model = ps.BINDy(
        sigma_x=1e-6,
        optimizer=optimizer,
        feature_library=ps.PolynomialLibrary(degree=2, include_bias=True),
        shared_support=True,
    )

    model.fit(x_list, t=[t, t, t], x_dot=x_dot_list)

    assert getattr(model.optimizer, "shared_support_", False) is True
    assert hasattr(model.optimizer, "coef_trajectories_")
    assert model.optimizer.coef_trajectories_.shape[0] == len(initial_conditions)
    assert model.optimizer.coef_trajectories_.shape[1] == 3
    assert model.optimizer.coef_trajectories_.shape[2] == model.optimizer.coef_.shape[1]

    np.testing.assert_array_equal(
        model.optimizer.trajectory_lengths_,
        np.array([len(t), len(t), len(t)]),
    )

    # Check that the public pooled model predicts the supplied exact derivatives
    # reasonably well on the training trajectories.
    pred_list = model.predict(x_list)

    true = np.vstack(x_dot_list)
    pred = np.vstack(pred_list)

    rel_rmse = np.sqrt(np.mean((true - pred) ** 2)) / np.sqrt(np.mean(true**2))

    assert rel_rmse < 0.2


def test_bindy_shared_support_lorenz_single_trajectory_fallback_warns():
    """shared_support=True with one trajectory should fall back with a warning."""
    t = np.linspace(0.0, 1.0, 201)
    x, x_dot = generate_lorenz_trajectory([-8.0, 8.0, 27.0], t)

    optimizer = ps.EvidenceGreedy(
        alpha=1.0,
        normalize_columns=True,
        unbias=False,
        max_iter=None,
    )

    model = ps.BINDy(
        sigma_x=1e-6,
        optimizer=optimizer,
        feature_library=ps.PolynomialLibrary(degree=2, include_bias=True),
        shared_support=True,
    )

    with np.testing.suppress_warnings() as sup:
        sup.filter(UserWarning, "BINDy: Noise is not propagated")
        sup.filter(UserWarning, "shared_support=True was requested")
        model.fit(x, t=t, x_dot=x_dot)

    assert getattr(model.optimizer, "shared_support_", None) is False
    assert not hasattr(model.optimizer, "coef_trajectories_")
