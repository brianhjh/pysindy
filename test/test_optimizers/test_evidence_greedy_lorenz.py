import numpy as np
from scipy.integrate import odeint

import pysindy as ps
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import EvidenceGreedy


def lorenz(z, t):
    """Standard Lorenz system."""
    x, y, z_ = z
    return [
        10.0 * (y - x),
        x * (28.0 - z_) - y,
        x * y - 8.0 / 3.0 * z_,
    ]


def test_evidence_greedy_lorenz_example():
    """Check that EvidenceGreedy can reasonably recover the Lorenz dynamics.

    This mirrors the Lorenz example in the EvidenceGreedy docstring.
    """
    # Time grid and data (same as in the docstring)
    t = np.arange(0, 2, 0.002)
    x0 = [-8.0, 8.0, 27.0]
    x = odeint(lorenz, x0, t)

    fd = FiniteDifference(
        order=2,
        d=1,
        axis=0,
        is_uniform=True,
        drop_endpoints=False,
        periodic=False,
    )

    sigma_x = 1e-2
    sigma2 = EvidenceGreedy.finite_difference_sigma2(fd, t, sigma_x)

    # EvidenceGreedy optimizer with the same hyperparameters as the docstring
    opt = EvidenceGreedy(
        alpha=1.0,
        sigma2=sigma2,
        max_iter=20,
    )

    model = ps.SINDy(optimizer=opt)
    model.fit(x, t=t[1] - t[0])
    model.print()

    # Forward simulate from the same initial condition
    x_sim = model.simulate(x0, t)

    # Check that the learned model reproduces the trajectory reasonably well
    rel_err = np.linalg.norm(x_sim - x) / np.linalg.norm(x)

    # Reasonable recovery tolerance
    assert rel_err < 1e-1


if __name__ == "__main__":
    test_evidence_greedy_lorenz_example()
