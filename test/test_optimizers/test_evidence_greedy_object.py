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


def main():
    # Generate Lorenz data
    t = np.arange(0.0, 2.0, 0.002)
    dt = float(t[1] - t[0])
    x0 = np.array([-8.0, 8.0, 27.0], dtype=float)
    x = odeint(lorenz, x0, t)

    # Differentiation method
    fd = FiniteDifference(
        order=2,
        d=1,
        axis=0,
        is_uniform=True,
        drop_endpoints=False,
        periodic=False,
    )

    # EvidenceGreedy optimizer
    sigma_x = 1e-2
    opt = EvidenceGreedy(alpha=1.0, sigma2=123.456, max_iter=None, unbias=False)

    # New wrapper object
    model = ps.EvidenceGreedySINDy(
        optimizer=opt,
        differentiation_method=fd,
        feature_library=ps.PolynomialLibrary(degree=2, include_bias=True),
        sigma_x=sigma_x,
    )

    print("\n=== EvidenceGreedySINDy (Lorenz) ===")
    print("sigma_x:", sigma_x)
    print("sigma2 before fit (sentinel):", opt.sigma2)

    # Fit using scalar dt
    model.fit(x, t=dt)

    print("sigma2 after fit (mapped):", model.optimizer.sigma2)
    print("\nRecovered equations:")
    model.print(precision=3)

    # Simulate and report error
    x_sim = model.simulate(x0, t)
    rel_err = np.linalg.norm(x_sim - x) / (np.linalg.norm(x) + 1e-12)
    print("\nRelative trajectory error:", rel_err)


if __name__ == "__main__":
    main()
