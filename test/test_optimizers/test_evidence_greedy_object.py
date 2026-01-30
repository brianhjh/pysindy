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
    t = np.arange(0, 10, 0.01)
    dt = float(t[1] - t[0])
    x0 = np.array([-8.0, 8.0, 27.0], dtype=float)
    x = odeint(lorenz, x0, t)

    sigma_x = 1e-2
    x = x + sigma_x * np.random.normal(size=x.shape)

    # Differentiation method
    fd = FiniteDifference(
        order=6,
        d=1,
        axis=0,
        is_uniform=True,
        drop_endpoints=False,
        periodic=False,
    )

    # EvidenceGreedy optimizer
    
    opt = EvidenceGreedy(alpha=1.0, max_iter=None, unbias=False)

    # New wrapper object
    model = ps.BINDy(
        optimizer=opt,
        differentiation_method=fd,
        feature_library=ps.PolynomialLibrary(degree=2, include_bias=True),
        sigma_x=sigma_x,
    )

    print("\n=== BINDy (Lorenz) ===")
    print("sigma_x:", sigma_x)
    print("_sigma2 before fit (sentinel):", opt._sigma2)

    x_dot = fd._differentiate(x, t)
    # Fit using scalar dt
    model.fit(x, t=dt)

    print("_sigma2 after fit (mapped):", model.optimizer._sigma2)
    print("\nRecovered equations:")
    model.print(precision=3)

    # Simulate and report error
    x_sim = model.simulate(x0, t)
    rel_err = np.linalg.norm(x_sim - x) / (np.linalg.norm(x) + 1e-12)
    print("\nRelative trajectory error:", rel_err)


if __name__ == "__main__":
    main()
