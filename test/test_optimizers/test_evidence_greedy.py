import numpy as np

import pysindy as ps
from pysindy.optimizers import EvidenceGreedy


def test_evidence_greedy_recovers_linear_system():
    """EvidenceGreedy should recover a simple 2D linear system.

    True system:
        x' = -2 x
        y' = 1 y
    """

    # Time grid and analytic solution
    t = np.linspace(0, 10, 1001)
    x0 = 3.0
    y0 = 0.5

    X = np.column_stack(
        [
            x0 * np.exp(-2.0 * t),
            y0 * np.exp(t),
        ]
    )

    # Linear library: [1, x, y]
    feature_library = ps.PolynomialLibrary(degree=1)

    optimizer = EvidenceGreedy(
        alpha=1.0,
        sigma2=1e-6,
        max_iter=10,
        normalize_columns=True,
        unbias=False,
        verbose=True,
    )

    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=feature_library,
    )

    model.fit(X, t=t, feature_names=["x", "y"])

    model.print()

    X_sim = model.simulate([x0, y0], t=t)
    rel_err = np.linalg.norm(X_sim - X) / np.linalg.norm(X)
    assert rel_err < 1e-2


if __name__ == "__main__":
    test_evidence_greedy_recovers_linear_system()
