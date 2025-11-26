"""
EvidenceGreedy optimizer: greedy Bayesian evidence-based sparse regression.

This implements a backward elimination strategy that maximizes the
Bayesian log evidence for a linear model with Gaussian prior on weights:

    w ~ N(0, alpha^{-1} I)
    y | w ~ N(Theta w, sigma2 I).

For each output dimension y_j, the algorithm:

  1. Starts from the full support (all library terms active).
  2. At each step, tries removing each active term in turn.
  3. For each candidate removal, computes the log evidence
     log p(y_j | alpha, sigma2, support).
  4. Removes the term whose removal gives the largest increase in evidence.
  5. Stops when no removal increases the evidence.
"""
from __future__ import annotations

from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from .base import BaseOptimizer


class EvidenceGreedy(BaseOptimizer):
    """
    Backward evidence-based sparse regression for SINDy.

    This optimizer performs backward feature elimination driven by the
    Bayesian log evidence for a linear Gaussian model with an
    isotropic Gaussian prior on the coefficients:

        w ~ N(0, alpha^{-2} I)
        y | w ~ N(Theta w, sigma2 I).

    Parameters
    ----------
    alpha : float
        Prior precision on the coefficients (sigma_p^{-2}). Must be positive.

    sigma2 : float
        Observation noise variance (sigma^2). Must be positive.

    max_iter : int, optional (default 100)
        Maximum number of backward elimination steps. At most M - 1 steps
        are needed, where M is the number of library terms.

    normalize_columns : bool, optional (default False)
        Passed to BaseOptimizer. If True, columns of the library matrix
        are normalized before regression.

    copy_X : bool, optional (default True)
        Passed to BaseOptimizer. If True, input data are copied.

    initial_guess : array-like or None, optional (default None)
        Currently ignored by the greedy algorithm; present for API
        compatibility with BaseOptimizer.

    unbias : bool, optional (default False)
        Whether to perform an additional unregularized refit after support
        selection. For a Bayesian evidence interpretation the regularized
        posterior mean is natural, so the default is False.

    verbose : bool, optional (default False)
        If True, prints a short trace of evidence values during backward
        elimination.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        sigma2: float = 1.0,
        max_iter: int = 100,
        normalize_columns: bool = False,
        copy_X: bool = True,
        initial_guess: np.ndarray | None = None,
        unbias: bool = False,
        verbose: bool = False,
    ):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if sigma2 <= 0:
            raise ValueError("sigma2 (noise variance) must be positive.")

        self.alpha = float(alpha)
        self.sigma2 = float(sigma2)
        self.verbose = bool(verbose)

        super().__init__(
            max_iter=max_iter,
            normalize_columns=normalize_columns,
            initial_guess=initial_guess,
            copy_X=copy_X,
            unbias=unbias,
        )

    def _reduce(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Run backward evidence selection for each target dimension.

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_features)
            Library matrix Theta(X). This has already been preprocessed
            by BaseOptimizer (and may be normalized).

        y : ndarray, shape (n_samples, n_targets)
            Target derivatives.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        n_samples, n_features = x.shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_targets = y.shape[1]

        # Shared Gram matrix for all outputs
        G = x.T @ x  # (M, M)
        # y^T y for each output dimension
        yTy_all = np.sum(y**2, axis=0)  # (n_targets,)

        coef = np.zeros((n_targets, n_features), dtype=float)
        ind = np.zeros((n_targets, n_features), dtype=bool)
        # We keep a simple per-target log-evidence trace, not the full
        # coefficient path, to limit memory usage.
        all_histories: List[List[Dict[str, float]]] = []

        for j in range(n_targets):
            b = x.T @ y[:, j]  # (M,)
            yTy = float(yTy_all[j])

            coef_j, ind_j, history_j = _backward_evidence_greedy_single(
                G=G,
                b=b,
                yTy=yTy,
                n_samples=n_samples,
                alpha=self.alpha,
                sigma2=self.sigma2,
                max_iter=self.max_iter,
                verbose=self.verbose,
            )

            coef[j, :] = coef_j
            ind[j, :] = ind_j
            all_histories.append(history_j)

        self.coef_ = coef
        self.ind_ = ind
        # Minimal history: final coefficients only. We keep the log-evidence
        # trace separately if needed later.
        self.history_ = [self.coef_]


def _log_evidence_from_G(
    G_active: np.ndarray,
    b_active: np.ndarray,
    yTy: float,
    n_samples: int,
    alpha: float,
    sigma2: float,
) -> Tuple[float, np.ndarray]:
    """
    Compute log evidence and posterior mean m_N for a given active set.

    Notation:

      - y in R^T, Theta in R^{T x M}
      - alpha = sigma_p^{-2}
      - beta = sigma^{-2}
      - G = Theta^T Theta,  b = Theta^T y,  yTy = y^T y
      - Lambda = alpha I_M + beta G_active
      - m_N = beta * Lambda^{-1} b_active

    Evidence approximation:

        log p(y) =
            -1/2 [ T log(2 pi) + T log sigma^2
                   + log|Lambda| - M log alpha
                   + beta ||y - Theta m_N||^2
                   + alpha ||m_N||^2 ]

    where

        ||y - Theta m_N||^2
            = yTy - 2 m_N^T b_active + m_N^T G_active m_N

    Parameters
    ----------
    G_active : ndarray, shape (K, K)
        Gram matrix for active features.

    b_active : ndarray, shape (K,)
        Theta^T y restricted to active features.

    yTy : float
        y^T y (scalar).

    n_samples : int
        T, number of time samples.

    alpha : float
        Prior precision on weights.

    sigma2 : float
        Observation noise variance.

    Returns
    -------
    log_ev : float
        Bayesian log evidence.

    m_N : ndarray, shape (K,)
        Posterior mean coefficients for the active set.
    """
    K = G_active.shape[0]

    # Degenerate empty model: p(y) = N(0, sigma2 I)
    if K == 0:
        term1 = n_samples * np.log(2.0 * np.pi)
        term2 = n_samples * np.log(sigma2)
        term3 = (1.0 / sigma2) * yTy
        log_ev = -0.5 * (term1 + term2 + term3)
        return float(log_ev), np.zeros(0, dtype=float)

    beta = 1.0 / sigma2

    # Posterior precision and mean
    Lambda = alpha * np.eye(K) + beta * G_active

    try:
        m_N = beta * np.linalg.solve(Lambda, b_active)
    except np.linalg.LinAlgError:
        # Fallback to least-squares solve if Lambda is nearly singular
        m_N = (
            beta
            * np.linalg.lstsq(Lambda, b_active.reshape(-1, 1), rcond=None)[0].ravel()
        )

    # Residual norm using precomputed stats:
    #   ||y - Theta m_N||^2 = yTy - 2 m_N^T b_active + m_N^T G_active m_N
    residual_sq = yTy - 2.0 * float(m_N.T @ b_active) + float(m_N.T @ (G_active @ m_N))

    # log|Lambda|
    sign, logdet_Lambda = np.linalg.slogdet(Lambda)
    if sign <= 0:
        # Numerically bad model; treat as very low evidence.
        return -np.inf, m_N

    term1 = n_samples * np.log(2.0 * np.pi)
    term2 = n_samples * np.log(sigma2)
    term3 = logdet_Lambda - K * np.log(alpha)
    term4 = (1.0 / sigma2) * residual_sq
    term5 = alpha * float(m_N.T @ m_N)

    log_ev = -0.5 * (term1 + term2 + term3 + term4 + term5)
    return float(log_ev), m_N


def _backward_evidence_greedy_single(
    G: np.ndarray,
    b: np.ndarray,
    yTy: float,
    n_samples: int,
    alpha: float,
    sigma2: float,
    max_iter: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """
    Backward greedy evidence maximization for a single output dimension.

    Parameters
    ----------
    G : ndarray, shape (M, M)
        Full Gram matrix Theta^T Theta.

    b : ndarray, shape (M,)
        Full vector Theta^T y.

    yTy : float
        Scalar y^T y.

    n_samples : int
        Number of time samples T.

    alpha : float
        Prior precision on weights.

    sigma2 : float
        Observation noise variance.

    max_iter : int
        Maximum number of elimination steps. At most M - 1 steps are needed.

    verbose : bool, optional (default False)
        If True, prints a short trace of evidence values.

    Returns
    -------
    coef_full : ndarray, shape (M,)
        Final coefficient vector (zeros outside the selected support).

    active_mask : ndarray, shape (M,), dtype bool
        Boolean mask for active features.

    history : list of dict
        Diagnostics for each step:
        [{"step": ..., "support_size": ..., "log_evidence": ...}, ...]
    """
    G = np.asarray(G)
    b = np.asarray(b)

    M = G.shape[0]
    if b.shape[0] != M:
        raise ValueError("Dimensions of G and b are inconsistent.")

    # Start with full support
    active = np.ones(M, dtype=bool)
    history: List[Dict[str, float]] = []

    G_act = G
    b_act = b
    log_ev, m_N = _log_evidence_from_G(
        G_active=G_act,
        b_active=b_act,
        yTy=yTy,
        n_samples=n_samples,
        alpha=alpha,
        sigma2=sigma2,
    )

    best_log_ev = log_ev
    best_m = m_N.copy()
    best_active = active.copy()

    if verbose:
        print(
            f"[EvidenceGreedy] start: "
            f"support={np.count_nonzero(active)}, "
            f"log_evidence={best_log_ev:.3f}"
        )

    history.append(
        {
            "step": 0,
            "support_size": int(np.count_nonzero(active)),
            "log_evidence": float(best_log_ev),
        }
    )

    # At most M - 1 removals are possible
    n_steps_max = min(max_iter, max(M - 1, 0))

    for step in range(1, n_steps_max + 1):
        active_indices = np.where(active)[0]
        if active_indices.size <= 1:
            break

        best_step_log_ev = -np.inf
        best_step_idx = None
        best_step_indices = None
        best_step_m = None

        # Try removing each currently active feature
        for idx in active_indices:
            mask_candidate = active.copy()
            mask_candidate[idx] = False
            J = np.where(mask_candidate)[0]

            G_J = G[np.ix_(J, J)]
            b_J = b[J]

            log_ev_J, m_J = _log_evidence_from_G(
                G_active=G_J,
                b_active=b_J,
                yTy=yTy,
                n_samples=n_samples,
                alpha=alpha,
                sigma2=sigma2,
            )

            if log_ev_J > best_step_log_ev:
                best_step_log_ev = log_ev_J
                best_step_idx = idx
                best_step_indices = J
                best_step_m = m_J

        # If no candidate improves evidence, stop
        if best_step_log_ev <= best_log_ev:
            if verbose:
                print(
                    f"[EvidenceGreedy] stop at step {step}: "
                    f"no evidence improvement "
                    f"(current={best_log_ev:.3f}, "
                    f"best_candidate={best_step_log_ev:.3f})"
                )
            break

        # Accept the best removal
        active[best_step_idx] = False
        best_log_ev = best_step_log_ev

        best_m = np.zeros(M, dtype=float)
        best_m[best_step_indices] = best_step_m
        best_active = active.copy()

        if verbose:
            print(
                f"[EvidenceGreedy] step {step}: removed term {best_step_idx}, "
                f"support={np.count_nonzero(active)}, "
                f"log_evidence={best_log_ev:.3f}"
            )

        history.append(
            {
                "step": step,
                "removed": int(best_step_idx),
                "support_size": int(np.count_nonzero(active)),
                "log_evidence": float(best_log_ev),
            }
        )

    return best_m, best_active, history
