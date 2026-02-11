"""EvidenceGreedy optimizer: greedy Bayesian evidence-based sparse regression."""
from __future__ import annotations

import sys
import warnings
from typing import Iterable

import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ridge_regression

from .base import _normalize_features
from .base import BaseOptimizer


class EvidenceGreedy(BaseOptimizer):
    r"""
    Backward evidence-based sparse regression for SINDy.

    This optimizer performs backward feature elimination driven by the
    Bayesian log evidence for a linear Gaussian model with an isotropic
    Gaussian prior on the coefficients.

    Shared-support multi-trajectory fitting
    ---------------------------------------
    When ``trajectory_lengths`` and ``trajectory_sigma2s`` are passed to
    ``fit(..., shared_support=True)``, the optimizer no
    longer treats concatenated trajectories as a single regression problem.
    Instead, it performs *shared-support, trajectory-specific-coefficient*
    model selection:

    * the active term set is shared across trajectories;
    * each trajectory retains its own MAP coefficients;
    * candidate removals are ranked by the sum of per-trajectory log evidences.

    For API compatibility, the public ``coef_`` stored on the optimizer is a
    pooled MAP refit on the final shared support, while the trajectory-specific
    MAP coefficients are exposed in ``coef_trajectories_``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        _sigma2: float = (np.finfo(float).eps) ** 2,
        max_iter: int | None = None,
        normalize_columns: bool = True,
        copy_X: bool = True,
        initial_guess: np.ndarray | None = None,
        unbias: bool = False,
        verbose: bool = False,
    ):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if _sigma2 <= 0:
            raise ValueError("_sigma2 (noise variance) must be positive.")

        if max_iter is None:
            max_iter = sys.maxsize
        elif max_iter <= 0:
            raise ValueError("max_iter must be a positive integer or None.")

        self.alpha = float(alpha)
        self._sigma2 = float(_sigma2)
        self.verbose = bool(verbose)

        super().__init__(
            max_iter=max_iter,
            normalize_columns=normalize_columns,
            initial_guess=initial_guess,
            copy_X=copy_X,
            unbias=unbias,
        )

    @staticmethod
    def TemporalNoisePropagation(
        differentiator,
        t,
        sigma_x: float,
    ) -> float:
        """
        Estimate the derivative noise variance ``_sigma2`` induced by a
        finite-difference differentiator.

        This reconstructs the linear differentiation operator by applying the
        differentiator to the identity matrix and averages the induced row-wise
        derivative noise variance.
        """

        t = np.asarray(t)
        if t.ndim != 1:
            raise ValueError("t must be a 1D time grid.")
        if sigma_x < 0:
            raise ValueError("sigma_x must be non-negative.")

        n_samples = t.shape[0]
        X_probe = np.eye(n_samples, dtype=float)
        L_dt = differentiator._differentiate(X_probe, t)

        if L_dt.shape != (n_samples, n_samples):
            raise RuntimeError(
                "Unexpected shape from differentiator._differentiate; "
                f"expected ({n_samples}, {n_samples}), got {L_dt.shape}."
            )

        finite_row_mask = np.all(np.isfinite(L_dt), axis=1)
        if not np.any(finite_row_mask):
            raise RuntimeError(
                "Could not find any rows of the finite-difference operator "
                "without NaNs; check differentiator settings."
            )

        row_norm_sq = np.sum(L_dt[finite_row_mask] ** 2, axis=1)
        factor = float(np.mean(row_norm_sq))
        return float(sigma_x**2 * factor)

    def _unbias(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Optional unregularized refit on the selected support.

        For shared-support multi-trajectory fitting, ``coef_`` is the
        pooled/public coefficient matrix. If ``unbias=True``, only this public
        pooled coefficient matrix is unregularized; ``coef_trajectories_``
        remains the trajectory-wise MAP estimate.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        _, n_features = x.shape
        _, n_targets = y.shape

        if self.coef_.shape != (n_targets, n_features):
            raise RuntimeError(
                "EvidenceGreedy._unbias: unexpected coef_ shape "
                f"{self.coef_.shape}, expected {(n_targets, n_features)}."
            )

        for i in range(n_targets):
            active_mask = self.ind_[i]
            if not np.any(active_mask):
                continue

            X_active = x[:, active_mask]
            y_i = y[:, i]
            optvar = LinearRegression(fit_intercept=False).fit(X_active, y_i).coef_
            self.coef_[i, active_mask] = optvar

    def _reduce(
        self,
        x: np.ndarray,
        y: np.ndarray,
        trajectory_lengths: Iterable[int] | None = None,
        trajectory_sigma2s: Iterable[float] | None = None,
        shared_support: bool = False,
    ) -> None:
        """
        Run backward evidence selection.

        Single-trajectory behaviour is unchanged.

        ``trajectory_lengths`` and ``trajectory_sigma2s`` are internal keyword
        arguments passed by ``BINDy.fit`` when ``shared_support=True``. Users
        should not provide these manually when using ``BINDy``. They are used
        only so the optimizer can split the concatenated feature and target
        matrices back into trajectory blocks for shared-support evidence
        selection.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        trajectory_lengths_list = (
            None if trajectory_lengths is None else list(trajectory_lengths)
        )
        if shared_support and trajectory_lengths_list is None:
            raise ValueError(
                "trajectory_lengths must be provided internally when "
                "shared_support=True."
            )
        if (
            shared_support
            and trajectory_lengths_list is not None
            and len(trajectory_lengths_list) <= 1
        ):
            warnings.warn(
                "shared_support=True was requested, but only one trajectory was "
                "provided. Falling back to ordinary single-trajectory fitting. "
                "Pass a list of two or more trajectories to BINDy.fit to enable "
                "shared-support multi-trajectory selection.",
                UserWarning,
            )

        use_shared_support_multi_trajectory = (
            shared_support
            and trajectory_lengths_list is not None
            and len(trajectory_lengths_list) > 1
        )

        if use_shared_support_multi_trajectory:
            self._reduce_multi_trajectory_shared_support(
                x=x,
                y=y,
                trajectory_lengths=trajectory_lengths_list,
                trajectory_sigma2s=trajectory_sigma2s,
            )
            return

        self._reduce_single_trajectory(x=x, y=y)

    def _reduce_single_trajectory(self, x: np.ndarray, y: np.ndarray) -> None:
        """Original single-trajectory / stacked-data behaviour."""
        x = np.asarray(x)
        y = np.asarray(y)

        n_samples, n_features = x.shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_targets = y.shape[1]

        if self.normalize_columns:
            y_norm, y = _normalize_features(y)

        G = x.T @ x
        B = x.T @ y
        yTy_all = np.sum(y**2, axis=0)

        coef = np.zeros((n_targets, n_features), dtype=float)
        ind = np.zeros((n_targets, n_features), dtype=bool)
        all_histories: list[list[dict[str, float]]] = []

        for j in range(n_targets):
            b = B[:, j]
            yTy = float(yTy_all[j])

            eps = float(np.finfo(float).eps)
            if (not np.isfinite(yTy)) or (yTy <= eps):
                coef[j, :] = 0.0
                ind[j, :] = False
                log_ev = _log_evidence_from_G(
                    G_active=np.zeros((0, 0), dtype=float),
                    b_active=np.zeros((0,), dtype=float),
                    yTy=yTy,
                    n_samples=n_samples,
                    alpha=self.alpha,
                    _sigma2=float(self._sigma2),
                    m_N=None,
                )
                history_j = [
                    {
                        "step": 0,
                        "support_size": 0,
                        "log_evidence": float(log_ev),
                    }
                ]
                all_histories.append(history_j)
                history_tmp = np.full((n_targets, n_features), np.nan, dtype=float)
                history_tmp[j, :] = 0.0
                self.history_.append(history_tmp)
                continue

            if self.normalize_columns:
                yn = float(y_norm[j])
                denom = max(yn * yn, eps)
                _sigma2_ = float(self._sigma2) / denom
            else:
                _sigma2_ = float(self._sigma2)

            coef_j, ind_j, history_j, coef_hist = _backward_evidence_greedy_single(
                x=x,
                y_col=y[:, j],
                G=G,
                b=b,
                yTy=yTy,
                n_samples=n_samples,
                alpha=self.alpha,
                _sigma2=_sigma2_,
                max_iter=self.max_iter,
                verbose=self.verbose,
            )

            coef[j, :] = coef_j
            ind[j, :] = ind_j
            all_histories.append(history_j)

            for i in range(np.shape(coef_hist)[1]):
                history_tmp = np.full((n_targets, n_features), np.nan, dtype=float)
                history_tmp[j, :] = coef_hist[:, i]
                self.history_.append(history_tmp)

        self.coef_ = coef
        self.ind_ = ind

        if self.normalize_columns:
            self.coef_ = self.coef_ * y_norm.reshape(-1, 1)

        self.evidence_history_ = all_histories
        self.shared_support_ = False
        self.multi_trajectory_mode_ = "stacked"

    def _reduce_multi_trajectory_shared_support(
        self,
        x: np.ndarray,
        y: np.ndarray,
        trajectory_lengths: list[int],
        trajectory_sigma2s: Iterable[float] | None,
    ) -> None:
        """
        Shared-support multi-trajectory support selection.

        The support-elimination score for a candidate support ``S`` is

            sum_i log p(y_i | Theta_i, S, alpha, sigma_i^2),

        where each trajectory gets its own posterior mean / MAP coefficients.

        For API compatibility, the public ``coef_`` is a pooled MAP refit on the
        final shared support. The trajectory-specific MAP coefficients are stored
        in ``coef_trajectories_`` with shape
        ``(n_trajectories, n_targets, n_features)``.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        n_samples, n_features = x.shape
        n_targets = y.shape[1]

        lengths, sigma2s = _validate_trajectory_info(
            n_samples=n_samples,
            trajectory_lengths=trajectory_lengths,
            trajectory_sigma2s=trajectory_sigma2s,
            default_sigma2=float(self._sigma2),
        )
        n_trajectories = len(lengths)

        if self.normalize_columns:
            y_norm, y = _normalize_features(y)
        else:
            y_norm = np.ones(n_targets, dtype=float)

        x_list = _split_by_lengths(x, lengths)
        y_list = _split_by_lengths(y, lengths)
        G_list = [xi.T @ xi for xi in x_list]
        B_list = [xi.T @ yi for xi, yi in zip(x_list, y_list, strict=True)]
        yTy_list = [np.sum(yi**2, axis=0) for yi in y_list]
        n_list = [int(xi.shape[0]) for xi in x_list]

        coef = np.zeros((n_targets, n_features), dtype=float)
        coef_trajectories = np.zeros(
            (n_trajectories, n_targets, n_features), dtype=float
        )
        ind = np.zeros((n_targets, n_features), dtype=bool)
        all_histories: list[list[dict[str, float]]] = []

        eps = float(np.finfo(float).eps)
        base_sigma2s = np.asarray(sigma2s, dtype=float)

        for j in range(n_targets):
            total_yTy = float(sum(yTy_j[j] for yTy_j in yTy_list))
            if (not np.isfinite(total_yTy)) or (total_yTy <= eps):
                coef[j, :] = 0.0
                ind[j, :] = False
                coef_trajectories[:, j, :] = 0.0
                history_j = [
                    {
                        "step": 0,
                        "support_size": 0,
                        "log_evidence": float(
                            sum(
                                _log_evidence_from_G(
                                    G_active=np.zeros((0, 0), dtype=float),
                                    b_active=np.zeros((0,), dtype=float),
                                    yTy=float(yTy_i[j]),
                                    n_samples=int(n_i),
                                    alpha=self.alpha,
                                    _sigma2=float(s2),
                                    m_N=None,
                                )
                                for yTy_i, n_i, s2 in zip(
                                    yTy_list, n_list, base_sigma2s, strict=True
                                )
                            )
                        ),
                    }
                ]
                all_histories.append(history_j)
                history_tmp = np.full((n_targets, n_features), np.nan, dtype=float)
                history_tmp[j, :] = 0.0
                self.history_.append(history_tmp)
                continue

            denom = max(float(y_norm[j]) ** 2, eps)
            sigma2s_j = base_sigma2s / denom

            b_list_j = [Bi[:, j] for Bi in B_list]
            yTy_list_j = [float(yTy_i[j]) for yTy_i in yTy_list]

            (
                coef_j,
                coef_traj_j,
                ind_j,
                history_j,
                coef_hist_j,
            ) = _backward_evidence_greedy_shared_support_single(
                G_list=G_list,
                b_list=b_list_j,
                yTy_list=yTy_list_j,
                n_samples_list=n_list,
                alpha=self.alpha,
                sigma2_list=sigma2s_j,
                max_iter=self.max_iter,
                verbose=self.verbose,
            )

            coef[j, :] = coef_j
            coef_trajectories[:, j, :] = coef_traj_j
            ind[j, :] = ind_j
            all_histories.append(history_j)

            for hist_col in range(coef_hist_j.shape[1]):
                history_tmp = np.full((n_targets, n_features), np.nan, dtype=float)
                history_tmp[j, :] = coef_hist_j[:, hist_col]
                self.history_.append(history_tmp)

        self.coef_ = coef
        self.coef_trajectories_ = coef_trajectories
        self.ind_ = ind

        if self.normalize_columns:
            scale = y_norm.reshape(-1, 1)
            self.coef_ = self.coef_ * scale
            self.coef_trajectories_ = (
                self.coef_trajectories_ * y_norm.reshape(1, -1, 1)
            )

        self.evidence_history_ = all_histories
        self.trajectory_sigma2s_ = base_sigma2s
        self.trajectory_lengths_ = np.asarray(lengths, dtype=int)
        self.shared_support_ = True
        self.multi_trajectory_mode_ = "shared_support"


# -----------------------------------------------------------------------------
# Single-trajectory helpers (existing behaviour)
# -----------------------------------------------------------------------------


def _ridge_map(
    X_active: np.ndarray,
    y_active: np.ndarray,
    alpha_prior: float,
    _sigma2: float,
    ridge_kw: dict | None = None,
) -> np.ndarray:
    """Compute the MAP coefficients for a given active set using ridge."""
    X_active = np.asarray(X_active)
    y_active = np.asarray(y_active).ravel()

    lam = alpha_prior * _sigma2
    kw = ridge_kw or {}

    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("always", category=LinAlgWarning)
        coef = ridge_regression(X_active, y_active, lam, **kw)

    for w in caught:
        if issubclass(w.category, LinAlgWarning):
            warnings.warn(
                "EvidenceGreedy: linear algebra warning encountered while "
                "computing MAP coefficients; results may be unreliable.",
                RuntimeWarning,
            )
            break

    return coef


# -----------------------------------------------------------------------------
# Shared-support multi-trajectory helpers
# -----------------------------------------------------------------------------


def _validate_trajectory_info(
    n_samples: int,
    trajectory_lengths: list[int],
    trajectory_sigma2s: Iterable[float] | None,
    default_sigma2: float,
) -> tuple[list[int], np.ndarray]:
    """Validate trajectory partition and noise list."""
    if trajectory_lengths is None or len(trajectory_lengths) == 0:
        raise ValueError(
            "trajectory_lengths must be provided internally by BINDy.fit when "
            "shared_support=True. When using BINDy, pass multiple trajectories "
            "as a list of arrays rather than one concatenated array."
        )

    lengths = [int(v) for v in trajectory_lengths]
    if any(v <= 0 for v in lengths):
        raise ValueError("All trajectory lengths must be positive integers.")
    if sum(lengths) != int(n_samples):
        raise ValueError(
            "trajectory_lengths must sum to the concatenated sample count. "
            f"Expected {n_samples}, got {sum(lengths)}."
        )

    if trajectory_sigma2s is None:
        sigma2s = np.full(len(lengths), float(default_sigma2), dtype=float)
    else:
        sigma2s = np.asarray(list(trajectory_sigma2s), dtype=float)
        if sigma2s.ndim != 1 or sigma2s.shape[0] != len(lengths):
            raise ValueError(
                "trajectory_sigma2s must be a 1D iterable with the same length "
                "as trajectory_lengths."
            )
        if np.any(~np.isfinite(sigma2s)) or np.any(sigma2s <= 0):
            raise ValueError("All trajectory_sigma2s must be finite and positive.")

    return lengths, sigma2s


def _split_by_lengths(arr: np.ndarray, lengths: list[int]) -> list[np.ndarray]:
    """Split a concatenated array into consecutive trajectory blocks."""
    arr = np.asarray(arr)
    parts = []
    start = 0
    for n in lengths:
        stop = start + int(n)
        parts.append(arr[start:stop])
        start = stop
    return parts


def _ridge_map_from_stats(
    G_active: np.ndarray,
    b_active: np.ndarray,
    alpha_prior: float,
    _sigma2: float,
) -> np.ndarray:
    """
    MAP coefficients from sufficient statistics.

    Solves
        (G + alpha * sigma^2 * I) m = b.
    """
    G_active = np.asarray(G_active, dtype=float)
    b_active = np.asarray(b_active, dtype=float).reshape(-1)

    K = G_active.shape[0]
    if K == 0:
        return np.zeros((0,), dtype=float)

    A = G_active + (alpha_prior * _sigma2) * np.eye(K)
    try:
        return np.linalg.solve(A, b_active)
    except np.linalg.LinAlgError:
        warnings.warn(
            "EvidenceGreedy: singular/ill-conditioned normal system in "
            "_ridge_map_from_stats; falling back to least squares.",
            RuntimeWarning,
        )
        return np.linalg.lstsq(A, b_active, rcond=None)[0]


def _pooled_map_shared_support_coefficients(
    active_mask: np.ndarray,
    G_list: list[np.ndarray],
    b_list: list[np.ndarray],
    sigma2_list: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute public pooled MAP coefficients on the final shared support."""
    M = G_list[0].shape[0]
    J = np.where(active_mask)[0]
    coef_full = np.zeros(M, dtype=float)

    if J.size == 0:
        return coef_full

    A = alpha * np.eye(J.size)
    h = np.zeros(J.size, dtype=float)

    for G_i, b_i, sigma2_i in zip(G_list, b_list, sigma2_list, strict=True):
        beta_i = 1.0 / float(sigma2_i)
        A += beta_i * G_i[np.ix_(J, J)]
        h += beta_i * b_i[J]

    try:
        m = np.linalg.solve(A, h)
    except np.linalg.LinAlgError:
        warnings.warn(
            "EvidenceGreedy: singular pooled MAP system in shared-support "
            "aggregation; falling back to least squares.",
            RuntimeWarning,
        )
        m = np.linalg.lstsq(A, h, rcond=None)[0]

    coef_full[J] = m
    return coef_full


def _backward_evidence_greedy_shared_support_single(
    G_list: list[np.ndarray],
    b_list: list[np.ndarray],
    yTy_list: list[float],
    n_samples_list: list[int],
    alpha: float,
    sigma2_list: np.ndarray,
    max_iter: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]], np.ndarray]:
    """
    Shared-support multi-trajectory support search for a single output dimension.

    Shared support is selected by maximizing the *sum* of per-trajectory log
    evidences, while each trajectory keeps its own MAP coefficients.

    Returns
    -------
    coef_public : ndarray, shape (M,)
        Public/predictive coefficient vector. In this branch it is a pooled MAP
        refit on the final shared support.
    coef_trajectories : ndarray, shape (n_trajectories, M)
        Trajectory-specific MAP coefficients on the final shared support.
    active_mask : ndarray, shape (M,), dtype bool
        Shared support mask.
    history : list of dict
        Evidence trace with total and per-trajectory evidence values.
    coef_hist : ndarray, shape (M, n_steps_recorded)
        History of the pooled/public coefficient vector.
    """
    if len(G_list) == 0:
        raise ValueError("At least one trajectory is required.")

    M = G_list[0].shape[0]
    n_trajectories = len(G_list)
    if (
        len(b_list) != n_trajectories
        or len(yTy_list) != n_trajectories
        or len(n_samples_list) != n_trajectories
    ):
        raise ValueError("Trajectory statistics lists must all have equal length.")

    active = np.ones(M, dtype=bool)
    history: list[dict[str, float]] = []

    coef_traj_full = np.zeros((n_trajectories, M), dtype=float)
    log_evs = []
    for i, (G_i, b_i, yTy_i, n_i, sigma2_i) in enumerate(
        zip(G_list, b_list, yTy_list, n_samples_list, sigma2_list, strict=True)
    ):
        m_i = _ridge_map_from_stats(
            G_i, b_i, alpha_prior=alpha, _sigma2=float(sigma2_i)
        )
        coef_traj_full[i, :] = m_i
        log_ev_i = _log_evidence_from_G(
            G_active=G_i,
            b_active=b_i,
            yTy=float(yTy_i),
            n_samples=int(n_i),
            alpha=alpha,
            _sigma2=float(sigma2_i),
            m_N=m_i,
        )
        log_evs.append(float(log_ev_i))

    best_log_ev = float(np.sum(log_evs))
    best_coef_traj = coef_traj_full.copy()
    best_active = active.copy()
    best_coef_public = _pooled_map_shared_support_coefficients(
        active_mask=best_active,
        G_list=G_list,
        b_list=b_list,
        sigma2_list=sigma2_list,
        alpha=alpha,
    )

    if verbose:
        print(
            "[EvidenceGreedy][shared_support] start: "
            f"support={np.count_nonzero(active)}, "
            f"log_evidence_total={best_log_ev:.3f}"
        )

    history.append(
        {
            "step": 0,
            "support_size": int(np.count_nonzero(active)),
            "log_evidence": float(best_log_ev),
            "log_evidence_per_trajectory": [float(v) for v in log_evs],
        }
    )

    coef_hist = [best_coef_public.copy()]
    n_steps_max = min(max_iter, max(M - 1, 0))

    for step in range(1, n_steps_max + 1):
        active_indices = np.where(active)[0]
        if active_indices.size <= 1:
            break

        best_step_log_ev = -np.inf
        best_step_idx: int | None = None
        best_step_coef_traj: np.ndarray | None = None
        best_step_per_traj_logs: list[float] | None = None

        for idx in active_indices:
            mask_candidate = active.copy()
            mask_candidate[idx] = False
            J = np.where(mask_candidate)[0]

            cand_coef_traj = np.zeros((n_trajectories, M), dtype=float)
            cand_logs = []

            for traj_i, (G_i, b_i, yTy_i, n_i, sigma2_i) in enumerate(
                zip(G_list, b_list, yTy_list, n_samples_list, sigma2_list, strict=True)
            ):
                if J.size == 0:
                    log_ev_i = _log_evidence_from_G(
                        G_active=np.zeros((0, 0), dtype=float),
                        b_active=np.zeros((0,), dtype=float),
                        yTy=float(yTy_i),
                        n_samples=int(n_i),
                        alpha=alpha,
                        _sigma2=float(sigma2_i),
                        m_N=None,
                    )
                else:
                    G_J = G_i[np.ix_(J, J)]
                    b_J = b_i[J]
                    m_J = _ridge_map_from_stats(
                        G_active=G_J,
                        b_active=b_J,
                        alpha_prior=alpha,
                        _sigma2=float(sigma2_i),
                    )
                    log_ev_i = _log_evidence_from_G(
                        G_active=G_J,
                        b_active=b_J,
                        yTy=float(yTy_i),
                        n_samples=int(n_i),
                        alpha=alpha,
                        _sigma2=float(sigma2_i),
                        m_N=m_J,
                    )
                    cand_coef_traj[traj_i, J] = m_J

                cand_logs.append(float(log_ev_i))

            log_ev_total = float(np.sum(cand_logs))
            if log_ev_total > best_step_log_ev:
                best_step_log_ev = log_ev_total
                best_step_idx = int(idx)
                best_step_coef_traj = cand_coef_traj
                best_step_per_traj_logs = cand_logs

        if best_step_idx is None or best_step_log_ev <= best_log_ev:
            if verbose:
                print(
                    f"[EvidenceGreedy][shared_support] stop at step {step}: "
                    f"no evidence improvement (current={best_log_ev:.3f}, "
                    f"best_candidate={best_step_log_ev:.3f})"
                )
            break

        active[best_step_idx] = False
        best_log_ev = float(best_step_log_ev)
        best_coef_traj = best_step_coef_traj.copy()
        best_active = active.copy()
        best_coef_public = _pooled_map_shared_support_coefficients(
            active_mask=best_active,
            G_list=G_list,
            b_list=b_list,
            sigma2_list=sigma2_list,
            alpha=alpha,
        )
        coef_hist.append(best_coef_public.copy())

        if verbose:
            print(
                (
                    f"[EvidenceGreedy][shared_support] step {step}: removed term "
                    f"{best_step_idx}, "
                    f"support={np.count_nonzero(active)}, "
                    f"log_evidence_total={best_log_ev:.3f}"
                )
            )

        history.append(
            {
                "step": step,
                "removed": int(best_step_idx),
                "support_size": int(np.count_nonzero(active)),
                "log_evidence": float(best_log_ev),
                "log_evidence_per_trajectory": [
                    float(v) for v in best_step_per_traj_logs
                ],
            }
        )

    return (
        best_coef_public,
        best_coef_traj,
        best_active,
        history,
        np.column_stack(coef_hist),
    )


# -----------------------------------------------------------------------------
# Shared utility functions
# -----------------------------------------------------------------------------


def _log_evidence_from_G(
    G_active: np.ndarray,
    b_active: np.ndarray,
    yTy: float,
    n_samples: int,
    alpha: float,
    _sigma2: float,
    m_N: np.ndarray | None,
) -> float:
    r"""Compute the Bayesian log evidence for a given active set and MAP."""
    G_active = np.asarray(G_active)
    b_active = np.asarray(b_active)

    K = G_active.shape[0]

    if K == 0:
        term1 = n_samples * np.log(2.0 * np.pi)
        term2 = n_samples * np.log(_sigma2)
        term3 = (1.0 / _sigma2) * yTy
        log_ev = -0.5 * (term1 + term2 + term3)
        return float(log_ev)

    if m_N is None:
        raise ValueError("m_N must be provided for a non-empty active set.")

    m_N = np.asarray(m_N).reshape(-1)
    if m_N.shape[0] != K:
        raise ValueError("m_N has incompatible shape for the active set.")

    beta = 1.0 / _sigma2
    residual_sq = yTy - 2.0 * float(m_N.T @ b_active) + float(m_N.T @ (G_active @ m_N))

    Lambda = alpha * np.eye(K) + beta * G_active
    sign, logdet_Lambda = np.linalg.slogdet(Lambda)
    if sign <= 0:
        return float(-np.inf)

    term1 = n_samples * np.log(2.0 * np.pi)
    term2 = n_samples * np.log(_sigma2)
    term3 = logdet_Lambda - K * np.log(alpha)
    term4 = (1.0 / _sigma2) * residual_sq
    term5 = alpha * float(m_N.T @ m_N)

    log_ev = -0.5 * (term1 + term2 + term3 + term4 + term5)
    return float(log_ev)


def _backward_evidence_greedy_single(
    x: np.ndarray,
    y_col: np.ndarray,
    G: np.ndarray,
    b: np.ndarray,
    yTy: float,
    n_samples: int,
    alpha: float,
    _sigma2: float,
    max_iter: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], np.ndarray]:
    """Original single-output, single-problem backward evidence search."""
    x = np.asarray(x)
    y_col = np.asarray(y_col).ravel()
    G = np.asarray(G)
    b = np.asarray(b)

    n_samples_x, M = x.shape
    if n_samples_x != n_samples:
        raise ValueError("Mismatch between n_samples and x.shape[0].")
    if G.shape != (M, M):
        raise ValueError("G must have shape (M, M).")
    if b.shape[0] != M:
        raise ValueError("Dimensions of G and b are inconsistent.")

    active = np.ones(M, dtype=bool)
    history: list[dict[str, float]] = []

    J_full = np.where(active)[0]
    m_full = _ridge_map(x[:, J_full], y_col, alpha_prior=alpha, _sigma2=_sigma2)
    log_ev = _log_evidence_from_G(
        G_active=G,
        b_active=b,
        yTy=yTy,
        n_samples=n_samples,
        alpha=alpha,
        _sigma2=_sigma2,
        m_N=m_full,
    )

    best_log_ev = log_ev
    best_m = np.zeros(M, dtype=float)
    best_m[J_full] = m_full
    best_active = active.copy()

    if verbose:
        print(
            f"[EvidenceGreedy] start: support={np.count_nonzero(active)}, "
            f"log_evidence={best_log_ev:.3f}"
        )

    history.append(
        {
            "step": 0,
            "support_size": int(np.count_nonzero(active)),
            "log_evidence": float(best_log_ev),
        }
    )

    if max_iter is None:
        n_steps_max = max(M - 1, 0)
    else:
        n_steps_max = min(max_iter, max(M - 1, 0))

    m_hist = [best_m.copy()]

    for step in range(1, n_steps_max + 1):
        active_indices = np.where(active)[0]
        if active_indices.size <= 1:
            break

        best_step_log_ev = -np.inf
        best_step_idx: int | None = None
        best_step_m_full: np.ndarray | None = None

        for idx in active_indices:
            mask_candidate = active.copy()
            mask_candidate[idx] = False
            J = np.where(mask_candidate)[0]

            if J.size == 0:
                log_ev_J = _log_evidence_from_G(
                    G_active=G[np.ix_(J, J)],
                    b_active=b[J],
                    yTy=yTy,
                    n_samples=n_samples,
                    alpha=alpha,
                    _sigma2=_sigma2,
                    m_N=None,
                )
                m_full_candidate = np.zeros(M, dtype=float)
            else:
                G_J = G[np.ix_(J, J)]
                b_J = b[J]
                m_J = _ridge_map(x[:, J], y_col, alpha_prior=alpha, _sigma2=_sigma2)
                log_ev_J = _log_evidence_from_G(
                    G_active=G_J,
                    b_active=b_J,
                    yTy=yTy,
                    n_samples=n_samples,
                    alpha=alpha,
                    _sigma2=_sigma2,
                    m_N=m_J,
                )
                m_full_candidate = np.zeros(M, dtype=float)
                m_full_candidate[J] = m_J

            if log_ev_J > best_step_log_ev:
                best_step_log_ev = log_ev_J
                best_step_idx = int(idx)
                best_step_m_full = m_full_candidate

        if best_step_log_ev <= best_log_ev or best_step_idx is None:
            if verbose:
                print(
                    (
                        f"[EvidenceGreedy] stop at step {step}: no evidence "
                        f"improvement "
                        f"(current={best_log_ev:.3f}, "
                        f"best_candidate={best_step_log_ev:.3f})"
                    )
                )
            break

        active[best_step_idx] = False
        best_log_ev = best_step_log_ev
        best_m = best_step_m_full
        best_active = active.copy()
        m_hist.append(best_m.copy())

        if verbose:
            print(
                f"[EvidenceGreedy] step {step}: removed term {best_step_idx}, "
                f"support={np.count_nonzero(active)}, log_evidence={best_log_ev:.3f}"
            )

        history.append(
            {
                "step": step,
                "removed": int(best_step_idx),
                "support_size": int(np.count_nonzero(active)),
                "log_evidence": float(best_log_ev),
            }
        )

    return best_m, best_active, history, np.column_stack(m_hist)
