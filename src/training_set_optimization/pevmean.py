"""
PEVmean objective for training-set optimization in PC-ridge / GBLUP models.

Convention (Akdemir & Sanchez & Jannink 2015, Akdemir & Isidro-Sanchez 2019)
---------------------------------------------------------------------------
For a feature matrix ``X`` (raw SNPs or PC scores) with target rows ``X_t``
and training rows ``X_s``, the prediction-error variance of a ridge regression
with shrinkage ``lam`` is

    PEV_t = X_t (X_s^T X_s + lam I)^{-1} X_t^T,                              (1)

up to the constant ``sigma_e^2``. Eq. (1) is the *marker form*; for k features
it requires a single ``k x k`` Cholesky factorisation and one triangular solve
per target individual. When ``k`` is much smaller than the training-set size
``n_train`` this is dramatically cheaper than the kernel-form expression
``K_tt - K_tS (K_SS + lam I)^{-1} K_St`` (Woodbury identity), which would
require an ``n_train x n_train`` Cholesky.

This module implements the marker form. For E3 we pass PC scores as the
feature matrix and use a small number of PCs (k ~ 100), making each PEVmean
evaluation effectively free relative to the GA bookkeeping.

Only genotype information (PCs of marker matrix) is used -- no phenotypes.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
from scipy.linalg import cho_factor, solve_triangular

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Marker-form PEVmean (Akdemir Eq. 1; PCs as feature matrix)
# ------------------------------------------------------------------

def _pev_per_target_marker(
    P: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Return ``PEV_t = p_t^T (P_S^T P_S + lam I)^{-1} p_t`` for each target ``t``."""
    P_train = P[train_idx]
    P_target = P[target_idx]
    k = P_train.shape[1]

    # A = P_S^T P_S + lam * I (penalises every component equally, matching the
    # ridge-with-centered-features convention; PCs are mean-centred by sklearn).
    A = P_train.T @ P_train
    A.flat[:: k + 1] += float(lam)

    try:
        c, lower = cho_factor(A, lower=True, check_finite=False, overwrite_a=True)
    except np.linalg.LinAlgError:
        # Recompute A (overwrite_a may have corrupted it) and add jitter.
        A = P_train.T @ P_train
        A.flat[:: k + 1] += float(lam)
        jitter = 1e-6 * float(np.trace(A)) / max(k, 1)
        A.flat[:: k + 1] += jitter
        c, lower = cho_factor(A, lower=True, check_finite=False, overwrite_a=True)

    # W = L^{-1} P_target^T  ->  PEV_t = ||W[:, t]||^2.
    W = solve_triangular(c, P_target.T, lower=lower, check_finite=False, overwrite_b=True)
    return np.einsum("ij,ij->j", W, W)


def pev_mean(
    P: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    lam: float,
) -> float:
    """Marker-form PEVmean: mean of ``PEV_t`` over the target individuals.

    Parameters
    ----------
    P : (n, k) feature matrix (PC scores, stacked as ``[source; target]``).
    train_idx : 1-D int array of row indices in ``P`` for the training set.
    target_idx : 1-D int array of row indices in ``P`` for the target set.
    lam : ridge shrinkage applied to ``P_S^T P_S``.

    Returns
    -------
    float -- PEVmean. Lower is better. Up to a constant ``sigma_e^2`` factor
    that does not affect the GA's argmin.
    """
    if len(train_idx) == 0 or len(target_idx) == 0:
        return float("inf")
    pev_t = _pev_per_target_marker(P, train_idx, target_idx, lam)
    return float(pev_t.mean())


def pev_per_target(
    P: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Like :func:`pev_mean` but return the per-target PEV vector."""
    if len(train_idx) == 0:
        return np.full(len(target_idx), float("inf"))
    return _pev_per_target_marker(P, train_idx, target_idx, lam)


# ------------------------------------------------------------------
# Batched evaluation
# ------------------------------------------------------------------

def pev_mean_batch(
    P: np.ndarray,
    train_subsets: Sequence[np.ndarray],
    target_idx: np.ndarray,
    lam: float,
    n_jobs: int = 1,
) -> List[float]:
    """Evaluate PEVmean for many candidate training subsets.

    Each call is independent so can be parallelised across chromosomes; ``loky``
    workers are configured with ``inner_max_num_threads=1`` so BLAS does not
    oversubscribe inside each process.
    """
    if n_jobs <= 1 or len(train_subsets) <= 1:
        return [pev_mean(P, s, target_idx, lam) for s in train_subsets]

    try:
        from joblib import Parallel, delayed
    except ImportError:
        logger.warning("joblib not available; falling back to serial PEV evaluation.")
        return [pev_mean(P, s, target_idx, lam) for s in train_subsets]

    return Parallel(
        n_jobs=n_jobs,
        backend="loky",
        inner_max_num_threads=1,
    )(
        delayed(pev_mean)(P, s, target_idx, lam) for s in train_subsets
    )


# ------------------------------------------------------------------
# BPCRR paper-style lambda (Aspheim 2024, Akdemir et al. 2019)
# ------------------------------------------------------------------

def sum_pc_variances(P_train: np.ndarray, ddof: int = 1) -> float:
    """Return ``sum_j Var(PC_j[train])`` -- used in BPCRR paper-style lambda."""
    if P_train.ndim != 2 or P_train.shape[0] < 2 or P_train.shape[1] < 1:
        return float("nan")
    v_sum = float(np.sum(np.var(P_train, axis=0, ddof=ddof)))
    if not np.isfinite(v_sum) or v_sum <= 0:
        v_sum = float(np.sum(np.var(P_train, axis=0, ddof=0)))
    return v_sum


def paper_style_lambda(
    sigma_e2: float,
    va_apriori: float,
    sum_var_pc: float,
) -> float:
    """Aspheim 2024 / Akdemir 2019 convention.

    ``sigma_u^2 = va_apriori / sum_j Var(PC_j)`` (PC-effect prior variance).
    PEVmean shrinkage is the residual-to-PC-effect variance ratio:

        lam = sigma_e^2 / sigma_u^2 = sigma_e^2 * sum_var_pc / va_apriori.
    """
    if va_apriori <= 0 or sum_var_pc <= 0:
        raise ValueError("paper_style_lambda requires positive va_apriori and sum_var_pc.")
    return float(sigma_e2) * float(sum_var_pc) / float(va_apriori)


# ------------------------------------------------------------------
# Legacy kernel-form PEVmean (kept for callers that still pre-build K).
# New code should prefer the marker-form pev_mean above.
# ------------------------------------------------------------------

def build_kernel(X: np.ndarray, dtype: np.dtype = np.float64):
    """Linear kernel K = X X^T for kernel-form PEVmean. Legacy interface."""
    X_cast = np.asarray(X, dtype=dtype)
    K = X_cast @ X_cast.T
    diag_K = np.diag(K).copy()
    return K, diag_K


def _pev_q_kernel(
    K: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    lam: float,
) -> np.ndarray:
    n_train = len(train_idx)
    K_SS = K[np.ix_(train_idx, train_idx)]
    K_ST = K[np.ix_(train_idx, target_idx)]
    K_SS.flat[:: n_train + 1] += lam

    try:
        c, lower = cho_factor(K_SS, lower=True, check_finite=False, overwrite_a=True)
    except np.linalg.LinAlgError:
        K_SS = K[np.ix_(train_idx, train_idx)]
        K_SS.flat[:: n_train + 1] += lam
        jitter = 1e-6 * np.trace(K_SS) / n_train
        K_SS.flat[:: n_train + 1] += jitter
        c, lower = cho_factor(K_SS, lower=True, check_finite=False, overwrite_a=True)

    W = solve_triangular(c, K_ST, lower=lower, check_finite=False, overwrite_b=True)
    return np.einsum("ij,ij->j", W, W)


def pev_mean_kernel(
    K: np.ndarray,
    diag_K: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    lam: float,
) -> float:
    """Kernel-form PEVmean: ``K_tt - K_tS (K_SS + lam I)^{-1} K_St`` averaged over t.

    This requires an ``n_train x n_train`` Cholesky and is the inverse-Woodbury
    of the marker form. Prefer :func:`pev_mean` when PC scores are available,
    since that runs in ``k x k`` and is orders of magnitude faster for k << n.
    """
    if len(train_idx) == 0 or len(target_idx) == 0:
        return float("inf")
    q = _pev_q_kernel(K, train_idx, target_idx, lam)
    pev_t = diag_K[target_idx] - q
    return float(pev_t.mean())
