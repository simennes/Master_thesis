"""
PEVmean objective for training-set optimization in ridge / GBLUP models.

Convention
----------
We define the linear kernel as  K = X @ X.T  (no 1/p scaling).
The ridge penalty ``alpha`` used in sklearn.linear_model.Ridge minimises

    ||y - X beta||^2 + alpha * ||beta||^2

which corresponds to the mixed-model formulation with lambda = alpha.
Hence the PEV for target individual *t* given a training set S is:

    PEV_t = K_tt - K_tS (K_SS + alpha I)^{-1} K_St

and the objective is  PEVmean = mean_t PEV_t  (to be *minimised*).

Only genotype information (X) is used – no phenotypes.
"""
from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, solve_triangular

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Kernel construction
# ------------------------------------------------------------------

def build_kernel(
    X: np.ndarray,
    dtype: np.dtype = np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the linear kernel K = X X^T and return (K, diag_K).

    Parameters
    ----------
    X : (n, p) feature matrix.
    dtype : numerical precision for the kernel (default ``float64``).
        Passing ``np.float32`` roughly halves linalg cost and is safe
        when the ridge ``alpha`` is large enough that the system stays
        well-conditioned (e.g. ``alpha=1e5`` with standardized SNPs).

    Returns
    -------
    K : (n, n) kernel matrix (``dtype``).
    diag_K : (n,) diagonal of K (``dtype``).
    """
    X_cast = np.asarray(X, dtype=dtype)
    K = X_cast @ X_cast.T                  # (n, n)
    diag_K = np.diag(K).copy()             # (n,)
    return K, diag_K


# ------------------------------------------------------------------
# PEVmean computation
# ------------------------------------------------------------------

def _pev_q(
    K: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Compute q_t = K_tS (K_SS + lam I)^{-1} K_St for all targets t.

    Uses a single triangular solve via the identity
    q_t = ||L^{-1} K_St||^2 where L L^T = K_SS + lam I.
    """
    n_train = len(train_idx)

    # Fancy indexing returns fresh arrays we can mutate.
    K_SS = K[np.ix_(train_idx, train_idx)]
    K_ST = K[np.ix_(train_idx, target_idx)]

    # A = K_SS + lam * I, modified in place to avoid a full eye() allocation.
    K_SS.flat[::n_train + 1] += lam

    try:
        c, lower = cho_factor(K_SS, lower=True, check_finite=False, overwrite_a=True)
    except np.linalg.LinAlgError:
        # Re-extract (overwrite_a may have corrupted K_SS) and retry with jitter.
        K_SS = K[np.ix_(train_idx, train_idx)]
        K_SS.flat[::n_train + 1] += lam
        jitter = 1e-6 * np.trace(K_SS) / n_train
        K_SS.flat[::n_train + 1] += jitter
        c, lower = cho_factor(K_SS, lower=True, check_finite=False, overwrite_a=True)

    # W = L^{-1} K_ST  →  q_t = ||W[:, t]||^2
    W = solve_triangular(c, K_ST, lower=lower, check_finite=False, overwrite_b=True)
    return np.einsum("ij,ij->j", W, W)


def pev_mean(
    K: np.ndarray,
    diag_K: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    lam: float,
) -> float:
    """
    Compute PEVmean for a subset of training individuals.

    Parameters
    ----------
    K : (n, n) precomputed kernel matrix.
    diag_K : (n,) diagonal of K.
    train_idx : 1-D int array of indices into K for the training set S.
    target_idx : 1-D int array of indices into K for the target set T.
    lam : regularisation parameter (= ``alpha`` from sklearn Ridge).

    Returns
    -------
    float  – mean PEV over the target individuals.  Lower is better.
    """
    n_train = len(train_idx)
    n_target = len(target_idx)
    if n_train == 0 or n_target == 0:
        return float("inf")

    q = _pev_q(K, train_idx, target_idx, lam)
    pev_t = diag_K[target_idx] - q
    return float(pev_t.mean())


def pev_per_target(
    K: np.ndarray,
    diag_K: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Like :func:`pev_mean` but return per-target PEV vector (useful for diagnostics).
    """
    n_train = len(train_idx)
    if n_train == 0:
        return np.full(len(target_idx), float("inf"))

    q = _pev_q(K, train_idx, target_idx, lam)
    return diag_K[target_idx] - q


# ------------------------------------------------------------------
# Batched evaluation
# ------------------------------------------------------------------

def pev_mean_batch(
    K: np.ndarray,
    diag_K: np.ndarray,
    train_subsets: Sequence[np.ndarray],
    target_idx: np.ndarray,
    lam: float,
    n_jobs: int = 1,
) -> List[float]:
    """
    Compute PEVmean for many candidate subsets, optionally in parallel.

    Parameters
    ----------
    K, diag_K, target_idx, lam : as in :func:`pev_mean`.
    train_subsets : list of 1-D index arrays.
    n_jobs : number of parallel worker processes.  When > 1, joblib's
        loky backend is used with ``inner_max_num_threads=1`` so BLAS
        does not oversubscribe inside each worker.

    Returns
    -------
    list of float – one PEVmean per input subset, in order.
    """
    if n_jobs <= 1 or len(train_subsets) <= 1:
        return [pev_mean(K, diag_K, s, target_idx, lam) for s in train_subsets]

    try:
        from joblib import Parallel, delayed
    except ImportError:
        logger.warning("joblib not available; falling back to serial PEV evaluation.")
        return [pev_mean(K, diag_K, s, target_idx, lam) for s in train_subsets]

    return Parallel(
        n_jobs=n_jobs,
        backend="loky",
        inner_max_num_threads=1,
    )(
        delayed(pev_mean)(K, diag_K, s, target_idx, lam) for s in train_subsets
    )
