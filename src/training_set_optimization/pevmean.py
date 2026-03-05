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
from typing import Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Kernel construction
# ------------------------------------------------------------------

def build_kernel(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the linear kernel K = X X^T and return (K, diag_K).

    Parameters
    ----------
    X : (n, p) feature matrix (float32 or float64).

    Returns
    -------
    K : (n, n) kernel matrix (float64).
    diag_K : (n,) diagonal of K (float64).
    """
    X64 = np.asarray(X, dtype=np.float64)
    K = X64 @ X64.T                       # (n, n)
    diag_K = np.diag(K).copy()             # (n,)
    return K, diag_K


# ------------------------------------------------------------------
# PEVmean computation
# ------------------------------------------------------------------

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

    # K_SS  (n_train, n_train)  and  K_ST  (n_train, n_target)
    K_SS = K[np.ix_(train_idx, train_idx)]
    K_ST = K[np.ix_(train_idx, target_idx)]

    # A = K_SS + lam * I
    A = K_SS + lam * np.eye(n_train, dtype=K.dtype)

    # Cholesky factorization – numerically stable solve
    try:
        L, lower = cho_factor(A, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        # Fallback: add jitter and retry
        jitter = 1e-6 * np.trace(A) / n_train
        A += jitter * np.eye(n_train, dtype=A.dtype)
        L, lower = cho_factor(A, lower=True, check_finite=False)

    # M = A^{-1} K_ST   shape (n_train, n_target)
    M = cho_solve((L, lower), K_ST, check_finite=False)

    # q_t = K_tS A^{-1} K_St  =  sum_j K_ST[j,t] * M[j,t]
    q = np.einsum("ij,ij->j", K_ST, M)            # (n_target,)

    # PEV_t = K_tt - q_t
    pev_t = diag_K[target_idx] - q                 # (n_target,)

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

    K_SS = K[np.ix_(train_idx, train_idx)]
    K_ST = K[np.ix_(train_idx, target_idx)]
    A = K_SS + lam * np.eye(n_train, dtype=K.dtype)

    try:
        L, lower = cho_factor(A, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        jitter = 1e-6 * np.trace(A) / n_train
        A += jitter * np.eye(n_train, dtype=A.dtype)
        L, lower = cho_factor(A, lower=True, check_finite=False)

    M = cho_solve((L, lower), K_ST, check_finite=False)
    q = np.einsum("ij,ij->j", K_ST, M)
    return diag_K[target_idx] - q
