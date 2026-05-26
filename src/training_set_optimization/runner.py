"""
PEVmean-GA training-set optimisation experiment runner.

Integrates PEVmean-based training-set selection (via genetic algorithm) into
the same island-wise experiment framework used by TMC-Shapley.

For each target island / training-set size the pipeline:
  1. Candidate pool C = all non-target islands.
     Target set T = all individuals on the target island.
  2. Builds a linear kernel K = X X^T  once on (C ∪ T).
  3. Runs the GA to select S ⊂ C  of the desired size, minimising PEVmean
     (genotype-only objective — no phenotypes used for selection).
  4. Fits ridge regression on S  (using y) with the existing code path.
  5. Evaluates on T with Pearson-r / MSE — identical to Shapley experiments.
  6. Stores results in the same CSV schema so comparison plots work out-of-the-box.

No calibration / test split of the target island is needed because PEVmean
uses only genotypes (X), not phenotypes (y), for training-set selection.

Convention:  lambda = ridge_alpha  (see pevmean.py docstring for derivation).
"""
from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..utils import _pearson_corr
from .ga_subset import GAConfig, run_ga
from .pevmean import build_kernel, pev_mean_kernel

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Single evaluation helper (mirrors _evaluate_individual_subset)
# ------------------------------------------------------------------

def _evaluate_ridge_subset(
    train_idx: np.ndarray,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    alpha: float,
    snp_cols: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Fit ridge on ``X_source[train_idx]`` and evaluate on the test set.

    Mirrors the ridge branch of
    ``tmc_shapley_islands._evaluate_individual_subset``.
    """
    if len(train_idx) < 2:
        return {"corr_eval": 0.0, "mse_adj": float("inf")}

    X_train = X_source[train_idx]
    y_train = y_source[train_idx]

    if snp_cols is not None:
        X_train = X_train[:, snp_cols]
        X_test_sel = X_test[:, snp_cols]
    else:
        X_test_sel = X_test

    model = Ridge(alpha=max(float(alpha), 1e-12))
    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        sw = np.asarray(sample_weight, dtype=float)
        if sw.shape[0] != y_train.shape[0]:
            raise ValueError("sample_weight length must match selected training subset length")
        model.fit(X_train, y_train, sample_weight=sw)
    pred = model.predict(X_test_sel)

    corr_eval = float(_pearson_corr(pred, y_eval_test))
    if not np.isfinite(corr_eval):
        corr_eval = 0.0
    mse_adj = float(np.mean((pred - y_test) ** 2))
    return {"corr_eval": corr_eval, "mse_adj": mse_adj}


# ------------------------------------------------------------------
# PEVmean-GA optimiser for one (target, n_train) setting
# ------------------------------------------------------------------

def run_pevmean_ga(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    n_train: int,
    ridge_alpha: float,
    ga_cfg: GAConfig,
    snp_cols: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run PEVmean-GA for one setting and evaluate the chosen training set.

    Parameters
    ----------
    X_source : (N_cand, p) candidate genotypes (already preprocessed).
    y_source : (N_cand,) candidate phenotypes (used only for ridge fit).
    X_test   : (N_test, p) target genotypes.
    y_test   : (N_test,) adjusted target phenotypes (for MSE).
    y_eval_test : (N_test,) original target phenotypes (for Pearson-r).
    n_train  : desired training-set size.
    ridge_alpha : regularisation parameter for ridge & PEVmean.
    ga_cfg   : GA hyper-parameters.
    snp_cols : optional SNP column indices for feature selection.

    Returns
    -------
    dict with keys: best_subset, best_pevmean, corr_eval, mse_adj, ga_stats.
    """
    N_cand = X_source.shape[0]
    N_test = X_test.shape[0]

    # ---- Feature selection for kernel -----------------------------------------
    if snp_cols is not None:
        X_cand_sel = X_source[:, snp_cols]
        X_test_sel = X_test[:, snp_cols]
    else:
        X_cand_sel = X_source
        X_test_sel = X_test

    # ---- Precompute kernel on candidate ∪ target union -----------------------
    X_all = np.vstack([X_cand_sel, X_test_sel])      # (N_cand+N_test, p)
    K, diag_K = build_kernel(X_all)

    cand_idx = np.arange(N_cand, dtype=np.int64)
    target_idx = np.arange(N_cand, N_cand + N_test, dtype=np.int64)

    logger.info(
        "PEVmean-GA: n_cand=%d, n_test=%d, n_train=%d, alpha=%.2e, K shape=%s",
        N_cand, N_test, n_train, ridge_alpha, K.shape,
    )

    # ---- Fitness function (indices are into K, i.e. 0..N_cand-1) -------------
    def fitness_fn(subset: np.ndarray) -> float:
        return pev_mean_kernel(K, diag_K, subset, target_idx, lam=ridge_alpha)

    # ---- Run GA ---------------------------------------------------------------
    best_subset, best_fitness, ga_stats = run_ga(
        n_candidates=N_cand,
        n_train=n_train,
        fitness_fn=fitness_fn,
        cfg=ga_cfg,
        candidate_indices=cand_idx,
    )

    # best_subset contains indices into K (== indices into X_source since
    # candidates occupy rows 0..N_cand-1).

    # ---- Evaluate ridge on the selected subset --------------------------------
    eval_result = _evaluate_ridge_subset(
        train_idx=best_subset,
        X_source=X_source,
        y_source=y_source,
        X_test=X_test,
        y_test=y_test,
        y_eval_test=y_eval_test,
        alpha=ridge_alpha,
        snp_cols=snp_cols,
    )

    return {
        "best_subset": best_subset,
        "best_pevmean": best_fitness,
        "corr_eval": eval_result["corr_eval"],
        "mse_adj": eval_result["mse_adj"],
        "ga_stats": ga_stats,
    }


# ------------------------------------------------------------------
# Full experiment: multiple n_train sizes => add-curve-like output
# ------------------------------------------------------------------

def run_pevmean_experiment(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    locality: np.ndarray,
    y_eval: np.ndarray,
    target_island_code: int,
    target_island_name: str,
    source_codes: List[int],
    ridge_alpha: float,
    ga_cfg: GAConfig,
    step_counts: np.ndarray,
    snp_cols: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run PEVmean-GA experiment for one target island, producing a DataFrame
    compatible with the Shapley add/remove-curve schema.

    No calibration / test split is performed: the target set T is the
    **full** target island (PEVmean uses only genotypes for selection).

    Parameters
    ----------
    step_counts : 1-D int array
        Training-set sizes at each step.
    """
    # ---- Separate source (candidates) and target (full island) ----------------
    target_mask = (locality == target_island_code)
    source_mask = ~target_mask & np.isin(locality, source_codes)

    X_source = X[source_mask]
    y_source = y[source_mask]
    locality_source = locality[source_mask]

    X_target = X[target_mask]
    y_target = y[target_mask]       # adjusted phenotype (for MSE)
    y_eval_target = y_eval[target_mask]  # original phenotype (for Pearson r)

    N_source = len(X_source)

    logger.info(
        "PEVmean experiment: target=%s (%s), source=%d, target=%d",
        target_island_code, target_island_name, N_source, len(X_target),
    )

    rows: List[Dict[str, Any]] = []

    for step_i, n_train in enumerate(step_counts):
        n_train = int(min(n_train, N_source))
        if n_train < 2:
            rows.append({
                "n_individuals": n_train,
                "corr_eval": 0.0,
                "mse_adj": float("inf"),
                "method": "pevmean_ga",
                "order_seed": -1,
                "pevmean_obj": float("inf"),
            })
            continue

        # Different GA seed per step to reduce correlation
        step_ga_cfg = copy.deepcopy(ga_cfg)
        step_ga_cfg.seed = seed + step_i * 7919

        result = run_pevmean_ga(
            X_source=X_source,
            y_source=y_source,
            X_test=X_target,
            y_test=y_target,
            y_eval_test=y_eval_target,
            n_train=n_train,
            ridge_alpha=ridge_alpha,
            ga_cfg=step_ga_cfg,
            snp_cols=snp_cols,
        )

        rows.append({
            "n_individuals": n_train,
            "corr_eval": result["corr_eval"],
            "mse_adj": result["mse_adj"],
            "method": "pevmean_ga",
            "order_seed": -1,
            "pevmean_obj": result["best_pevmean"],
        })
        logger.info(
            "PEVmean step %d/%d: n_train=%d  PEVmean=%.6f  corr=%.4f",
            step_i + 1, len(step_counts), n_train,
            result["best_pevmean"], result["corr_eval"],
        )

    curve_df = pd.DataFrame(rows)
    return {
        "pevmean_curve_df": curve_df,
        "target_island_code": target_island_code,
        "target_island_name": target_island_name,
    }
