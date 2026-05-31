"""Shared helpers for "use PC scores as features" in nested CV runners.

When a config contains a ``feature_pca`` block with ``enabled=true``, the
runner replaces its feature matrix X with PC scores Z computed once on the
full filtered dataset. PCA is genotype-only, so no phenotype leakage. The
existing ``num_snps`` search-space knob is hijacked: it now picks the top-k
PC columns (by explained variance, i.e. natural PCA order), not top-k SNPs
by correlation with y.

Usage in a runner
-----------------
    from src.pc_feature import (
        maybe_apply_pca, pca_active, suggest_pc_count,
    )

    pca_state = maybe_apply_pca(X, config)
    if pca_active(pca_state):
        X = pca_state["Z"]

    # ... in the Optuna objective:
    if pca_active(pca_state):
        n_pcs = suggest_pc_count(trial, pca_state)
        snp_cols = np.arange(n_pcs, dtype=np.int64)
        # ... use snp_cols in _evaluate_ridge_subset
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import optuna
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def _resolve_pca_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """Look for `feature_pca` at top-level or inside `base_train`."""
    if isinstance(config.get("feature_pca"), dict):
        return dict(config["feature_pca"])
    base = config.get("base_train", {})
    if isinstance(base, dict) and isinstance(base.get("feature_pca"), dict):
        return dict(base["feature_pca"])
    return {}


def maybe_apply_pca(X: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute PCs if `feature_pca.enabled`. Returns a state dict.

    State keys:
        enabled    : bool
        Z          : np.ndarray of shape (n_samples, n_pcs_fit) when enabled, else None
        n_pcs_fit  : int total PCs fit (= upper bound on per-trial r)
        n_pcs_range / n_pcs_choices / n_pcs_fixed : how trials should pick r
    """
    cfg = _resolve_pca_cfg(config)
    enabled = bool(cfg.get("enabled", False))
    state: Dict[str, Any] = {"enabled": enabled, "Z": None, "n_pcs_fit": 0, "cfg": cfg}
    if not enabled:
        return state

    fixed = cfg.get("n_pcs_fixed", None)
    choices = cfg.get("n_pcs_choices", None)
    range_ = cfg.get("n_pcs_range", None)

    if fixed is not None:
        max_needed = int(fixed)
    elif choices is not None:
        max_needed = int(max(int(c) for c in choices))
    elif range_ is not None:
        max_needed = int(range_[1])
    else:
        raise ValueError(
            "feature_pca.enabled=true requires one of: n_pcs_fixed, n_pcs_choices, n_pcs_range"
        )

    requested_fit = int(cfg.get("n_pcs_fit", max_needed))
    n_components = int(min(requested_fit, X.shape[0], X.shape[1]))
    if n_components < 1:
        raise ValueError("feature_pca: cannot fit any PCs (n_samples/n_features too small)")
    if n_components < max_needed:
        logger.warning(
            "feature_pca: n_pcs_fit clamped from %d to %d by data dimensions",
            max_needed, n_components,
        )

    standardize = bool(cfg.get("standardize_for_pca", False))
    seed = int(cfg.get("pca_seed", config.get("seed", 14)))

    if standardize:
        mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = X.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std == 0] = 1.0
        X_in = (X - mean) / std
    else:
        X_in = X

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    t0 = time.perf_counter()
    Z = pca.fit_transform(X_in).astype(np.float32, copy=False)
    elapsed = float(time.perf_counter() - t0)
    cumvar = float(np.sum(pca.explained_variance_ratio_))
    logger.info(
        "feature_pca: fit PCA on (n=%d, p=%d) -> r=%d in %.1fs (cumvar=%.4f, standardize=%s)",
        X.shape[0], X.shape[1], n_components, elapsed, cumvar, standardize,
    )

    state["Z"] = Z
    state["n_pcs_fit"] = int(n_components)
    state["cumvar_explained"] = cumvar
    return state


def pca_active(state: Optional[Dict[str, Any]]) -> bool:
    return bool(state and state.get("enabled") and state.get("Z") is not None)


def suggest_pc_count(trial: optuna.Trial, state: Dict[str, Any]) -> int:
    """Suggest a PC count using the configured fixed value / choices / range."""
    cfg = state["cfg"]
    cap = int(state["n_pcs_fit"])

    fixed = cfg.get("n_pcs_fixed", None)
    if fixed is not None:
        r = int(min(int(fixed), cap))
        trial.set_user_attr("n_pcs", r)
        return r

    choices = cfg.get("n_pcs_choices", None)
    if choices is not None:
        usable = sorted({int(c) for c in choices if 1 <= int(c) <= cap})
        if not usable:
            raise ValueError("feature_pca.n_pcs_choices contained no value in [1, n_pcs_fit]")
        r = int(trial.suggest_categorical("n_pcs", usable))
        return r

    rng = cfg.get("n_pcs_range")
    low = max(1, int(rng[0]))
    high = min(cap, int(rng[1]))
    if high < low:
        raise ValueError(
            f"feature_pca.n_pcs_range [{rng[0]}, {rng[1]}] does not overlap [1, {cap}]"
        )
    step = int(cfg.get("n_pcs_step", 1))
    if step > 1:
        r = int(trial.suggest_int("n_pcs", low, high, step=step))
    else:
        r = int(trial.suggest_int("n_pcs", low, high, log=bool(cfg.get("n_pcs_log", False))))
    return r
