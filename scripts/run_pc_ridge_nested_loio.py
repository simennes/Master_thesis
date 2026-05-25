#!/usr/bin/env python
"""E1: PC-ridge nested LOIO (Algorithm 1).

Jointly tunes the number of retained PCs (r) and the ridge shrinkage (lambda)
per outer LOIO target island. PCs are fit on the training portion of each fold
to avoid leakage, then sliced for each trial. Designed to be run trait-by-trait
since traits get different best (r, lambda).

The script is the E1 counterpart of run_ridge_loio.py and mirrors its tidy-CSV
output layout under outputs/final_results/<run_name>/<trait>/.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from src.cv_utils import ISLAND_ID_TO_NAME, island_label, make_inner_loio_splits
from src.data import load_data
from src.utils import _pearson_corr, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Config parsing helpers (kept compatible with the rest of the repo)
# ---------------------------------------------------------------------------

def _resolve_island_code(value: Any, code_to_label: Dict[int, Any], present_codes: Set[int]) -> int:
    try:
        val = int(value)
    except (ValueError, TypeError):
        val = None

    if val is not None:
        if val in present_codes:
            return val
        for code, label in code_to_label.items():
            if int(label) == val:
                return int(code)

    if isinstance(value, str):
        target_lower = value.lower()
        for orig_label, name in ISLAND_ID_TO_NAME.items():
            if name.lower() == target_lower:
                for code, lbl in code_to_label.items():
                    if int(lbl) == int(orig_label):
                        return int(code)
        for code, label in code_to_label.items():
            if str(label).lower() == target_lower:
                return int(code)

    raise ValueError(f"Could not resolve island: {value!r}")


def _build_trait_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    traits_cfg = cfg.get("traits", None)
    if traits_cfg is None:
        paths = dict(cfg["paths"])
        return [{
            "name": "default",
            "paths": paths,
            "target_column": cfg.get("target_column", "y_adjusted"),
            "eval_target_column": cfg.get("eval_target_column", "y_mean"),
            "standardize_features": cfg.get("standardize_features", False),
            "min_count": int(cfg.get("min_count", 20)),
        }]

    specs: List[Dict[str, Any]] = []
    for t in traits_cfg:
        if "name" not in t or "npz" not in t:
            raise ValueError("Each trait must define 'name' and 'npz'.")
        specs.append({
            "name": str(t["name"]),
            "paths": {"npz": str(t["npz"])},
            "target_column": t.get("target_column", cfg.get("target_column", "y_adjusted")),
            "eval_target_column": t.get("eval_target_column", cfg.get("eval_target_column", "y_mean")),
            "standardize_features": bool(t.get("standardize_features", cfg.get("standardize_features", False))),
            "min_count": int(t.get("min_count", cfg.get("min_count", 20))),
        })
    return specs


def _filter_include_islands(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    locality: np.ndarray,
    y_eval: np.ndarray,
    include_islands: Optional[List[Any]],
    code_to_label: Dict[int, Any],
):
    if not include_islands:
        return X, y, ids, locality, y_eval

    present_codes = set(int(c) for c in np.unique(locality))
    include_codes = {
        _resolve_island_code(v, code_to_label, present_codes)
        for v in include_islands
    }
    mask = np.isin(locality, list(include_codes))
    if not np.any(mask):
        raise ValueError("include_islands filtered out all samples.")
    return X[mask], y[mask], ids[mask], locality[mask], y_eval[mask]


def _resolve_pc_grid(search_cfg: Dict[str, Any]) -> List[int]:
    grid = search_cfg.get("n_pcs_choices")
    if grid is None:
        range_cfg = search_cfg.get("n_pcs_range", [500, 5000])
        step = int(search_cfg.get("n_pcs_step", 500))
        lo, hi = int(range_cfg[0]), int(range_cfg[1])
        grid = list(range(lo, hi + 1, step))
    grid = sorted({int(v) for v in grid if int(v) > 0})
    if not grid:
        raise ValueError("PC grid must contain at least one positive integer.")
    return grid


def _resolve_alpha_range(search_cfg: Dict[str, Any]) -> tuple[float, float, bool]:
    range_cfg = search_cfg.get("alpha_range", search_cfg.get("alpha_loguniform", [1e3, 1e6]))
    lo, hi = float(range_cfg[0]), float(range_cfg[1])
    if lo <= 0 or hi <= 0 or lo >= hi:
        raise ValueError(f"alpha_range must satisfy 0<lo<hi; got {(lo, hi)}.")
    log = bool(search_cfg.get("alpha_log", True))
    return lo, hi, log


# ---------------------------------------------------------------------------
# PCA: fit once on the full filtered dataset for the largest r in the grid.
# Each Optuna trial then just slices Z[:, :r] -- PCs are nested, so the
# first r columns of a 5000-component PCA equal the r-component PCA result.
# Safe to fit on full X because PCA only sees genotypes, not target phenotypes.
# ---------------------------------------------------------------------------

def _fit_dataset_pca(
    X: np.ndarray,
    max_r: int,
    seed: int,
    standardize: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit randomized PCA on the full X once and return all PC scores.

    Returns Z (n_samples, feasible_r), fit_seconds, explained_variance_ratio.
    """
    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])
    feasible_r = int(min(int(max_r), n_samples, n_features))
    if feasible_r < 1:
        raise ValueError(
            f"Cannot fit PCA: feasible_r={feasible_r} (n={n_samples}, p={n_features}, max_r={max_r})."
        )

    X_in = X.astype(np.float32, copy=False)
    if standardize:
        mean = X_in.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = X_in.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std == 0] = 1.0
        X_in = (X_in - mean) / std

    pca = PCA(n_components=feasible_r, svd_solver="randomized", random_state=int(seed))
    t0 = time.perf_counter()
    Z = pca.fit_transform(X_in)
    dt = float(time.perf_counter() - t0)
    return (
        Z.astype(np.float32, copy=False),
        dt,
        pca.explained_variance_ratio_.astype(np.float32, copy=False),
    )


def _fit_ridge_eval(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_eval_test: np.ndarray,
    r: int,
    alpha: float,
) -> float:
    r_eff = int(min(r, Z_train.shape[1]))
    model = Ridge(alpha=max(float(alpha), 1e-12))
    model.fit(Z_train[:, :r_eff], y_train)
    pred = model.predict(Z_test[:, :r_eff])
    corr = float(_pearson_corr(y_eval_test, pred))
    if not np.isfinite(corr):
        corr = 0.0
    return corr


# ---------------------------------------------------------------------------
# Per-trait runner
# ---------------------------------------------------------------------------

def _run_trait(
    trait_spec: Dict[str, Any],
    *,
    cv_cfg: Dict[str, Any],
    search_cfg: Dict[str, Any],
    output_root: Path,
    file_stem: str,
    seed: int,
    n_trials: int,
    pc_grid: List[int],
    alpha_lo: float,
    alpha_hi: float,
    alpha_log: bool,
    selected_test_islands_global: Optional[List[Any]],
    target_islands_override: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    trait_name = str(trait_spec["name"])
    logger.info("==== Trait '%s' ====", trait_name)

    X, y, ids, _, locality, code_to_label, y_eval = load_data(
        paths=trait_spec["paths"],
        target_column=trait_spec["target_column"],
        standardize_features=False,  # standardization is handled inside the PCA helper if requested
        return_locality=True,
        min_count=trait_spec["min_count"],
        return_eval=True,
        eval_target_column=trait_spec["eval_target_column"],
    )
    if y_eval is None:
        y_eval = y.copy()

    include_islands = cv_cfg.get("include_islands", None)
    X, y, ids, locality, y_eval = _filter_include_islands(
        X, y, ids, locality, y_eval, include_islands, code_to_label,
    )

    standardize = bool(trait_spec.get("standardize_features", False))
    max_r = int(max(pc_grid))

    present_codes = set(int(c) for c in np.unique(locality))
    selected_test_codes: Optional[Set[int]] = None
    selected_source = selected_test_islands_global
    if target_islands_override:
        selected_source = target_islands_override
    if selected_source:
        selected_test_codes = {
            _resolve_island_code(v, code_to_label, present_codes)
            for v in selected_source
        }

    unique_islands = sorted(present_codes)
    logger.info(
        "Trait=%s n=%d p=%d islands=%d max_r=%d n_trials=%d",
        trait_name, X.shape[0], X.shape[1], len(unique_islands), max_r, n_trials,
    )

    # Fit PCA once on the full filtered dataset (PCs only use X, not y).
    Z_full, pca_seconds, var_ratio = _fit_dataset_pca(
        X=X, max_r=max_r, seed=seed, standardize=standardize,
    )
    logger.info(
        "Trait=%s PCA fit on full dataset (n=%d, p=%d, r=%d) in %.2fs (cumulative var=%.3f)",
        trait_name, X.shape[0], X.shape[1], int(Z_full.shape[1]), pca_seconds, float(np.sum(var_ratio)),
    )

    # Output directory per trait
    trait_dir = output_root / trait_name
    trait_dir.mkdir(parents=True, exist_ok=True)

    all_fold_rows: List[Dict[str, Any]] = []
    best_params_per_fold: List[Dict[str, Any]] = []

    for outer_idx, test_isl in enumerate(unique_islands, start=1):
        if selected_test_codes is not None and test_isl not in selected_test_codes:
            continue

        idx_outer_train = np.where(locality != test_isl)[0]
        idx_outer_test = np.where(locality == test_isl)[0]
        isl_name = island_label(test_isl, code_to_label)
        logger.info(
            "OUTER %d/%d test island=%s (code=%d, n_test=%d, n_train=%d)",
            outer_idx, len(unique_islands), isl_name, int(test_isl),
            int(idx_outer_test.size), int(idx_outer_train.size),
        )

        # Inner LOIO folds: just index into the shared Z_full.
        inner_plan = make_inner_loio_splits(locality, idx_outer_train)
        inner_folds: List[Dict[str, Any]] = []
        for in_tr, in_va, in_isl in inner_plan:
            if in_tr.size < 2 or in_va.size == 0:
                continue
            inner_folds.append({
                "in_tr": in_tr,
                "in_va": in_va,
                "in_isl": int(in_isl),
            })

        if not inner_folds:
            logger.warning("OUTER %d: no usable inner folds, skipping.", outer_idx)
            continue

        # ---- Optuna study over (r, alpha) ----
        def objective(trial: optuna.Trial) -> float:
            r = int(trial.suggest_categorical("n_pcs", [int(v) for v in pc_grid]))
            alpha = float(trial.suggest_float("alpha", alpha_lo, alpha_hi, log=alpha_log))
            r_vals: List[float] = []
            for step, fold in enumerate(inner_folds, start=1):
                corr = _fit_ridge_eval(
                    Z_train=Z_full[fold["in_tr"]], y_train=y[fold["in_tr"]],
                    Z_test=Z_full[fold["in_va"]], y_eval_test=y_eval[fold["in_va"]],
                    r=r, alpha=alpha,
                )
                r_vals.append(corr)
                trial.report(float(np.mean(r_vals)), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(r_vals))

        sampler = optuna.samplers.TPESampler(seed=seed + outer_idx)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
        study = optuna.create_study(
            direction="maximize",
            study_name=f"pc_ridge_outer{outer_idx}",
            sampler=sampler,
            pruner=pruner,
        )
        t_study0 = time.perf_counter()
        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
        t_study = float(time.perf_counter() - t_study0)

        best_trial = study.best_trial
        best_r = int(best_trial.params["n_pcs"])
        best_alpha = float(best_trial.params["alpha"])
        mean_inner_r = float(best_trial.value)
        logger.info(
            "OUTER %d done in %.1fs: best n_pcs=%d alpha=%.3e inner_mean_r=%.4f (%d trials)",
            outer_idx, t_study, best_r, best_alpha, mean_inner_r, len(study.trials),
        )

        # ---- Refit on outer train (PC scores), evaluate on outer test ----
        t0 = time.perf_counter()
        outer_test_r = _fit_ridge_eval(
            Z_train=Z_full[idx_outer_train], y_train=y[idx_outer_train],
            Z_test=Z_full[idx_outer_test], y_eval_test=y_eval[idx_outer_test],
            r=best_r, alpha=best_alpha,
        )
        fit_time = float(time.perf_counter() - t0)
        cum_var_at_best_r = float(np.sum(var_ratio[:best_r]))
        logger.info(
            "OUTER %d test island=%s test_r=%.4f (best_r=%d alpha=%.3e cumvar=%.3f)",
            outer_idx, isl_name, outer_test_r, best_r, best_alpha, cum_var_at_best_r,
        )

        fold_row = {
            "trait": trait_name,
            "fold": int(outer_idx),
            "test_island_code": int(test_isl),
            "test_island": isl_name,
            "test_island_original_label": (
                str(code_to_label[int(test_isl)]) if (code_to_label and int(test_isl) in code_to_label) else None
            ),
            "n_train": int(idx_outer_train.size),
            "n_test": int(idx_outer_test.size),
            "n_features_available": int(X.shape[1]),
            "best_n_pcs": int(best_r),
            "best_alpha": float(best_alpha),
            "mean_inner_pearson_r": float(mean_inner_r),
            "pearson_r": float(outer_test_r),
            "outer_pca_cumvar_at_best_r": float(cum_var_at_best_r),
            "fit_time_seconds": float(fit_time),
            "study_time_seconds": float(t_study),
            "n_trials_completed": int(len(study.trials)),
        }
        all_fold_rows.append(fold_row)
        best_params_per_fold.append({
            "fold": int(outer_idx),
            "test_island_code": int(test_isl),
            "test_island": isl_name,
            "best_params": {
                "n_pcs": int(best_r),
                "alpha": float(best_alpha),
            },
            "mean_inner_pearson_r": float(mean_inner_r),
            "pearson_r": float(outer_test_r),
        })

    # ---- Tidy outputs per trait ----
    if all_fold_rows:
        per_fold_df = pd.DataFrame(all_fold_rows)
        per_fold_path = trait_dir / f"{file_stem}_per_fold_results.csv"
        per_fold_df.to_csv(per_fold_path, index=False)
        logger.info("Saved per-fold results: %s", per_fold_path)

        summary_row = {
            "trait": trait_name,
            "pearson_r_mean": float(per_fold_df["pearson_r"].mean()),
            "pearson_r_std": float(per_fold_df["pearson_r"].std()),
            "n_folds": int(per_fold_df.shape[0]),
            "best_alpha_geomean": float(np.exp(np.log(per_fold_df["best_alpha"]).mean())),
            "best_n_pcs_median": float(per_fold_df["best_n_pcs"].median()),
            "best_n_pcs_mode": int(per_fold_df["best_n_pcs"].mode().iloc[0]),
        }
        summary_path = trait_dir / f"{file_stem}_summary.csv"
        pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
        logger.info("Saved summary: %s", summary_path)

        best_params_path = trait_dir / f"{file_stem}_best_params.json"
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump({
                "trait": trait_name,
                "best_params_per_fold": best_params_per_fold,
                "summary": summary_row,
            }, f, indent=2)
        logger.info("Saved best-params JSON: %s", best_params_path)

    return all_fold_rows


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_pc_ridge_nested_loio(
    config: Dict[str, Any],
    target_islands_override: Optional[List[str]] = None,
) -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    cv_cfg = config.get("cv", {})
    strategy = str(cv_cfg.get("strategy", "leave_island_out")).lower()
    if strategy != "leave_island_out":
        raise ValueError("This runner currently supports only cv.strategy='leave_island_out'.")

    search_cfg = config.get("search_space", {})
    pc_grid = _resolve_pc_grid(search_cfg)
    alpha_lo, alpha_hi, alpha_log = _resolve_alpha_range(search_cfg)
    n_trials = int(config.get("n_trials", 100))

    output_cfg = config.get("output", {})
    output_root = Path(output_cfg.get("root_dir", "outputs/final_results/pc_ridge_nested_loio"))
    file_stem = str(output_cfg.get("file_stem", "pc_ridge_nested_loio"))
    output_root.mkdir(parents=True, exist_ok=True)

    # Persist the config that was used (alongside per-trait outputs)
    with open(output_root / f"{file_stem}_config_used.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    trait_specs = _build_trait_specs(config)
    selected_test_islands_global = cv_cfg.get("selected_test_islands")

    all_rows: List[Dict[str, Any]] = []
    for trait_spec in trait_specs:
        rows = _run_trait(
            trait_spec=trait_spec,
            cv_cfg=cv_cfg,
            search_cfg=search_cfg,
            output_root=output_root,
            file_stem=file_stem,
            seed=seed,
            n_trials=n_trials,
            pc_grid=pc_grid,
            alpha_lo=alpha_lo,
            alpha_hi=alpha_hi,
            alpha_log=alpha_log,
            selected_test_islands_global=selected_test_islands_global,
            target_islands_override=target_islands_override,
        )
        all_rows.extend(rows)

    # ---- Combined tidy outputs across traits ----
    if all_rows:
        combined_df = pd.DataFrame(all_rows)
        combined_path = output_root / f"{file_stem}_per_fold_results.csv"
        combined_df.to_csv(combined_path, index=False)
        logger.info("Saved combined per-fold results: %s", combined_path)

        summary_df = (
            combined_df.groupby("trait", as_index=False)
            .agg(
                pearson_r_mean=("pearson_r", "mean"),
                pearson_r_std=("pearson_r", "std"),
                n_folds=("pearson_r", "size"),
            )
        )
        summary_df["best_alpha_geomean"] = (
            combined_df.groupby("trait")["best_alpha"]
            .apply(lambda s: float(np.exp(np.log(s).mean())))
            .reindex(summary_df["trait"]).to_numpy()
        )
        summary_df["best_n_pcs_median"] = (
            combined_df.groupby("trait")["best_n_pcs"]
            .median().reindex(summary_df["trait"]).to_numpy()
        )
        summary_df["best_n_pcs_mode"] = (
            combined_df.groupby("trait")["best_n_pcs"]
            .agg(lambda s: int(s.mode().iloc[0]))
            .reindex(summary_df["trait"]).to_numpy()
        )
        summary_path = output_root / f"{file_stem}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved combined summary: %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="E1: PC-ridge nested LOIO runner")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--target_islands",
        nargs="+",
        default=None,
        help="Optional override for cv.selected_test_islands",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    run_pc_ridge_nested_loio(cfg, target_islands_override=args.target_islands)


if __name__ == "__main__":
    main()
