#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from src.cv_utils import ISLAND_ID_TO_NAME, island_label, make_outer_splits
from src.data import load_data
from src.utils import _pearson_corr, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _build_snp_experiment_specs(cfg: Dict[str, Any]) -> tuple[List[Dict[str, Any]], bool]:
    """Return SNP experiment specs and whether they came from the new multi-spec config."""
    explicit_specs = "snp_experiments" in cfg
    raw_specs = cfg.get("snp_experiments", None)
    if raw_specs is None:
        use_snp_selection = bool(cfg.get("use_snp_selection", False))
        num_snps = cfg.get("num_snps", None)
        mode = str(cfg.get("snp_selection_mode", "random")).lower()
        use_pcs = bool(cfg.get("use_pcs", False))
        n_pcs = cfg.get("n_pcs", None)
        if use_pcs:
            name = f"pcs_{int(n_pcs)}" if n_pcs is not None else "pcs"
        else:
            name = "random_snps" if use_snp_selection else "all_snps"
        raw_specs = [{
            "name": name,
            "use_snp_selection": use_snp_selection,
            "snp_selection_mode": mode,
            "num_snps": num_snps,
            "use_pcs": use_pcs,
            "n_pcs": n_pcs,
            "n_repeats": int(cfg.get("n_snp_repeats", 1)),
            "seed": cfg.get("seed", 42),
            "seed_stride": int(cfg.get("seed_stride", 1000)),
        }]

    if not isinstance(raw_specs, list) or len(raw_specs) == 0:
        raise ValueError("snp_experiments must be a non-empty list")

    specs: List[Dict[str, Any]] = []
    for i, spec in enumerate(raw_specs):
        if not isinstance(spec, dict):
            raise ValueError("Each snp_experiments entry must be an object")
        use_snp_selection = bool(spec.get("use_snp_selection", False))
        mode = str(spec.get("snp_selection_mode", cfg.get("snp_selection_mode", "random"))).lower()
        num_snps = spec.get("num_snps", None)
        use_pcs = bool(spec.get("use_pcs", False))
        n_pcs = spec.get("n_pcs", None)
        n_repeats = int(spec.get("n_repeats", 1))
        if n_repeats < 1:
            raise ValueError("snp_experiments.n_repeats must be >= 1")
        if use_pcs and use_snp_selection:
            raise ValueError("snp_experiments entry cannot set both use_pcs and use_snp_selection")
        if use_pcs:
            if n_pcs is None:
                raise ValueError("snp_experiments entry has use_pcs=True but n_pcs is missing")
            n_pcs = int(n_pcs)
            if n_pcs < 1:
                raise ValueError("n_pcs must be >= 1")
        if mode != "random":
            raise ValueError("run_ridge_loio currently supports only snp_selection_mode='random'")
        if use_snp_selection and num_snps is None:
            raise ValueError("SNP selection is enabled but num_snps is missing")
        if num_snps is not None:
            num_snps = int(num_snps)
            if num_snps < 1:
                raise ValueError("num_snps must be >= 1")

        if use_pcs:
            default_name = f"pcs_{n_pcs}"
        elif use_snp_selection:
            default_name = f"random_{num_snps}"
        else:
            default_name = "all_snps"
        specs.append({
            "name": str(spec.get("name", default_name if default_name else f"snp_set_{i}")),
            "use_snp_selection": use_snp_selection,
            "snp_selection_mode": mode,
            "num_snps": num_snps,
            "use_pcs": use_pcs,
            "n_pcs": n_pcs,
            "n_repeats": n_repeats,
            "seed": int(spec.get("seed", cfg.get("seed", 42))),
            "seed_stride": int(spec.get("seed_stride", cfg.get("seed_stride", 1000))),
        })
    return specs, explicit_specs


def _select_snp_columns(
    n_features: int,
    spec: Dict[str, Any],
    repeat: int,
) -> tuple[Optional[np.ndarray], Optional[int], int]:
    if not bool(spec["use_snp_selection"]):
        return None, None, int(n_features)

    num_snps = int(spec["num_snps"])
    if num_snps >= n_features:
        return None, None, int(n_features)

    snp_seed = int(spec["seed"]) + int(repeat) * int(spec["seed_stride"])
    rng = np.random.default_rng(snp_seed)
    snp_cols = rng.choice(n_features, size=num_snps, replace=False)
    snp_cols.sort()
    return snp_cols.astype(int, copy=False), snp_seed, int(num_snps)


def _compute_dataset_pcs(
    X: np.ndarray,
    n_pcs: int,
    seed: int,
    standardize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PC scores for the whole dataset using all SNP features.

    Returns the PC score matrix (n_samples, n_pcs_used) and the
    explained variance ratio. PCs are fit once on the full provided X
    (i.e. all SNPs across all retained islands) so that the same
    representation is reused across LOIO folds. Each fold then trains
    ridge on the train rows of these PC scores.
    """
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    max_pcs = int(min(int(n_pcs), n_samples, n_features))
    if max_pcs < 1:
        raise ValueError("n_pcs must yield at least one component")

    pca = PCA(
        n_components=max_pcs,
        svd_solver="randomized",
        random_state=int(seed),
    )
    if standardize:
        std = X.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std == 0] = 1.0
        X_in = (X - X.mean(axis=0, keepdims=True)) / std
        scores = pca.fit_transform(X_in)
    else:
        scores = pca.fit_transform(X)
    return scores.astype(np.float32, copy=False), pca.explained_variance_ratio_.astype(np.float32, copy=False)


def _original_island_label(test_island_code: int, code_to_label: Dict[int, Any]) -> Optional[str]:
    if code_to_label is None:
        return None
    if int(test_island_code) not in code_to_label:
        return None
    return str(code_to_label[int(test_island_code)])


def _write_tidy_outputs(
    output_root: Path,
    file_stem: str,
    config: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / f"{file_stem}_config_used.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if not rows:
        return

    results_df = pd.DataFrame(rows)
    results_path = output_root / f"{file_stem}_per_fold_results.csv"
    results_df.to_csv(results_path, index=False)

    repeat_summary = (
        results_df.groupby(
            ["trait", "snp_experiment", "snp_repeat"],
            dropna=False,
            as_index=False,
        )
        .agg(
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_std=("pearson_r", "std"),
            n_folds=("pearson_r", "size"),
            n_features_fit=("n_features_fit", "first"),
            fit_time_total_s=("fit_time_seconds", "sum"),
        )
        .sort_values(["trait", "snp_experiment", "snp_repeat"])
    )
    repeat_summary_path = output_root / f"{file_stem}_summary_by_repeat.csv"
    repeat_summary.to_csv(repeat_summary_path, index=False)

    summary = (
        repeat_summary.groupby(["trait", "snp_experiment"], as_index=False)
        .agg(
            pearson_r_mean=("pearson_r_mean", "mean"),
            pearson_r_std_across_repeats=("pearson_r_mean", "std"),
            n_repeats=("snp_repeat", "size"),
            n_folds_total=("n_folds", "sum"),
            n_features_fit=("n_features_fit", "first"),
            fit_time_total_s=("fit_time_total_s", "sum"),
        )
        .sort_values(["trait", "snp_experiment"])
    )
    summary_path = output_root / f"{file_stem}_summary.csv"
    summary.to_csv(summary_path, index=False)

    logger.info("Saved tidy ridge LOIO results: %s", results_path)
    logger.info("Saved repeat summary: %s", repeat_summary_path)
    logger.info("Saved summary: %s", summary_path)


def run_ridge_loio(config: Dict[str, Any], target_islands_override: Optional[List[str]] = None) -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    cv_cfg = config.get("cv", {})
    strategy = str(cv_cfg.get("strategy", "leave_island_out")).lower()
    if strategy != "leave_island_out":
        raise ValueError("This runner currently supports only cv.strategy='leave_island_out'.")

    include_islands = cv_cfg.get("include_islands", None)
    selected_test_islands = cv_cfg.get("selected_test_islands", None)

    model_cfg = config.get("ridge", {})
    alpha = float(model_cfg.get("alpha", 1e5))

    output_cfg = config.get("output", {})
    output_root = Path(output_cfg.get("root_dir", "outputs/nested_cv"))
    file_stem = str(output_cfg.get("file_stem", "ridge"))

    trait_specs = _build_trait_specs(config)
    snp_specs, has_explicit_snp_specs = _build_snp_experiment_specs(config)
    all_rows: List[Dict[str, Any]] = []

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        logger.info("Running trait '%s'", trait_name)

        X, y, ids, _, locality, code_to_label, y_eval = load_data(
            paths=trait_spec["paths"],
            target_column=trait_spec["target_column"],
            standardize_features=trait_spec["standardize_features"],
            return_locality=True,
            min_count=trait_spec["min_count"],
            return_eval=True,
            eval_target_column=trait_spec["eval_target_column"],
        )
        if y_eval is None:
            y_eval = y.copy()

        X, y, ids, locality, y_eval = _filter_include_islands(
            X, y, ids, locality, y_eval, include_islands, code_to_label,
        )

        present_codes = set(int(c) for c in np.unique(locality))
        selected_test_codes: Optional[Set[int]] = None

        if target_islands_override:
            selected_test_codes = {
                _resolve_island_code(v, code_to_label, present_codes)
                for v in target_islands_override
            }
        elif selected_test_islands:
            selected_test_codes = {
                _resolve_island_code(v, code_to_label, present_codes)
                for v in selected_test_islands
            }

        pc_cache: Dict[int, np.ndarray] = {}
        pc_var_cache: Dict[int, np.ndarray] = {}

        def _get_pcs(n_pcs_req: int) -> np.ndarray:
            key = int(n_pcs_req)
            if key not in pc_cache:
                t0_pca = time.perf_counter()
                scores, var_ratio = _compute_dataset_pcs(
                    X,
                    n_pcs=key,
                    seed=seed,
                    standardize=bool(trait_spec["standardize_features"]),
                )
                logger.info(
                    "Trait=%s computed %d PCs on full dataset (n=%d, p=%d) in %.2fs (var explained=%.3f)",
                    trait_name,
                    int(scores.shape[1]),
                    int(X.shape[0]),
                    int(X.shape[1]),
                    float(time.perf_counter() - t0_pca),
                    float(np.sum(var_ratio)),
                )
                pc_cache[key] = scores
                pc_var_cache[key] = var_ratio
            return pc_cache[key]

        for snp_spec in snp_specs:
            exp_name = str(snp_spec["name"])
            use_pcs_flag = bool(snp_spec.get("use_pcs", False))
            for repeat in range(int(snp_spec["n_repeats"])):
                if use_pcs_flag:
                    n_pcs_req = int(snp_spec["n_pcs"])
                    X_fit = _get_pcs(n_pcs_req)
                    snp_cols = None
                    snp_seed = None
                    n_features_fit = int(X_fit.shape[1])
                else:
                    snp_cols, snp_seed, n_features_fit = _select_snp_columns(
                        n_features=int(X.shape[1]),
                        spec=snp_spec,
                        repeat=repeat,
                    )
                    X_fit = X[:, snp_cols] if snp_cols is not None else X

                logger.info(
                    "Trait=%s experiment=%s repeat=%d/%d n_features=%d seed=%s use_pcs=%s",
                    trait_name,
                    exp_name,
                    repeat + 1,
                    int(snp_spec["n_repeats"]),
                    int(n_features_fit),
                    "none" if snp_seed is None else str(snp_seed),
                    use_pcs_flag,
                )

                outer_results: List[float] = []
                per_fold: List[Dict[str, Any]] = []
                fold_i = 0

                for tr_idx, te_idx, test_island_code in make_outer_splits(
                    strategy="leave_island_out",
                    locality=locality,
                    n_splits=len(np.unique(locality)),
                    shuffle=False,
                    random_state=seed,
                    n=len(X_fit),
                ):
                    if test_island_code is None:
                        continue
                    test_island_code = int(test_island_code)
                    if selected_test_codes is not None and test_island_code not in selected_test_codes:
                        continue

                    fold_i += 1
                    model = Ridge(alpha=max(alpha, 1e-12))
                    t0 = time.perf_counter()
                    model.fit(X_fit[tr_idx], y[tr_idx])
                    pred = model.predict(X_fit[te_idx])
                    fit_time_s = float(time.perf_counter() - t0)
                    r = float(_pearson_corr(y_eval[te_idx], pred))
                    if not np.isfinite(r):
                        r = 0.0

                    fold_row = {
                        "fold": int(fold_i),
                        "pearson_r": float(r),
                        "test_island": island_label(test_island_code, code_to_label),
                        "test_island_code": int(test_island_code),
                        "test_island_original_label": _original_island_label(
                            test_island_code, code_to_label
                        ),
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                    }
                    outer_results.append(r)
                    per_fold.append(fold_row)
                    all_rows.append({
                        "trait": trait_name,
                        "snp_experiment": exp_name,
                        "snp_repeat": int(repeat),
                        "snp_seed": snp_seed,
                        "snp_selection_mode": str(snp_spec["snp_selection_mode"]),
                        "num_snps_requested": snp_spec["num_snps"],
                        "use_pcs": bool(use_pcs_flag),
                        "n_pcs_requested": (int(snp_spec["n_pcs"]) if use_pcs_flag else None),
                        "n_features_available": int(X.shape[1]),
                        "n_features_fit": int(n_features_fit),
                        "alpha": float(alpha),
                        "target_column": trait_spec["target_column"],
                        "eval_target_column": trait_spec["eval_target_column"],
                        "standardize_features": bool(trait_spec["standardize_features"]),
                        "min_count": int(trait_spec["min_count"]),
                        "fit_time_seconds": fit_time_s,
                        **fold_row,
                    })
                    logger.info(
                        "Trait=%s experiment=%s repeat=%d fold=%d island=%s r=%.4f",
                        trait_name,
                        exp_name,
                        repeat,
                        fold_i,
                        island_label(test_island_code, code_to_label),
                        r,
                    )

                out_dir = output_root / trait_name / "loio"
                if has_explicit_snp_specs:
                    out_dir = out_dir / exp_name / f"repeat_{repeat:02d}"
                out_dir.mkdir(parents=True, exist_ok=True)

                summary = {
                    "mode": "ridge",
                    "cv_strategy": "leave_island_out",
                    "per_fold": per_fold,
                    "overall": {
                        "pearson_r": float(np.mean(outer_results)) if outer_results else None,
                    },
                    "outer_test_corr": outer_results,
                    "outer_test_corr_mean": float(np.mean(outer_results)) if outer_results else None,
                    "outer_test_corr_std": float(np.std(outer_results)) if outer_results else None,
                    "outer_splits": int(len(outer_results)),
                    "inner_splits": None,
                    "best_params_per_fold": [
                        {
                            "fold": int(row["fold"]),
                            "best_params": {
                                "model_type": "ridge",
                                "alpha": float(alpha),
                            },
                            "mean_inner_r": None,
                        }
                        for row in per_fold
                    ],
                    "config_used": {
                        "alpha": float(alpha),
                        "include_islands": include_islands,
                        "selected_test_islands": list(selected_test_codes) if selected_test_codes is not None else None,
                        "target_column": trait_spec["target_column"],
                        "eval_target_column": trait_spec["eval_target_column"],
                        "snp_experiment": exp_name,
                        "snp_repeat": int(repeat),
                        "snp_seed": snp_seed,
                        "num_snps_requested": snp_spec["num_snps"],
                        "use_pcs": bool(use_pcs_flag),
                        "n_pcs_requested": (int(snp_spec["n_pcs"]) if use_pcs_flag else None),
                        "n_features_available": int(X.shape[1]),
                        "n_features_fit": int(n_features_fit),
                    },
                }

                out_path = out_dir / f"{file_stem}_results.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)

                logger.info(
                    "Saved ridge LOIO results for trait '%s' experiment '%s' repeat %d to %s (mean r=%.4f)",
                    trait_name,
                    exp_name,
                    repeat,
                    out_path,
                    summary["outer_test_corr_mean"] if summary["outer_test_corr_mean"] is not None else float("nan"),
                )

    _write_tidy_outputs(output_root, file_stem, config, all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ridge leave-one-island-out runner")
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

    run_ridge_loio(cfg, target_islands_override=args.target_islands)


if __name__ == "__main__":
    main()
