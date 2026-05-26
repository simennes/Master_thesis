#!/usr/bin/env python
"""PC-ridge LOIO with fixed (r, lambda) per trait.

Companion to run_pc_ridge_nested_loio.py: instead of tuning the number of PCs
and the ridge shrinkage per outer fold, this runner uses one fixed (n_pcs,
alpha) per trait (typically the median of E1's per-fold selections) and runs
plain outer leave-one-island-out evaluation. Useful for asking "how much do
we lose if we commit to one global setting?".
"""
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

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
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
    traits_cfg = cfg.get("traits")
    if not traits_cfg:
        raise ValueError("Config must define a 'traits' list with fixed (n_pcs, alpha) per trait.")

    specs: List[Dict[str, Any]] = []
    for t in traits_cfg:
        for key in ("name", "npz", "n_pcs", "alpha"):
            if key not in t:
                raise ValueError(f"Trait entry missing required field '{key}': {t}")
        specs.append({
            "name": str(t["name"]),
            "paths": {"npz": str(t["npz"])},
            "target_column": t.get("target_column", cfg.get("target_column", "y_adjusted")),
            "eval_target_column": t.get("eval_target_column", cfg.get("eval_target_column", "y_mean")),
            "standardize_features": bool(t.get("standardize_features", cfg.get("standardize_features", False))),
            "min_count": int(t.get("min_count", cfg.get("min_count", 20))),
            "n_pcs": int(t["n_pcs"]),
            "alpha": float(t["alpha"]),
        })
    return specs


def _filter_include_islands(X, y, ids, locality, y_eval, include_islands, code_to_label):
    if not include_islands:
        return X, y, ids, locality, y_eval
    present_codes = set(int(c) for c in np.unique(locality))
    include_codes = {
        _resolve_island_code(v, code_to_label, present_codes) for v in include_islands
    }
    mask = np.isin(locality, list(include_codes))
    if not np.any(mask):
        raise ValueError("include_islands filtered out all samples.")
    return X[mask], y[mask], ids[mask], locality[mask], y_eval[mask]


def _fit_dataset_pca(X: np.ndarray, max_r: int, seed: int, standardize: bool):
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
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
    return Z.astype(np.float32, copy=False), dt, pca.explained_variance_ratio_.astype(np.float32, copy=False)


def _run_trait(
    trait_spec: Dict[str, Any],
    *,
    cv_cfg: Dict[str, Any],
    output_root: Path,
    file_stem: str,
    seed: int,
    selected_test_islands_global: Optional[List[Any]],
    target_islands_override: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    trait_name = trait_spec["name"]
    n_pcs = int(trait_spec["n_pcs"])
    alpha = float(trait_spec["alpha"])
    logger.info("==== Trait '%s' (fixed n_pcs=%d, alpha=%.4e) ====", trait_name, n_pcs, alpha)

    X, y, ids, _, locality, code_to_label, y_eval = load_data(
        paths=trait_spec["paths"],
        target_column=trait_spec["target_column"],
        standardize_features=False,
        return_locality=True,
        min_count=trait_spec["min_count"],
        return_eval=True,
        eval_target_column=trait_spec["eval_target_column"],
    )
    if y_eval is None:
        y_eval = y.copy()

    include_islands = cv_cfg.get("include_islands")
    X, y, ids, locality, y_eval = _filter_include_islands(
        X, y, ids, locality, y_eval, include_islands, code_to_label,
    )

    standardize = bool(trait_spec.get("standardize_features", False))

    present_codes = set(int(c) for c in np.unique(locality))
    selected_test_codes: Optional[Set[int]] = None
    selected_source = target_islands_override or selected_test_islands_global
    if selected_source:
        selected_test_codes = {
            _resolve_island_code(v, code_to_label, present_codes) for v in selected_source
        }

    Z_full, pca_seconds, var_ratio = _fit_dataset_pca(
        X=X, max_r=n_pcs, seed=seed, standardize=standardize,
    )
    cum_var = float(np.sum(var_ratio[:n_pcs]))
    logger.info(
        "Trait=%s PCA fit (n=%d, p=%d, r=%d) in %.2fs (cumvar=%.3f)",
        trait_name, X.shape[0], X.shape[1], int(Z_full.shape[1]), pca_seconds, cum_var,
    )

    trait_dir = output_root / trait_name
    trait_dir.mkdir(parents=True, exist_ok=True)

    unique_islands = sorted(present_codes)
    fold_rows: List[Dict[str, Any]] = []
    for outer_idx, test_isl in enumerate(unique_islands, start=1):
        if selected_test_codes is not None and test_isl not in selected_test_codes:
            continue

        idx_train = np.where(locality != test_isl)[0]
        idx_test = np.where(locality == test_isl)[0]
        isl_name = island_label(test_isl, code_to_label)

        r_eff = int(min(n_pcs, Z_full.shape[1]))
        model = Ridge(alpha=max(alpha, 1e-12))
        t0 = time.perf_counter()
        model.fit(Z_full[idx_train, :r_eff], y[idx_train])
        pred = model.predict(Z_full[idx_test, :r_eff])
        corr = float(_pearson_corr(y_eval[idx_test], pred))
        if not np.isfinite(corr):
            corr = 0.0
        fit_time = float(time.perf_counter() - t0)

        logger.info(
            "OUTER %d/%d test=%s (code=%d, n_test=%d) r=%.4f (%.2fs)",
            outer_idx, len(unique_islands), isl_name, int(test_isl), int(idx_test.size), corr, fit_time,
        )

        fold_rows.append({
            "trait": trait_name,
            "fold": int(outer_idx),
            "test_island_code": int(test_isl),
            "test_island": isl_name,
            "test_island_original_label": (
                str(code_to_label[int(test_isl)]) if (code_to_label and int(test_isl) in code_to_label) else None
            ),
            "n_train": int(idx_train.size),
            "n_test": int(idx_test.size),
            "n_features_available": int(X.shape[1]),
            "n_pcs": int(r_eff),
            "alpha": float(alpha),
            "pearson_r": float(corr),
            "outer_pca_cumvar": float(cum_var),
            "fit_time_seconds": float(fit_time),
        })

    if fold_rows:
        df = pd.DataFrame(fold_rows)
        df.to_csv(trait_dir / f"{file_stem}_per_fold_results.csv", index=False)
        summary = {
            "trait": trait_name,
            "n_pcs": n_pcs,
            "alpha": alpha,
            "pearson_r_mean": float(df["pearson_r"].mean()),
            "pearson_r_std": float(df["pearson_r"].std()),
            "n_folds": int(df.shape[0]),
        }
        pd.DataFrame([summary]).to_csv(trait_dir / f"{file_stem}_summary.csv", index=False)
        logger.info("Trait %s summary: mean r=%.4f (sd=%.4f, n=%d)",
                    trait_name, summary["pearson_r_mean"], summary["pearson_r_std"], summary["n_folds"])

    return fold_rows


def run_pc_ridge_fixed_loio(config: Dict[str, Any], target_islands_override: Optional[List[str]] = None) -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    cv_cfg = config.get("cv", {})
    strategy = str(cv_cfg.get("strategy", "leave_island_out")).lower()
    if strategy != "leave_island_out":
        raise ValueError("Only cv.strategy='leave_island_out' is supported.")

    output_cfg = config.get("output", {})
    output_root = Path(output_cfg.get("root_dir", "outputs/final_results/pc_ridge_fixed_loio"))
    file_stem = str(output_cfg.get("file_stem", "pc_ridge_fixed_loio"))
    output_root.mkdir(parents=True, exist_ok=True)

    with open(output_root / f"{file_stem}_config_used.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    trait_specs = _build_trait_specs(config)
    selected_test_islands_global = cv_cfg.get("selected_test_islands")

    all_rows: List[Dict[str, Any]] = []
    for spec in trait_specs:
        all_rows.extend(_run_trait(
            trait_spec=spec,
            cv_cfg=cv_cfg,
            output_root=output_root,
            file_stem=file_stem,
            seed=seed,
            selected_test_islands_global=selected_test_islands_global,
            target_islands_override=target_islands_override,
        ))

    if all_rows:
        combined = pd.DataFrame(all_rows)
        combined.to_csv(output_root / f"{file_stem}_per_fold_results.csv", index=False)
        summary = (
            combined.groupby("trait", as_index=False)
            .agg(
                n_pcs=("n_pcs", "first"),
                alpha=("alpha", "first"),
                pearson_r_mean=("pearson_r", "mean"),
                pearson_r_std=("pearson_r", "std"),
                n_folds=("pearson_r", "size"),
            )
        )
        summary.to_csv(output_root / f"{file_stem}_summary.csv", index=False)
        logger.info("Saved combined per-fold results and summary under %s", output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="PC-ridge LOIO with fixed (n_pcs, alpha) per trait")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--target_islands", nargs="+", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    run_pc_ridge_fixed_loio(cfg, target_islands_override=args.target_islands)


if __name__ == "__main__":
    main()
