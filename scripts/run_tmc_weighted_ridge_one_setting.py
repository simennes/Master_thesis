#!/usr/bin/env python
"""
Run one TMC-Shapley setting (single target, single n_permutations, single cal_fraction)
and evaluate multiple Ridge sample-weighting methods derived from island-level Shapley values.

This script is intentionally focused and reuses existing project functions where possible.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.removal_curve import split_target_island
from src.tmc_shapley_islands import ShapleyConfig, run_shapley_experiment
from src.utils import _pearson_corr, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_island_code(target_island, code_to_label: dict, present_codes: set) -> int:
    try:
        val = int(target_island)
    except (ValueError, TypeError):
        val = None

    if val is not None:
        if val in present_codes:
            return val
        for code, label in code_to_label.items():
            if int(label) == val:
                return code
        raise ValueError(f"Island {val} not found as code or label. Present codes: {sorted(present_codes)}")

    if isinstance(target_island, str):
        target_lower = target_island.lower()
        for orig_label, name in ISLAND_ID_TO_NAME.items():
            if name.lower() == target_lower:
                for code, lbl in code_to_label.items():
                    if int(lbl) == orig_label:
                        return code
        for code, label in code_to_label.items():
            if str(label).lower() == target_lower:
                return code

    raise ValueError(f"Could not resolve target island: {target_island!r}")


def _parse_methods(methods_raw: List[Any]) -> List[Dict[str, Any]]:
    methods: List[Dict[str, Any]] = []
    for item in methods_raw:
        if isinstance(item, str):
            methods.append({"name": item})
        elif isinstance(item, dict) and "name" in item:
            methods.append(item)
        else:
            raise ValueError("weighting.methods entries must be either strings or objects with a 'name' field")
    if len(methods) == 0:
        raise ValueError("weighting.methods must contain at least one method")
    return methods


def _normalize_weights(w: np.ndarray, floor: float, clip_max: float | None) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, floor)
    mean_w = float(np.mean(w))
    if mean_w <= 0:
        w = np.ones_like(w, dtype=float)
    else:
        w = w / mean_w
    if clip_max is not None:
        w = np.minimum(w, float(clip_max))
    return w


def _method_weights(phi_per_n: np.ndarray, method_cfg: Dict[str, Any]) -> np.ndarray:
    name = str(method_cfg.get("name", "relu")).lower()
    floor = float(method_cfg.get("floor", 0.05))
    clip_max = method_cfg.get("clip_max", None)

    x = np.asarray(phi_per_n, dtype=float)

    if name == "relu":
        w = np.maximum(0.0, x)
    elif name == "shifted_relu":
        q = float(method_cfg.get("quantile", 0.25))
        thr = float(np.quantile(x, q))
        w = np.maximum(0.0, x - thr)
    elif name == "minmax":
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            w = np.ones_like(x)
        else:
            w = (x - mn) / (mx - mn)
    elif name == "softmax":
        temp = float(method_cfg.get("temperature", 1.0))
        temp = max(temp, 1e-6)
        z = (x - np.max(x)) / temp
        ex = np.exp(z)
        w = ex / np.sum(ex)
    elif name == "rank":
        order = np.argsort(-x)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(x) + 1)
        mode = str(method_cfg.get("mode", "inverse")).lower()
        if mode == "linear":
            w = (len(x) - ranks + 1).astype(float)
        else:
            w = 1.0 / ranks.astype(float)
    else:
        raise ValueError(f"Unknown weighting method: {name}")

    return _normalize_weights(w, floor=floor, clip_max=clip_max)


def _fit_eval_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None,
) -> Tuple[float, float]:
    model = Ridge(alpha=max(float(alpha), 1e-12))
    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    pred = model.predict(X_test)
    corr_eval = float(_pearson_corr(pred, y_eval_test))
    if not np.isfinite(corr_eval):
        corr_eval = 0.0
    mse_adj = float(np.mean((pred - y_test) ** 2))
    return corr_eval, mse_adj


def _make_repeat_seed(global_seed: int, target_code: int, cal_fraction: float, repeat_idx: int) -> int:
    token = f"{int(global_seed)}|{int(target_code)}|{float(cal_fraction):.12g}|{int(repeat_idx)}"
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % 2_147_483_647


def main() -> None:
    parser = argparse.ArgumentParser(description="One-setting weighted Ridge from island-level TMC-Shapley")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--target_island", default=None, help="Override target island")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    output_dir = Path(cfg.get("output_dir", "outputs/tmc_weighted_ridge_one_setting"))
    output_dir.mkdir(parents=True, exist_ok=True)

    global_seed = int(cfg.get("seed", 42))
    set_seed(global_seed)

    model_cfg = cfg.get("model", {})
    if str(model_cfg.get("type", "ridge")).lower() != "ridge":
        raise ValueError("This script is ridge-only. Set model.type = 'ridge'.")
    ridge_alpha = float(model_cfg.get("alpha", 1e5))

    tmc_raw = cfg.get("tmc", {})
    n_permutations = int(tmc_raw.get("n_permutations", 50))

    cal_fraction = float(cfg.get("cal_fraction", 0.2))
    max_cal_fraction = cfg.get("max_cal_fraction", None)

    weighting_cfg = cfg.get("weighting", {})
    methods = _parse_methods(weighting_cfg.get("methods", ["relu", "softmax", "rank", "minmax"]))

    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    paths = cfg["paths"]
    X, y, ids, _, locality, code_to_label, y_eval = load_data(
        paths=paths,
        target_column=cfg.get("target_column", "y_adjusted"),
        standardize_features=cfg.get("standardize_features", True),
        return_locality=True,
        min_count=cfg.get("min_count", 20),
        return_eval=True,
        eval_target_column=cfg.get("eval_target_column", "y_mean"),
    )

    present_codes = set(int(c) for c in np.unique(locality))
    raw_target = args.target_island if args.target_island is not None else cfg.get("target_island", cfg.get("target_islands", [0])[0])
    target_code = resolve_island_code(raw_target, code_to_label, present_codes)
    target_name = island_label(target_code, code_to_label)

    included_raw = cfg.get("included_islands", None)
    if included_raw is not None:
        included_island_codes = [resolve_island_code(v, code_to_label, present_codes) for v in included_raw]
    else:
        included_island_codes = sorted(present_codes)
    source_codes = [c for c in included_island_codes if c != target_code]
    source_names = {c: island_label(c, code_to_label) for c in source_codes}

    # fixed SNP subset used consistently in all steps
    fixed_snp_cols: np.ndarray | None = None
    use_snp_selection = bool(cfg.get("use_snp_selection", False))
    num_snps = cfg.get("num_snps", None)
    snp_mode = str(cfg.get("snp_selection_mode", "random")).lower()
    if use_snp_selection and num_snps is not None and int(num_snps) < int(X.shape[1]):
        if snp_mode != "random":
            raise ValueError("This script currently supports snp_selection_mode='random' for fixed global SNP subset.")
        rng_snp = np.random.default_rng(global_seed)
        fixed_snp_cols = np.sort(rng_snp.choice(X.shape[1], size=int(num_snps), replace=False).astype(np.int64))
        np.save(output_dir / "fixed_snp_columns.npy", fixed_snp_cols)

    shapley_cfg = ShapleyConfig(
        n_permutations=n_permutations,
        eps_trunc=float(tmc_raw.get("eps_trunc", 0.005)),
        min_prefix_islands=int(tmc_raw.get("min_prefix_islands", 3)),
        use_truncation=bool(tmc_raw.get("use_truncation", False)),
        island_sampling_mode=tmc_raw.get("island_sampling_mode", "mean"),
        cap_per_island=tmc_raw.get("cap_per_island", None),
        n_train_seeds_per_eval=int(tmc_raw.get("n_train_seeds_per_eval", 1)),
        cache_dir=tmc_raw.get("cache_dir", str(output_dir / "cache")),
        n_random_orders=int(cfg.get("baselines", {}).get("n_random_orders", 1)),
        use_early_stopping=bool(tmc_raw.get("use_early_stopping", False)),
        early_stopping_patience=int(tmc_raw.get("early_stopping_patience", 0)),
        retrain_use_early_stopping=bool(tmc_raw.get("retrain_use_early_stopping", True)),
        retrain_early_stopping_patience=int(tmc_raw.get("retrain_early_stopping_patience", 20)),
        snp_selection_mode=cfg.get("snp_selection_mode", "random"),
        use_snp_selection=use_snp_selection,
        num_snps=num_snps,
        model_type="ridge",
        ridge_alpha=ridge_alpha,
        seed=global_seed,
    )

    uncertainty_cfg = cfg.get("uncertainty", {})
    n_repeats = int(cfg.get("n_repeats", uncertainty_cfg.get("n_target_split_repeats", 1)))
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    logger.info(
        "Running weighted-ridge uncertainty: target=%s(%s), perms=%d, cal_fraction=%.3f, repeats=%d",
        target_code,
        target_name,
        n_permutations,
        cal_fraction,
        n_repeats,
    )

    all_result_rows: List[Dict[str, Any]] = []
    all_weight_rows: List[Dict[str, Any]] = []
    v_full_rows: List[Dict[str, Any]] = []

    for repeat_idx in range(n_repeats):
        repeat_seed = _make_repeat_seed(global_seed, target_code, cal_fraction, repeat_idx)
        repeat_cfg = copy.deepcopy(shapley_cfg)
        repeat_cfg.seed = int(repeat_seed)

        logger.info(
            "Repeat %d/%d | split_seed=%d",
            repeat_idx + 1,
            n_repeats,
            repeat_seed,
        )

        shapley_result = run_shapley_experiment(
            X=X,
            y=y,
            ids=ids,
            locality=locality,
            y_eval=y_eval,
            target_island_code=target_code,
            target_island_name=target_name,
            source_codes=source_codes,
            source_names=source_names,
            train_params=None,
            cfg=repeat_cfg,
            output_dir=str(output_dir / f"island_{target_code}" / f"repeat_{repeat_idx}"),
            device=device,
            batch_size=int(cfg.get("batch_size", 64)),
            cal_fraction=cal_fraction,
            max_cal_fraction=max_cal_fraction,
            use_snp_selection=use_snp_selection,
            num_snps=num_snps,
            snp_selection_mode=cfg.get("snp_selection_mode", "random"),
            compute_random_individual=False,
            snp_cols_override=fixed_snp_cols,
        )

        phi = np.asarray(shapley_result["phi"], dtype=float)
        n_individuals = np.asarray(shapley_result["n_individuals"], dtype=float)
        phi_per_n = phi / np.maximum(n_individuals, 1.0)
        v_full = float(shapley_result["v_full"])

        split = split_target_island(
            X,
            y,
            ids,
            locality,
            target_island_code=target_code,
            y_eval=y_eval,
            cal_fraction=cal_fraction,
            seed=repeat_seed,
            max_cal_fraction=max_cal_fraction,
        )

        X_source = split["X_source"]
        y_source = split["y_source"]
        locality_source = split["locality_source"]
        X_test = split["X_test"]
        y_test = split["y_test"]
        y_eval_test = split["y_eval_test"]

        in_source = np.isin(locality_source, source_codes)
        X_source = X_source[in_source]
        y_source = y_source[in_source]
        locality_source = locality_source[in_source]

        if fixed_snp_cols is not None:
            X_source = X_source[:, fixed_snp_cols]
            X_test = X_test[:, fixed_snp_cols]

        # baseline: unweighted ridge on all source
        base_corr, base_mse = _fit_eval_ridge(
            X_train=X_source,
            y_train=y_source,
            X_test=X_test,
            y_test=y_test,
            y_eval_test=y_eval_test,
            alpha=ridge_alpha,
            sample_weight=None,
        )

        all_result_rows.append(
            {
                "repeat": int(repeat_idx),
                "split_seed": int(repeat_seed),
                "method": "unweighted",
                "corr_eval": base_corr,
                "mse_adj": base_mse,
                "gain_over_v_full": base_corr - v_full,
                "alpha": ridge_alpha,
                "v_full": v_full,
            }
        )

        for method_cfg in methods:
            method_name = str(method_cfg.get("name", "unknown")).lower()
            island_w = _method_weights(phi_per_n, method_cfg)

            for code, n_i, phi_i, phi_n_i, w_i in zip(source_codes, n_individuals, phi, phi_per_n, island_w):
                all_weight_rows.append(
                    {
                        "repeat": int(repeat_idx),
                        "split_seed": int(repeat_seed),
                        "source_island": int(code),
                        "source_island_name": str(source_names[code]),
                        "n_island": int(n_i),
                        "phi": float(phi_i),
                        "phi_per_individual": float(phi_n_i),
                        "method": method_name,
                        "weight": float(w_i),
                    }
                )

            sample_weight = np.ones(len(X_source), dtype=float)
            for code, w in zip(source_codes, island_w):
                sample_weight[locality_source == code] = float(w)

            corr, mse = _fit_eval_ridge(
                X_train=X_source,
                y_train=y_source,
                X_test=X_test,
                y_test=y_test,
                y_eval_test=y_eval_test,
                alpha=ridge_alpha,
                sample_weight=sample_weight,
            )

            all_result_rows.append(
                {
                    "repeat": int(repeat_idx),
                    "split_seed": int(repeat_seed),
                    "method": method_name,
                    "corr_eval": corr,
                    "mse_adj": mse,
                    "gain_over_v_full": corr - v_full,
                    "alpha": ridge_alpha,
                    "v_full": v_full,
                }
            )

        v_full_rows.append(
            {
                "repeat": int(repeat_idx),
                "split_seed": int(repeat_seed),
                "v_full": v_full,
            }
        )

    results_repeats_df = pd.DataFrame(all_result_rows)
    weights_repeats_df = pd.DataFrame(all_weight_rows)
    vfull_repeats_df = pd.DataFrame(v_full_rows)

    results_summary_df = (
        results_repeats_df.groupby("method", as_index=False)
        .agg(
            corr_mean=("corr_eval", "mean"),
            corr_std=("corr_eval", "std"),
            corr_p05=("corr_eval", lambda x: float(np.quantile(x, 0.05))),
            corr_p95=("corr_eval", lambda x: float(np.quantile(x, 0.95))),
            mse_mean=("mse_adj", "mean"),
            mse_std=("mse_adj", "std"),
            gain_mean=("gain_over_v_full", "mean"),
            gain_std=("gain_over_v_full", "std"),
            n_repeats=("corr_eval", "size"),
        )
        .sort_values("corr_mean", ascending=False)
        .reset_index(drop=True)
    )

    weight_summary_df = (
        weights_repeats_df.groupby(["method", "source_island", "source_island_name"], as_index=False)
        .agg(
            n_island=("n_island", "first"),
            phi_mean=("phi", "mean"),
            phi_per_ind_mean=("phi_per_individual", "mean"),
            weight_mean=("weight", "mean"),
            weight_std=("weight", "std"),
        )
    )

    results_repeats_path = output_dir / f"weighted_ridge_results_repeats_target_{target_code}.csv"
    results_summary_path = output_dir / f"weighted_ridge_results_summary_target_{target_code}.csv"
    weights_repeats_path = output_dir / f"weighted_island_weights_repeats_target_{target_code}.csv"
    weights_summary_path = output_dir / f"weighted_island_weights_summary_target_{target_code}.csv"
    vfull_repeats_path = output_dir / f"vfull_repeats_target_{target_code}.csv"
    summary_path = output_dir / f"weighted_ridge_summary_target_{target_code}.json"

    results_repeats_df.to_csv(results_repeats_path, index=False)
    results_summary_df.to_csv(results_summary_path, index=False)
    weights_repeats_df.to_csv(weights_repeats_path, index=False)
    weight_summary_df.to_csv(weights_summary_path, index=False)
    vfull_repeats_df.to_csv(vfull_repeats_path, index=False)

    summary = {
        "target_island": int(target_code),
        "target_island_name": str(target_name),
        "cal_fraction": float(cal_fraction),
        "n_permutations": int(n_permutations),
        "n_repeats": int(n_repeats),
        "ridge_alpha": float(ridge_alpha),
        "v_full_mean": float(vfull_repeats_df["v_full"].mean()),
        "v_full_std": float(vfull_repeats_df["v_full"].std(ddof=1)) if len(vfull_repeats_df) > 1 else 0.0,
        "best_method_by_mean_corr": str(results_summary_df.iloc[0]["method"]),
        "best_corr_mean": float(results_summary_df.iloc[0]["corr_mean"]),
        "methods_tested": ["unweighted"] + [str(m.get("name", "")) for m in methods],
        "outputs": {
            "results_repeats_csv": str(results_repeats_path),
            "results_summary_csv": str(results_summary_path),
            "weights_repeats_csv": str(weights_repeats_path),
            "weights_summary_csv": str(weights_summary_path),
            "vfull_repeats_csv": str(vfull_repeats_path),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved per-repeat weighted Ridge results: %s", results_repeats_path)
    logger.info("Saved weighted Ridge uncertainty summary: %s", results_summary_path)
    logger.info("Saved per-repeat island weights: %s", weights_repeats_path)
    logger.info("Saved island weight uncertainty summary: %s", weights_summary_path)
    logger.info("Saved summary: %s", summary_path)


if __name__ == "__main__":
    main()
