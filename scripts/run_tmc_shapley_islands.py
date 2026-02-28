#!/usr/bin/env python
"""
Run TMC-Shapley island-level analysis for one (or more) target islands.

For each target island the script:
1. Loads data, splits out target island into calibration + test sets
2. Runs TMC-Shapley on all source islands (possibly filtered via included_islands)
3. Produces Shapley value CSVs, add/remove-island curves, and plots

Usage
-----
    python scripts/run_tmc_shapley_islands.py --config config/tmc_shapley_islands.json
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.models import TrainParams
from src.tmc_shapley_islands import (
    ShapleyConfig,
    mirror_add_to_remove_curve,
    run_shapley_experiment,
)
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: resolve island specification to encoded code
# (identical to run_tracin_across_test.py version)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TMC-Shapley island-level analysis"
    )
    parser.add_argument(
        "--config", required=True, help="Path to tmc_shapley_islands.json config file"
    )
    parser.add_argument(
        "--target_islands",
        nargs="+",
        default=None,
        help="Override target_islands from config (encoded codes, labels, or names)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = json.load(f)

    # ------------------------------------------------------------------ paths
    paths = cfg["paths"]
    output_dir_base = paths.get("output_dir", "outputs/tmc_shapley")

    # ------------------------------------------------------------------ device
    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------ seed
    global_seed = cfg.get("seed", 42)
    set_seed(global_seed)

    # ------------------------------------------------------------------ load data
    logger.info("Loading data...")
    X, y, ids, GRM_df, locality, code_to_label, y_eval = load_data(
        paths=paths,
        target_column=cfg.get("target_column", "y_adjusted"),
        standardize_features=cfg.get("standardize_features", True),
        return_locality=True,
        min_count=cfg.get("min_count", 20),
        return_eval=True,
        eval_target_column=cfg.get("eval_target_column", "y_mean"),
    )
    logger.info(
        f"Data loaded: {X.shape[0]} samples, {X.shape[1]} SNPs, "
        f"{len(np.unique(locality))} islands"
    )

    present_codes = set(int(c) for c in np.unique(locality))

    # ------------------------------------------------------------------ train params
    tp_cfg = cfg["train_params"]
    train_params = TrainParams(
        lr=tp_cfg["lr"],
        weight_decay=tp_cfg.get("weight_decay", 1e-5),
        epochs=tp_cfg["epochs"],
        loss_name=tp_cfg.get("loss_name", "mse"),
        optimizer=tp_cfg.get("optimizer", "adam"),
        hidden_dims=tp_cfg.get("hidden_dims", [512, 64]),
        dropout=tp_cfg.get("dropout", 0.3),
        batch_norm=tp_cfg.get("batch_norm", False),
    )
    batch_size = tp_cfg.get("batch_size", cfg.get("batch_size", 64))

    # ------------------------------------------------------------------ Shapley config
    tmc_cfg_raw = cfg.get("tmc", {})
    shapley_cfg = ShapleyConfig(
        n_permutations=tmc_cfg_raw.get("n_permutations", 50),
        eps_trunc=tmc_cfg_raw.get("eps_trunc", 0.005),
        min_prefix_islands=tmc_cfg_raw.get("min_prefix_islands", 3),
        use_truncation=tmc_cfg_raw.get("use_truncation", False),
        island_sampling_mode=tmc_cfg_raw.get("island_sampling_mode", "mean"),
        cap_per_island=tmc_cfg_raw.get("cap_per_island", None),
        n_train_seeds_per_eval=tmc_cfg_raw.get("n_train_seeds_per_eval", 1),
        cache_dir=tmc_cfg_raw.get("cache_dir", None),
        n_random_orders=cfg.get("baselines", {}).get("n_random_orders", 1),
        use_early_stopping=tmc_cfg_raw.get("use_early_stopping", False),
        early_stopping_patience=tmc_cfg_raw.get("early_stopping_patience", 0),
        retrain_use_early_stopping=tmc_cfg_raw.get("retrain_use_early_stopping", True),
        retrain_early_stopping_patience=tmc_cfg_raw.get("retrain_early_stopping_patience", 20),
        snp_selection_mode=cfg.get("snp_selection_mode", "random"),
        use_snp_selection=cfg.get("use_snp_selection", False),
        num_snps=cfg.get("num_snps", None),
        seed=global_seed,
    )

    uncertainty_cfg = cfg.get("uncertainty", {})
    n_target_split_repeats = int(uncertainty_cfg.get("n_target_split_repeats", 5))
    if n_target_split_repeats < 1:
        raise ValueError("uncertainty.n_target_split_repeats must be >= 1")

    # ------------------------------------------------------------------ included islands
    included_raw = cfg.get("included_islands", None)
    if included_raw is not None:
        included_island_codes = [
            resolve_island_code(v, code_to_label, present_codes)
            for v in included_raw
        ]
        logger.info(f"Filtered source islands (included_islands): {included_island_codes}")
    else:
        included_island_codes = sorted(present_codes)
        logger.info(f"All {len(included_island_codes)} islands available as source")

    # ------------------------------------------------------------------ target islands
    raw_targets = args.target_islands if args.target_islands else cfg.get("target_islands", [0])
    target_codes = [
        resolve_island_code(t, code_to_label, present_codes)
        for t in raw_targets
    ]
    logger.info(f"Target islands to process: {target_codes}")

    # ------------------------------------------------------------------ per-target loop
    for target_code in target_codes:
        target_name = island_label(target_code, code_to_label)
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing target island: {target_code} ({target_name})")
        logger.info(f"{'='*60}")

        # Source islands = all included_island_codes except the target
        source_codes = [c for c in included_island_codes if c != target_code]
        if len(source_codes) == 0:
            logger.warning(f"No source islands for target {target_code}, skipping")
            continue

        # Build source_names dict
        source_names: dict = {c: island_label(c, code_to_label) for c in source_codes}

        output_dir = str(Path(output_dir_base) / f"island_{target_code}")

        repeat_results = []
        phi_rows = []
        add_rows = []

        for repeat_idx in range(n_target_split_repeats):
            repeat_seed = global_seed + 1_000 * repeat_idx
            repeat_cfg = copy.deepcopy(shapley_cfg)
            repeat_cfg.seed = repeat_seed

            repeat_output_dir = str(Path(output_dir) / f"repeat_{repeat_idx}")
            logger.info(
                f"Running repeat {repeat_idx + 1}/{n_target_split_repeats} "
                f"for target {target_code} with split seed={repeat_seed}"
            )

            try:
                result = run_shapley_experiment(
                    X=X,
                    y=y,
                    ids=ids,
                    locality=locality,
                    y_eval=y_eval,
                    target_island_code=target_code,
                    target_island_name=target_name,
                    source_codes=source_codes,
                    source_names=source_names,
                    train_params=train_params,
                    cfg=repeat_cfg,
                    output_dir=repeat_output_dir,
                    device=device,
                    batch_size=batch_size,
                    cal_fraction=cfg.get("cal_fraction", 0.2),
                    max_cal_fraction=cfg.get("max_cal_fraction", None),
                    use_snp_selection=cfg.get("use_snp_selection", False),
                    num_snps=cfg.get("num_snps", None),
                    snp_selection_mode=cfg.get("snp_selection_mode", "random"),
                )
            except Exception as e:
                logger.exception(
                    f"Repeat {repeat_idx} failed for island {target_code} ({target_name}): {e}"
                )
                continue

            repeat_results.append((repeat_idx, repeat_seed, result))

            phi = result["phi"]
            n_individuals = result["n_individuals"]
            for i, code in enumerate(source_codes):
                n_i = int(n_individuals[i])
                phi_rows.append({
                    "target_island": target_code,
                    "repeat": repeat_idx,
                    "split_seed": repeat_seed,
                    "source_island": code,
                    "source_island_name": source_names.get(code, str(code)),
                    "n_island": n_i,
                    "phi": float(phi[i]),
                    "phi_per_individual": float(phi[i] / max(n_i, 1)),
                    "v_full": float(result["v_full"]),
                })

            add_df = result["add_curve_df"].copy()
            add_df["target_island"] = target_code
            add_df["repeat"] = repeat_idx
            add_df["split_seed"] = repeat_seed
            add_rows.append(add_df)

        if len(repeat_results) == 0:
            logger.warning(
                f"All repeats failed for target {target_code} ({target_name}), skipping"
            )
            continue

        # Aggregate uncertainty across repeated target splits
        phi_repeats_df = pd.DataFrame(phi_rows)
        phi_repeats_path = Path(output_dir) / f"shapley_repeats_target_{target_code}.csv"
        phi_repeats_path.parent.mkdir(parents=True, exist_ok=True)
        phi_repeats_df.to_csv(phi_repeats_path, index=False)

        phi_summary_df = (
            phi_repeats_df
            .groupby(["target_island", "source_island", "source_island_name", "n_island"], as_index=False)
            .agg(
                phi_mean=("phi", "mean"),
                phi_std=("phi", "std"),
                phi_p05=("phi", lambda x: np.quantile(x, 0.05)),
                phi_p95=("phi", lambda x: np.quantile(x, 0.95)),
                phi_per_ind_mean=("phi_per_individual", "mean"),
                phi_per_ind_std=("phi_per_individual", "std"),
                n_repeats=("phi", "size"),
            )
        )
        phi_summary_df["rank_mean_phi_per_ind"] = (
            phi_summary_df["phi_per_ind_mean"].rank(ascending=False, method="min").astype(int)
        )
        phi_summary_path = Path(output_dir) / f"shapley_uncertainty_target_{target_code}.csv"
        phi_summary_df.to_csv(phi_summary_path, index=False)

        order_rows = []
        for repeat_idx in sorted(phi_repeats_df["repeat"].unique().tolist()):
            rep = phi_repeats_df[phi_repeats_df["repeat"] == repeat_idx].copy()
            rep = rep.sort_values("phi_per_individual", ascending=False).reset_index(drop=True)
            rep["rank"] = np.arange(1, len(rep) + 1)
            order_rows.append(rep[[
                "target_island", "repeat", "split_seed",
                "source_island", "source_island_name", "rank", "phi_per_individual"
            ]])
        order_df = pd.concat(order_rows, ignore_index=True)
        order_path = Path(output_dir) / f"add_order_repeats_target_{target_code}.csv"
        order_df.to_csv(order_path, index=False)

        add_repeats_df = pd.concat(add_rows, ignore_index=True)
        remove_repeats_df = mirror_add_to_remove_curve(add_repeats_df, n_source_islands=len(source_codes))

        add_repeats_path = Path(output_dir) / f"add_curve_repeats_target_{target_code}.csv"
        remove_repeats_path = Path(output_dir) / f"remove_curve_repeats_target_{target_code}.csv"
        add_repeats_df.to_csv(add_repeats_path, index=False)
        remove_repeats_df.to_csv(remove_repeats_path, index=False)

        add_summary_df = (
            add_repeats_df
            .groupby(["method", "n_islands"], as_index=False)
            .agg(
                corr_mean=("corr_eval", "mean"),
                corr_std=("corr_eval", "std"),
                mse_mean=("mse_adj", "mean"),
                mse_std=("mse_adj", "std"),
                n_rows=("corr_eval", "size"),
            )
        )
        add_summary_path = Path(output_dir) / f"add_curve_uncertainty_target_{target_code}.csv"
        add_summary_df.to_csv(add_summary_path, index=False)

        remove_summary_df = (
            remove_repeats_df
            .groupby(["method", "n_removed"], as_index=False)
            .agg(
                corr_mean=("corr_eval", "mean"),
                corr_std=("corr_eval", "std"),
                mse_mean=("mse_adj", "mean"),
                mse_std=("mse_adj", "std"),
                n_rows=("corr_eval", "size"),
            )
        )
        remove_summary_path = Path(output_dir) / f"remove_curve_uncertainty_target_{target_code}.csv"
        remove_summary_df.to_csv(remove_summary_path, index=False)

        summary_payload = {
            "target_island": int(target_code),
            "target_island_name": str(target_name),
            "n_target_split_repeats_requested": int(n_target_split_repeats),
            "n_target_split_repeats_completed": int(len(repeat_results)),
            "repeat_seeds_completed": [int(seed_i) for _, seed_i, _ in repeat_results],
            "n_random_orders": int(shapley_cfg.n_random_orders),
            "methods": ["shapley_mean", "random_individual"],
        }
        summary_json_path = Path(output_dir) / f"uncertainty_summary_target_{target_code}.json"
        with open(summary_json_path, "w") as f:
            json.dump(summary_payload, f, indent=2)

        logger.info(f"Completed island {target_code} ({target_name}), results in: {output_dir}")

    logger.info("All target islands processed.")


if __name__ == "__main__":
    main()
