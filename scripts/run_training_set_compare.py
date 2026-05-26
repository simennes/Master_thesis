#!/usr/bin/env python
"""
Unified training-set comparison: PEVmean-GA, Calibration-GA, and Random.

For each target island, training-set size, and repeat this script runs three
methods side-by-side and evaluates them on the same held-out test split:

  1. **PEVmean-GA** — selects source individuals that minimise PEVmean
     (genotype-only criterion; no phenotypes used for selection).
  2. **Calibration-GA** — selects source individuals that maximise
     ``corr_cal / n_individuals`` on a calibration subset of the target.
  3. **Random** — randomly selects individuals of the same size.

Shared ``n_train_sizes`` and ``n_repeats`` ensure identical experimental
grids across methods.

Efficiency
----------
PEVmean-GA does not depend on the cal/test split (genotype-only): it is
run **once** per (target, repeat, n_train_size) and then re-evaluated on
every cal/test split.  Calibration-GA is run per (target, repeat, n_cal,
n_train_size) since its fitness uses calibration phenotypes.

Sharding
--------
``num_shards == n_repeats``.  Each shard processes one repeat index
(all n_train_sizes and n_cal settings).

Usage
-----
Worker::

    python -m scripts.run_training_set_compare \\
        --mode worker --config config/training_set_compare_config.json

Merge shards::

    python -m scripts.run_training_set_compare \\
        --mode merge --config config/training_set_compare_config.json

Outputs
-------
``comparison_results.csv`` with columns:
    method, n_train_size, n_individuals, corr_eval_cal, corr_eval_test,
    mse_adj_test, pevmean_obj, fitness, target_island, target_island_name,
    n_cal_samples, repeat, repeat_seed, trait, order_seed,
    ga_generations, ga_elapsed_sec
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.training_set_optimization.ga_subset import GAConfig, run_ga
from src.training_set_optimization.pevmean import build_kernel, pev_mean_kernel
from src.utils import _pearson_corr, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_island_code(target_island, code_to_label: dict, present_codes: set) -> int:
    """Resolve a target island specification to its internal integer code."""
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
        raise ValueError(
            f"Island {val} not found. Present codes: {sorted(present_codes)}"
        )
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


def _as_int_list(v: Any, default: List[int]) -> List[int]:
    if v is None:
        return default
    if isinstance(v, list):
        return [int(x) for x in v]
    return [int(v)]


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    """Append *df* to a CSV, creating it (with header) if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def _make_cal_split_seed(
    global_seed: int, target_code: int, n_cal: float, repeat_idx: int,
) -> int:
    """Deterministic seed for cal/test split — matches calibration_ga."""
    token = (
        f"{int(global_seed)}|{int(target_code)}"
        f"|{float(n_cal):.12g}|{int(repeat_idx)}"
    )
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % 2_147_483_647


def _make_pev_seed(
    global_seed: int, target_code: int, repeat_idx: int,
) -> int:
    """Deterministic seed for PEVmean-GA — independent of cal settings."""
    token = f"pevmean|{int(global_seed)}|{int(target_code)}|{int(repeat_idx)}"
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % 2_147_483_647


def _build_trait_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build per-trait specs from config."""
    traits_cfg = cfg.get("traits", None)
    if traits_cfg is None:
        paths = dict(cfg["paths"])
        return [{
            "name": "default",
            "paths": paths,
            "target_column": cfg.get("target_column", "y_adjusted"),
            "eval_target_column": cfg.get("eval_target_column", "y_mean"),
            "standardize_features": cfg.get("standardize_features", True),
            "min_count": cfg.get("min_count", 20),
        }]
    specs: List[Dict[str, Any]] = []
    for t in traits_cfg:
        if not isinstance(t, dict):
            raise ValueError("traits must be a list of objects")
        if "name" not in t or "npz" not in t:
            raise ValueError("each trait must define 'name' and 'npz'")
        paths = dict(cfg["paths"])
        paths["npz"] = t["npz"]
        specs.append({
            "name": str(t["name"]),
            "paths": paths,
            "target_column": t.get(
                "target_column", cfg.get("target_column", "y_adjusted")
            ),
            "eval_target_column": t.get(
                "eval_target_column", cfg.get("eval_target_column", "y_mean")
            ),
            "standardize_features": t.get(
                "standardize_features", cfg.get("standardize_features", True)
            ),
            "min_count": int(
                t.get("min_count", cfg.get("min_count", 20))
            ),
        })
    return specs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_subset(
    train_indices: np.ndarray,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_cal: np.ndarray,
    y_eval_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    ridge_alpha: float,
    snp_cols: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Fit ridge on ``X_source[train_indices]``, evaluate on both cal and test.

    Returns dict with corr_eval_cal, corr_eval_test, mse_adj_test,
    n_individuals.
    """
    n_ind = int(len(train_indices))
    if n_ind < 2:
        return {
            "corr_eval_cal": 0.0,
            "corr_eval_test": 0.0,
            "mse_adj_test": float("inf"),
            "n_individuals": n_ind,
        }

    X_train = X_source[train_indices]
    y_train = y_source[train_indices]

    if snp_cols is not None:
        X_train_sel = X_train[:, snp_cols]
        X_cal_sel = X_cal[:, snp_cols]
        X_test_sel = X_test[:, snp_cols]
    else:
        X_train_sel = X_train
        X_cal_sel = X_cal
        X_test_sel = X_test

    model = Ridge(alpha=max(float(ridge_alpha), 1e-12))
    model.fit(X_train_sel, y_train)

    pred_cal = model.predict(X_cal_sel)
    pred_test = model.predict(X_test_sel)

    corr_cal = float(_pearson_corr(pred_cal, y_eval_cal))
    if not np.isfinite(corr_cal):
        corr_cal = 0.0

    corr_test = float(_pearson_corr(pred_test, y_eval_test))
    if not np.isfinite(corr_test):
        corr_test = 0.0

    mse_test = float(np.mean((pred_test - y_test) ** 2))

    return {
        "corr_eval_cal": corr_cal,
        "corr_eval_test": corr_test,
        "mse_adj_test": mse_test,
        "n_individuals": n_ind,
    }


# ---------------------------------------------------------------------------
# Merge mode
# ---------------------------------------------------------------------------

def run_merge(config_path: Path) -> None:
    """Merge shard outputs into per-trait CSVs."""
    with open(config_path, "r") as f:
        cfg = json.load(f)

    exp_cfg = cfg.get("experiment", {})
    output_dir = Path(
        exp_cfg.get(
            "output_dir",
            cfg["paths"].get("output_dir", "outputs/training_set_compare"),
        )
    )
    trait_specs = _build_trait_specs(cfg)
    shards_root = output_dir / "shards"

    if not shards_root.exists():
        raise FileNotFoundError(f"No shard directory found: {shards_root}")

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        parts: List[pd.DataFrame] = []

        for csv_path in sorted(
            shards_root.glob(f"shard_*/{trait_name}/comparison_results.csv")
        ):
            parts.append(pd.read_csv(csv_path))

        if not parts:
            logger.warning("No shard CSVs for trait '%s'", trait_name)
            continue

        merged = pd.concat(parts, ignore_index=True)
        trait_output = (
            output_dir / trait_name if len(trait_specs) > 1 else output_dir
        )
        trait_output.mkdir(parents=True, exist_ok=True)

        out_path = trait_output / "comparison_results.csv"
        merged.to_csv(out_path, index=False)

        summary = (
            merged.groupby(
                ["trait", "target_island", "method", "n_train_size", "n_cal_samples"],
                as_index=False,
            )
            .agg(
                corr_test_mean=("corr_eval_test", "mean"),
                corr_test_std=("corr_eval_test", "std"),
                corr_cal_mean=("corr_eval_cal", "mean"),
                mse_test_mean=("mse_adj_test", "mean"),
                n_rows=("corr_eval_test", "size"),
            )
        )
        summary_path = trait_output / "comparison_summary.csv"
        summary.to_csv(summary_path, index=False)

        logger.info(
            "Merged %d shards for trait '%s' → %s (%d rows)",
            len(parts), trait_name, out_path, len(merged),
        )

    logger.info("Merge complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified training-set comparison"
    )
    parser.add_argument(
        "--mode", choices=["worker", "merge"], default="worker",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--target_islands", nargs="+", default=None,
        help="Override target_islands from config",
    )
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    if args.mode == "merge":
        run_merge(config_path)
        return

    # ---- Shared experiment settings ------------------------------------------
    exp_cfg = cfg.get("experiment", {})
    output_dir = Path(
        exp_cfg.get(
            "output_dir",
            cfg["paths"].get("output_dir", "outputs/training_set_compare"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    global_seed = int(cfg.get("seed", 42))
    set_seed(global_seed)

    model_cfg = cfg.get("model", {})
    ridge_alpha = float(model_cfg.get("alpha", 1e5))

    n_train_sizes = _as_int_list(exp_cfg.get("n_train_sizes"), default=[])
    if not n_train_sizes:
        raise ValueError("experiment.n_train_sizes must be a non-empty list")

    n_cal_samples_list = _as_int_list(
        exp_cfg.get("n_cal_samples", 40), default=[40],
    )
    n_repeats = int(exp_cfg.get("n_repeats", 5))
    n_random_orders = int(cfg.get("baselines", {}).get("n_random_orders", 5))

    # ---- Calibration-GA hyper-params -----------------------------------------
    cal_raw = cfg.get("calibration_ga", {})
    cal_ga_template = GAConfig(
        pop_size=int(cal_raw.get("pop_size", 30)),
        n_generations=int(cal_raw.get("n_generations", 200)),
        n_elite=int(cal_raw.get("n_elite", 2)),
        tournament_k=int(cal_raw.get("tournament_k", 3)),
        crossover_prob=float(cal_raw.get("crossover_prob", 0.9)),
        mutation_prob=float(cal_raw.get("mutation_prob", 0.3)),
        n_swaps_per_mut=int(cal_raw.get("n_swaps_per_mut", 1)),
        seed=global_seed,
        verbose=bool(cal_raw.get("verbose", True)),
        stagnation_limit=int(cal_raw.get("stagnation_limit", 15)),
    )

    # ---- PEVmean-GA hyper-params ---------------------------------------------
    pev_raw = cfg.get("pevmean_ga", {})
    pev_ga_template = GAConfig(
        pop_size=int(pev_raw.get("pop_size", 50)),
        n_generations=int(pev_raw.get("n_generations", 100)),
        n_elite=int(pev_raw.get("n_elite", 2)),
        tournament_k=int(pev_raw.get("tournament_k", 3)),
        crossover_prob=float(pev_raw.get("crossover_prob", 0.9)),
        mutation_prob=float(pev_raw.get("mutation_prob", 0.3)),
        n_swaps_per_mut=int(pev_raw.get("n_swaps_per_mut", 2)),
        seed=global_seed,
        verbose=bool(pev_raw.get("verbose", True)),
        stagnation_limit=int(pev_raw.get("stagnation_limit", 20)),
    )

    # ---- SNP selection -------------------------------------------------------
    use_snp_selection = bool(cfg.get("use_snp_selection", False))
    num_snps = cfg.get("num_snps", None)

    # ---- Traits --------------------------------------------------------------
    trait_specs = _build_trait_specs(cfg)

    # ---- Shard controls (one shard per repeat) -------------------------------
    shard_index = args.shard_index
    if shard_index is None:
        shard_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    num_shards = args.num_shards
    if num_shards is None:
        num_shards = int(os.environ.get("SWEEP_NUM_SHARDS", "1"))

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}]")

    if num_shards > 1:
        if num_shards != n_repeats:
            raise ValueError(
                f"num_shards ({num_shards}) must equal "
                f"n_repeats ({n_repeats}) for repeat-level sharding."
            )
        assigned_repeat: Optional[int] = shard_index
        logger.info(
            "Shard %d/%d → repeat %d", shard_index, num_shards, assigned_repeat,
        )
    else:
        assigned_repeat = None  # run all repeats

    logger.info("Calibration-GA config: %s", cal_ga_template)
    logger.info("PEVmean-GA config:     %s", pev_ga_template)
    logger.info("n_train_sizes=%s, n_cal_samples=%s, n_repeats=%d",
                n_train_sizes, n_cal_samples_list, n_repeats)

    # ---- Per-trait loop ------------------------------------------------------
    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])

        if num_shards > 1:
            trait_output = (
                output_dir / "shards" / f"shard_{shard_index:03d}" / trait_name
            )
        else:
            trait_output = (
                output_dir / trait_name if len(trait_specs) > 1 else output_dir
            )
        trait_output.mkdir(parents=True, exist_ok=True)

        logger.info("Loading data for trait '%s'", trait_name)
        X, y, ids, GRM_df, locality, code_to_label, y_eval = load_data(
            paths=trait_spec["paths"],
            target_column=trait_spec["target_column"],
            standardize_features=trait_spec["standardize_features"],
            return_locality=True,
            min_count=trait_spec["min_count"],
            return_eval=True,
            eval_target_column=trait_spec["eval_target_column"],
        )
        logger.info("Loaded: %d samples, %d SNPs", X.shape[0], X.shape[1])

        present_codes = set(int(c) for c in np.unique(locality))
        included_raw = cfg.get("included_islands", None)
        if included_raw is not None:
            included_island_codes = [
                resolve_island_code(v, code_to_label, present_codes)
                for v in included_raw
            ]
        else:
            included_island_codes = sorted(present_codes)

        raw_targets = (
            args.target_islands
            if args.target_islands
            else cfg.get("target_islands", [0])
        )
        target_codes = [
            resolve_island_code(t, code_to_label, present_codes)
            for t in raw_targets
        ]

        # Fixed SNP subset
        fixed_snp_cols: Optional[np.ndarray] = None
        if (
            use_snp_selection
            and num_snps is not None
            and int(num_snps) < X.shape[1]
        ):
            rng_snp = np.random.default_rng(global_seed)
            fixed_snp_cols = np.sort(
                rng_snp.choice(X.shape[1], size=int(num_snps), replace=False)
                .astype(np.int64)
            )
            logger.info("Fixed SNP subset: n=%d", len(fixed_snp_cols))

        results_path = trait_output / "comparison_results.csv"
        if results_path.exists():
            results_path.unlink()

        # ---- Per-target loop -------------------------------------------------
        for target_code in target_codes:
            target_name = island_label(target_code, code_to_label)
            source_codes = [
                c for c in included_island_codes if c != target_code
            ]
            if not source_codes:
                logger.warning(
                    "No source islands for target %s, skipping", target_code
                )
                continue

            # Separate source and full target
            target_mask = locality == target_code
            source_mask = ~target_mask & np.isin(locality, source_codes)

            X_source = X[source_mask]
            y_source = y[source_mask]
            y_eval_source = y_eval[source_mask]

            X_target_full = X[target_mask]
            y_target_full = y[target_mask]
            y_eval_target_full = y_eval[target_mask]

            N_source = int(len(X_source))
            N_target = int(len(X_target_full))
            snp_cols = fixed_snp_cols

            logger.info(
                "Target %d (%s): n_source=%d, n_target=%d",
                target_code, target_name, N_source, N_target,
            )

            # Clip train sizes to available source pool
            valid_sizes = [s for s in n_train_sizes if 2 <= s < N_source]
            if not valid_sizes:
                logger.warning(
                    "No valid train sizes for target %s (N_source=%d), skipping",
                    target_name, N_source,
                )
                continue

            # ---- Build PEVmean kernel (source + full target) -----------------
            if snp_cols is not None:
                X_cand_sel = X_source[:, snp_cols]
                X_tgt_sel = X_target_full[:, snp_cols]
            else:
                X_cand_sel = X_source
                X_tgt_sel = X_target_full

            logger.info(
                "Building PEVmean kernel (%d + %d = %d) ...",
                N_source, N_target, N_source + N_target,
            )
            X_all_pev = np.vstack([X_cand_sel, X_tgt_sel])
            kernel_K, diag_K = build_kernel(X_all_pev)
            cand_idx = np.arange(N_source, dtype=np.int64)
            target_pev_idx = np.arange(
                N_source, N_source + N_target, dtype=np.int64
            )
            del X_all_pev
            logger.info("Kernel built: shape %s", kernel_K.shape)

            # PEVmean for all individuals (reference value)
            all_pev_obj = float(
                pev_mean_kernel(kernel_K, diag_K, cand_idx, target_pev_idx, lam=ridge_alpha)
            )
            logger.info("PEVmean(all source): %.6f", all_pev_obj)

            # ---- Repeat loop -------------------------------------------------
            repeat_indices = (
                [assigned_repeat]
                if assigned_repeat is not None
                else list(range(n_repeats))
            )

            for repeat_idx in repeat_indices:
                logger.info(
                    "=" * 70 + "\nRepeat %d/%d | target %d (%s)\n" + "=" * 70,
                    repeat_idx + 1, n_repeats, target_code, target_name,
                )

                # ============================================================
                # Phase 1: PEVmean-GA (once per n_train, independent of cal)
                # ============================================================
                pev_seed = _make_pev_seed(
                    global_seed, int(target_code), int(repeat_idx)
                )
                pev_selections: Dict[int, np.ndarray] = {}
                pev_objectives: Dict[int, float] = {}
                pev_stats: Dict[int, dict] = {}
                pev_elapsed: Dict[int, float] = {}

                logger.info("--- Phase 1: PEVmean-GA ---")
                for step_i, n_train in enumerate(valid_sizes):
                    n_train = int(n_train)

                    step_cfg = copy.deepcopy(pev_ga_template)
                    step_cfg.seed = int(
                        (pev_seed + step_i * 7919) % 2_147_483_647
                    )
                    step_cfg.n_swaps_per_mut = min(
                        step_cfg.n_swaps_per_mut, max(1, n_train - 1)
                    )

                    def pev_fitness(subset: np.ndarray) -> float:
                        return pev_mean_kernel(
                            kernel_K, diag_K, subset, target_pev_idx,
                            lam=ridge_alpha,
                        )

                    t0 = time.perf_counter()
                    best_sub, best_obj, stats = run_ga(
                        n_candidates=N_source,
                        n_train=n_train,
                        fitness_fn=pev_fitness,
                        cfg=step_cfg,
                        candidate_indices=cand_idx,
                    )
                    el = time.perf_counter() - t0

                    pev_selections[n_train] = best_sub
                    pev_objectives[n_train] = float(best_obj)
                    pev_stats[n_train] = stats
                    pev_elapsed[n_train] = el

                    logger.info(
                        "  PEVmean n_train=%d: PEV=%.6f (%.1fs, %d gens)",
                        n_train, best_obj, el,
                        stats.get("generations_run", 0),
                    )

                # ============================================================
                # Phase 2: per cal setting — Calibration-GA + evaluate all
                # ============================================================
                for n_cal in n_cal_samples_list:
                    if n_cal >= N_target:
                        logger.warning(
                            "n_cal=%d >= n_target=%d, skipping", n_cal, N_target,
                        )
                        continue

                    cal_split_seed = _make_cal_split_seed(
                        global_seed, int(target_code),
                        float(n_cal), int(repeat_idx),
                    )

                    # deterministic cal/test split
                    rng_split = np.random.default_rng(cal_split_seed)
                    perm = rng_split.permutation(N_target)
                    cal_local = perm[:n_cal]
                    test_local = perm[n_cal:]

                    X_cal = X_target_full[cal_local]
                    y_eval_cal = y_eval_target_full[cal_local]

                    X_test = X_target_full[test_local]
                    y_test = y_target_full[test_local]
                    y_eval_test = y_eval_target_full[test_local]

                    logger.info(
                        "--- Phase 2: n_cal=%d (seed=%d) | "
                        "cal=%d, test=%d ---",
                        n_cal, cal_split_seed, len(X_cal), len(X_test),
                    )

                    # ---- All-individuals baseline ----------------------------
                    all_indices = np.arange(N_source, dtype=np.int64)
                    all_ev = _evaluate_subset(
                        all_indices, X_source, y_source,
                        X_cal, y_eval_cal,
                        X_test, y_test, y_eval_test,
                        ridge_alpha, snp_cols,
                    )
                    all_n = int(all_ev["n_individuals"])
                    all_fit = (
                        float(all_ev["corr_eval_cal"]) / all_n
                        if all_n > 0 else 0.0
                    )

                    _append_csv(pd.DataFrame([{
                        "method": "all_individuals",
                        "n_train_size": N_source,
                        "n_individuals": all_n,
                        "corr_eval_cal": float(all_ev["corr_eval_cal"]),
                        "corr_eval_test": float(all_ev["corr_eval_test"]),
                        "mse_adj_test": float(all_ev["mse_adj_test"]),
                        "pevmean_obj": float(all_pev_obj),
                        "fitness": float(all_fit),
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "n_cal_samples": int(n_cal),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(cal_split_seed),
                        "trait": trait_name,
                        "order_seed": -2,
                        "ga_generations": 0,
                        "ga_elapsed_sec": 0.0,
                    }]), results_path)

                    logger.info(
                        "  all_individuals: corr_test=%.4f corr_cal=%.4f",
                        all_ev["corr_eval_test"], all_ev["corr_eval_cal"],
                    )

                    # ---- Per n_train_size ------------------------------------
                    for n_train in valid_sizes:
                        n_train = int(n_train)

                        # --- Calibration-GA -----------------------------------
                        cal_ga_seed = int(
                            (cal_split_seed + n_train * 7919) % 2_147_483_647
                        )
                        cal_cfg = copy.deepcopy(cal_ga_template)
                        cal_cfg.seed = cal_ga_seed
                        cal_cfg.n_swaps_per_mut = min(
                            cal_cfg.n_swaps_per_mut, max(1, n_train - 1)
                        )

                        def cal_fitness(ind_subset: np.ndarray) -> float:
                            """Minimise -(corr_cal / n)."""
                            n = int(len(ind_subset))
                            if n < 2:
                                return float("inf")
                            X_tr = X_source[ind_subset]
                            y_tr = y_source[ind_subset]
                            if snp_cols is not None:
                                pred = Ridge(
                                    alpha=max(ridge_alpha, 1e-12)
                                ).fit(
                                    X_tr[:, snp_cols], y_tr
                                ).predict(X_cal[:, snp_cols])
                            else:
                                pred = Ridge(
                                    alpha=max(ridge_alpha, 1e-12)
                                ).fit(X_tr, y_tr).predict(X_cal)
                            c = float(_pearson_corr(pred, y_eval_cal))
                            if not np.isfinite(c):
                                c = 0.0
                            return -(c / n)

                        t0 = time.perf_counter()
                        cal_best, cal_best_fit, cal_stats = run_ga(
                            n_candidates=N_source,
                            n_train=n_train,
                            fitness_fn=cal_fitness,
                            cfg=cal_cfg,
                        )
                        cal_elapsed = time.perf_counter() - t0

                        cal_ev = _evaluate_subset(
                            cal_best, X_source, y_source,
                            X_cal, y_eval_cal,
                            X_test, y_test, y_eval_test,
                            ridge_alpha, snp_cols,
                        )
                        cal_n = int(cal_ev["n_individuals"])
                        cal_fit = (
                            float(cal_ev["corr_eval_cal"]) / cal_n
                            if cal_n > 0 else 0.0
                        )

                        _append_csv(pd.DataFrame([{
                            "method": "calibration_ga",
                            "n_train_size": n_train,
                            "n_individuals": cal_n,
                            "corr_eval_cal": float(cal_ev["corr_eval_cal"]),
                            "corr_eval_test": float(cal_ev["corr_eval_test"]),
                            "mse_adj_test": float(cal_ev["mse_adj_test"]),
                            "pevmean_obj": float("nan"),
                            "fitness": float(cal_fit),
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "n_cal_samples": int(n_cal),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(cal_split_seed),
                            "trait": trait_name,
                            "order_seed": -1,
                            "ga_generations": int(
                                cal_stats.get("generations_run", 0)
                            ),
                            "ga_elapsed_sec": float(cal_elapsed),
                        }]), results_path)

                        logger.info(
                            "    calibration_ga n=%d: corr_test=%.4f "
                            "corr_cal=%.4f fitness=%.6f (%.1fs)",
                            n_train, cal_ev["corr_eval_test"],
                            cal_ev["corr_eval_cal"], cal_fit, cal_elapsed,
                        )

                        # --- PEVmean-GA: evaluate on this cal/test split ------
                        pev_sub = pev_selections[n_train]
                        pev_ev = _evaluate_subset(
                            pev_sub, X_source, y_source,
                            X_cal, y_eval_cal,
                            X_test, y_test, y_eval_test,
                            ridge_alpha, snp_cols,
                        )
                        pev_n = int(pev_ev["n_individuals"])
                        pev_fit = (
                            float(pev_ev["corr_eval_cal"]) / pev_n
                            if pev_n > 0 else 0.0
                        )

                        _append_csv(pd.DataFrame([{
                            "method": "pevmean_ga",
                            "n_train_size": n_train,
                            "n_individuals": pev_n,
                            "corr_eval_cal": float(pev_ev["corr_eval_cal"]),
                            "corr_eval_test": float(pev_ev["corr_eval_test"]),
                            "mse_adj_test": float(pev_ev["mse_adj_test"]),
                            "pevmean_obj": float(
                                pev_objectives.get(n_train, float("nan"))
                            ),
                            "fitness": float(pev_fit),
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "n_cal_samples": int(n_cal),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(cal_split_seed),
                            "trait": trait_name,
                            "order_seed": -1,
                            "ga_generations": int(
                                pev_stats.get(n_train, {}).get(
                                    "generations_run", 0
                                )
                            ),
                            "ga_elapsed_sec": float(
                                pev_elapsed.get(n_train, 0.0)
                            ),
                        }]), results_path)

                        logger.info(
                            "    pevmean_ga   n=%d: corr_test=%.4f "
                            "corr_cal=%.4f PEV=%.6f",
                            n_train, pev_ev["corr_eval_test"],
                            pev_ev["corr_eval_cal"],
                            pev_objectives.get(n_train, float("nan")),
                        )

                        # --- Random baselines ---------------------------------
                        rand_corrs: List[float] = []
                        for r_order in range(n_random_orders):
                            rand_rng = np.random.default_rng(
                                cal_ga_seed + 1_000_000 + r_order
                            )
                            chosen = np.sort(
                                rand_rng.choice(
                                    N_source, size=n_train, replace=False,
                                )
                            )

                            rand_ev = _evaluate_subset(
                                chosen, X_source, y_source,
                                X_cal, y_eval_cal,
                                X_test, y_test, y_eval_test,
                                ridge_alpha, snp_cols,
                            )
                            rand_n = int(rand_ev["n_individuals"])
                            rand_fit = (
                                float(rand_ev["corr_eval_cal"]) / rand_n
                                if rand_n > 0 else 0.0
                            )

                            _append_csv(pd.DataFrame([{
                                "method": "random",
                                "n_train_size": n_train,
                                "n_individuals": rand_n,
                                "corr_eval_cal": float(
                                    rand_ev["corr_eval_cal"]
                                ),
                                "corr_eval_test": float(
                                    rand_ev["corr_eval_test"]
                                ),
                                "mse_adj_test": float(
                                    rand_ev["mse_adj_test"]
                                ),
                                "pevmean_obj": float("nan"),
                                "fitness": float(rand_fit),
                                "target_island": int(target_code),
                                "target_island_name": str(target_name),
                                "n_cal_samples": int(n_cal),
                                "repeat": int(repeat_idx),
                                "repeat_seed": int(cal_split_seed),
                                "trait": trait_name,
                                "order_seed": r_order,
                                "ga_generations": 0,
                                "ga_elapsed_sec": 0.0,
                            }]), results_path)
                            rand_corrs.append(
                                float(rand_ev["corr_eval_test"])
                            )

                        logger.info(
                            "    random (%d): mean corr_test=%.4f",
                            n_random_orders,
                            float(np.mean(rand_corrs))
                            if rand_corrs else float("nan"),
                        )

        # ---- Per-trait summary -----------------------------------------------
        if results_path.exists():
            all_df = pd.read_csv(results_path)
            summary = (
                all_df.groupby(
                    ["trait", "target_island", "method",
                     "n_train_size", "n_cal_samples"],
                    as_index=False,
                )
                .agg(
                    corr_test_mean=("corr_eval_test", "mean"),
                    corr_test_std=("corr_eval_test", "std"),
                    corr_cal_mean=("corr_eval_cal", "mean"),
                    mse_test_mean=("mse_adj_test", "mean"),
                    n_rows=("corr_eval_test", "size"),
                )
            )
            summary_path = trait_output / "comparison_summary.csv"
            summary.to_csv(summary_path, index=False)
            logger.info("Summary saved to %s", summary_path)

        logger.info("Trait '%s' complete. Results: %s", trait_name, results_path)

    logger.info("All traits done.")


if __name__ == "__main__":
    main()
