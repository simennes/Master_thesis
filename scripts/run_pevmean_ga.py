#!/usr/bin/env python
"""
Run PEVmean-GA training-set optimisation.

For each target island, select training sets of given sizes from the
candidate pool (all non-target islands) by minimising PEVmean — a
genotype-only criterion.  Then fit ridge regression using the selected
individuals and evaluate on the full target island.  A random-selection
baseline of the same sizes is run alongside for comparison.

No calibration/test split of the target island is needed because PEVmean
does not use phenotypes to select the training set.

Usage
-----
    python -m scripts.run_pevmean_ga --config config/pevmean_ga_config.json

Config keys
-----------
- ``pevmean_ga.n_train_sizes``: list of training-set sizes to evaluate.
  If null / omitted, sizes are derived from cumulative island counts
  (same as the Shapley add-curve steps).
- ``pevmean_ga.n_repeats``: number of GA repeats per (target, n_train)
  combination (different GA seeds for robustness).

Outputs
-------
``pevmean_ga_results.csv`` – one row per (target, n_train, method, repeat)
with columns compatible with the Shapley sweep CSVs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.training_set_optimization.ga_subset import GAConfig, run_ga
from src.training_set_optimization.pevmean import build_kernel, pev_mean
from src.training_set_optimization.runner import _evaluate_ridge_subset
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _as_int_list(v, default):
    if v is None:
        return default
    if isinstance(v, list):
        return [int(x) for x in v]
    return [int(v)]


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
        raise ValueError(f"Island {val} not found")
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


def _make_repeat_seed(global_seed: int, target_code: int, repeat_idx: int) -> int:
    """Deterministic per-(target, repeat) seed."""
    token = f"pevmean|{int(global_seed)}|{int(target_code)}|{int(repeat_idx)}"
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % 2_147_483_647


def _build_trait_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            "target_column": t.get("target_column", cfg.get("target_column", "y_adjusted")),
            "eval_target_column": t.get("eval_target_column", cfg.get("eval_target_column", "y_mean")),
            "standardize_features": t.get("standardize_features", cfg.get("standardize_features", True)),
            "min_count": int(t.get("min_count", cfg.get("min_count", 20))),
        })
    return specs


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def _resolve_shard_assignment(
    n_repeats: int,
    n_train_sizes_list: List[int],
    shard_index: int,
    num_shards: int,
) -> tuple[Optional[int], Optional[int]]:
    """
    Map shard -> (repeat_idx, n_train_size) when sharding is enabled.

    If num_shards == 1, returns (None, None) and caller runs full sweep.
    """
    if num_shards == 1:
        return None, None

    if len(n_train_sizes_list) == 0:
        raise ValueError(
            "Sharded execution requires pevmean_ga.n_train_sizes to be a non-empty list."
        )

    expected = int(n_repeats) * int(len(n_train_sizes_list))
    if int(num_shards) != expected:
        raise ValueError(
            f"num_shards ({num_shards}) must equal "
            f"n_repeats * len(n_train_sizes) ({expected}) "
            f"to assign one (repeat, n_train_size) pair per shard."
        )

    repeat_idx = int(shard_index) // int(len(n_train_sizes_list))
    n_train_size = int(n_train_sizes_list[int(shard_index) % int(len(n_train_sizes_list))])
    return repeat_idx, n_train_size


def run_merge(config_path: Path) -> None:
    """Merge shard outputs into the same per-trait CSV layout as non-sharded runs."""
    with open(config_path, "r") as f:
        cfg = json.load(f)

    paths_cfg = cfg["paths"]
    output_dir = Path(cfg.get("pevmean_ga", {}).get(
        "output_dir",
        paths_cfg.get("output_dir", "outputs/pevmean_ga"),
    ))
    trait_specs = _build_trait_specs(cfg)
    shards_root = output_dir / "shards"

    if not shards_root.exists():
        raise FileNotFoundError(f"No shard directory found: {shards_root}")

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        parts: List[pd.DataFrame] = []

        for csv_path in sorted(shards_root.glob(f"shard_*/{trait_name}/pevmean_ga_results.csv")):
            if csv_path.exists():
                parts.append(pd.read_csv(csv_path))

        if len(parts) == 0:
            logger.warning("No shard CSVs found for trait '%s'; skipping merge.", trait_name)
            continue

        merged = pd.concat(parts, ignore_index=True)
        trait_output = output_dir / trait_name if len(trait_specs) > 1 else output_dir
        trait_output.mkdir(parents=True, exist_ok=True)

        out_path = trait_output / "pevmean_ga_results.csv"
        merged.to_csv(out_path, index=False)

        summary = (
            merged.groupby(
                ["trait", "target_island", "method", "n_individuals"],
                as_index=False,
            )
            .agg(
                corr_mean=("corr_eval", "mean"),
                corr_std=("corr_eval", "std"),
                mse_mean=("mse_adj", "mean"),
                n_rows=("corr_eval", "size"),
            )
        )
        summary_path = trait_output / "pevmean_ga_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(
            "Merged %d shard files for trait '%s' into %s (%d rows)",
            len(parts),
            trait_name,
            out_path,
            len(merged),
        )

    logger.info("Merge complete.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEVmean-GA training set optimisation")
    parser.add_argument(
        "--mode",
        choices=["worker", "merge"],
        default="worker",
        help="worker: run one shard/full run, merge: combine shard outputs",
    )
    parser.add_argument("--config", required=True, help="Path to config JSON")
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

    # ---- Output ---------------------------------------------------------------
    paths_cfg = cfg["paths"]
    output_dir = Path(cfg.get("pevmean_ga", {}).get(
        "output_dir",
        paths_cfg.get("output_dir", "outputs/pevmean_ga"),
    ))
    output_dir.mkdir(parents=True, exist_ok=True)

    global_seed = int(cfg.get("seed", 42))
    set_seed(global_seed)

    # ---- Model ----------------------------------------------------------------
    model_cfg = cfg.get("model", {})
    ridge_alpha = float(model_cfg.get("alpha", 1e5))

    # ---- GA config ------------------------------------------------------------
    ga_raw = cfg.get("pevmean_ga", {})
    ga_cfg = GAConfig(
        pop_size=int(ga_raw.get("pop_size", 50)),
        n_generations=int(ga_raw.get("n_generations", 100)),
        n_elite=int(ga_raw.get("n_elite", 2)),
        tournament_k=int(ga_raw.get("tournament_k", 3)),
        crossover_prob=float(ga_raw.get("crossover_prob", 0.9)),
        mutation_prob=float(ga_raw.get("mutation_prob", 0.3)),
        n_swaps_per_mut=int(ga_raw.get("n_swaps_per_mut", 2)),
        seed=global_seed,
        verbose=bool(ga_raw.get("verbose", True)),
        stagnation_limit=int(ga_raw.get("stagnation_limit", 0)),
    )
    logger.info("GA config: %s", ga_cfg)

    # ---- Experiment settings --------------------------------------------------
    n_repeats = int(ga_raw.get("n_repeats", cfg.get("sweep", {}).get("n_target_split_repeats", 5)))
    n_random_orders = int(cfg.get("baselines", {}).get("n_random_orders", 5))
    n_train_sizes_raw = ga_raw.get("n_train_sizes", None)

    # For sharded mode we require explicit train sizes in config.
    n_train_sizes_cfg = _as_int_list(n_train_sizes_raw, default=[]) if n_train_sizes_raw is not None else []

    # ---- SNP selection --------------------------------------------------------
    use_snp_selection = bool(cfg.get("use_snp_selection", False))
    num_snps = cfg.get("num_snps", None)

    # ---- Traits ---------------------------------------------------------------
    trait_specs = _build_trait_specs(cfg)

    # ---- Shard controls -------------------------------------------------------
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

    assigned_repeat_idx, assigned_n_train = _resolve_shard_assignment(
        n_repeats=n_repeats,
        n_train_sizes_list=n_train_sizes_cfg,
        shard_index=shard_index,
        num_shards=num_shards,
    )

    # In sharded mode, each repeat is split across n_train_sizes shards.
    # Write all-individual baselines from exactly one shard per repeat to
    # avoid duplicate rows in merged outputs.
    if assigned_n_train is None:
        write_all_individuals_baseline = True
    else:
        write_all_individuals_baseline = (
            len(n_train_sizes_cfg) > 0 and int(assigned_n_train) == int(min(n_train_sizes_cfg))
        )

    if num_shards > 1:
        logger.info(
            "Shard %d/%d assigned repeat=%s, n_train_size=%s",
            shard_index,
            num_shards,
            assigned_repeat_idx,
            assigned_n_train,
        )

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        if num_shards > 1:
            trait_output = output_dir / "shards" / f"shard_{shard_index:03d}" / trait_name
        else:
            trait_output = output_dir / trait_name if len(trait_specs) > 1 else output_dir
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
            included_island_codes = [resolve_island_code(v, code_to_label, present_codes) for v in included_raw]
        else:
            included_island_codes = sorted(present_codes)

        raw_targets = args.target_islands if args.target_islands else cfg.get("target_islands", [0])
        target_codes = [resolve_island_code(t, code_to_label, present_codes) for t in raw_targets]

        # Fixed SNP subset (same logic as Shapley sweep)
        fixed_snp_cols: Optional[np.ndarray] = None
        if use_snp_selection and num_snps is not None and int(num_snps) < X.shape[1]:
            rng_snp = np.random.default_rng(global_seed)
            fixed_snp_cols = np.sort(rng_snp.choice(X.shape[1], size=int(num_snps), replace=False).astype(np.int64))
            logger.info("Fixed SNP subset: n=%d", len(fixed_snp_cols))

        result_rows_path = trait_output / "pevmean_ga_results.csv"
        if result_rows_path.exists():
            result_rows_path.unlink()

        total_jobs = len(target_codes) * n_repeats
        done_jobs = 0

        for target_code in target_codes:
            target_name = island_label(target_code, code_to_label)
            source_codes = [c for c in included_island_codes if c != target_code]
            if len(source_codes) == 0:
                logger.warning("No source islands for target %s, skipping", target_code)
                continue

            # ---- Separate source (candidates) and target (full island) --------
            target_mask = (locality == target_code)
            source_mask = ~target_mask

            # Filter source to included islands
            source_mask = source_mask & np.isin(locality, source_codes)

            X_source = X[source_mask]
            y_source = y[source_mask]
            locality_source = locality[source_mask]

            X_target = X[target_mask]
            y_target = y[target_mask]  # adjusted phenotype (for MSE)
            y_eval_target = y_eval[target_mask]  # original phenotype (for Pearson r)

            N_source = len(X_source)
            N_target = len(X_target)

            logger.info(
                "Target %s (%s): n_source=%d, n_target=%d",
                target_code, target_name, N_source, N_target,
            )

            # ---- Determine training-set sizes ---------------------------------
            if n_train_sizes_raw is not None:
                step_counts = np.array(sorted(int(x) for x in n_train_sizes_raw), dtype=np.int64)
            else:
                # Derive from island structure (cumulative island counts,
                # largest island first – same steps as Shapley add-curve)
                n_per_island = np.array(
                    [(locality_source == c).sum() for c in source_codes],
                    dtype=np.int64,
                )
                step_counts = np.cumsum(np.sort(n_per_island)[::-1])

            step_counts = np.clip(step_counts, 1, N_source)

            if assigned_n_train is not None:
                step_counts = np.array([x for x in step_counts if int(x) == int(assigned_n_train)], dtype=np.int64)
                if len(step_counts) == 0:
                    logger.info(
                        "Target %s (%s): assigned n_train_size=%d unavailable after clipping to n_source=%d; skipping target for this shard.",
                        target_code,
                        target_name,
                        int(assigned_n_train),
                        int(N_source),
                    )
                    continue

            logger.info("Training-set sizes: %s", step_counts.tolist())

            # ---- Precompute kernel once (candidate ∪ target) ------------------
            snp_cols = fixed_snp_cols
            if snp_cols is not None:
                X_cand_sel = X_source[:, snp_cols]
                X_tgt_sel = X_target[:, snp_cols]
            else:
                X_cand_sel = X_source
                X_tgt_sel = X_target

            X_all = np.vstack([X_cand_sel, X_tgt_sel])
            kernel_K, diag_K = build_kernel(X_all)

            cand_idx = np.arange(N_source, dtype=np.int64)
            target_idx = np.arange(N_source, N_source + N_target, dtype=np.int64)

            repeat_indices = [assigned_repeat_idx] if assigned_repeat_idx is not None else list(range(n_repeats))
            for repeat_idx in repeat_indices:
                repeat_seed = _make_repeat_seed(global_seed, int(target_code), int(repeat_idx))

                done_jobs += 1
                logger.info(
                    "PEVmean-GA %d/%d | trait=%s target=%s repeat=%d/%d",
                    done_jobs, total_jobs, trait_name, target_code,
                    repeat_idx + 1, n_repeats,
                )

                # ---- Always compare against training on all individuals ------
                # This does not need GA: subset is the full candidate pool.
                if write_all_individuals_baseline:
                    full_subset = cand_idx
                    full_pev = float(pev_mean(kernel_K, diag_K, full_subset, target_idx, lam=ridge_alpha))
                    full_eval = _evaluate_ridge_subset(
                        train_idx=full_subset,
                        X_source=X_source,
                        y_source=y_source,
                        X_test=X_target,
                        y_test=y_target,
                        y_eval_test=y_eval_target,
                        alpha=ridge_alpha,
                        snp_cols=snp_cols,
                    )

                    # For all individuals, the PEVmean and random baselines are
                    # identical by definition, so write matching rows for both.
                    all_pev_row = {
                        "n_individuals": int(N_source),
                        "corr_eval": float(full_eval["corr_eval"]),
                        "mse_adj": float(full_eval["mse_adj"]),
                        "method": "pevmean_ga",
                        "order_seed": -2,
                        "pevmean_obj": float(full_pev),
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "trait": trait_name,
                    }
                    all_rand_row = {
                        "n_individuals": int(N_source),
                        "corr_eval": float(full_eval["corr_eval"]),
                        "mse_adj": float(full_eval["mse_adj"]),
                        "method": "random_individual",
                        "order_seed": -2,
                        "pevmean_obj": float(full_pev),
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "trait": trait_name,
                    }
                    _append_csv(pd.DataFrame([all_pev_row, all_rand_row]), result_rows_path)

                # ---- PEVmean-GA for each training-set size --------------------
                for step_i, n_train in enumerate(step_counts):
                    n_train = int(n_train)

                    # All-individual case is handled explicitly above.
                    if n_train >= N_source:
                        continue

                    if n_train < 2:
                        row = {
                            "n_individuals": n_train,
                            "corr_eval": 0.0,
                            "mse_adj": float("inf"),
                            "method": "pevmean_ga",
                            "order_seed": -1,
                            "pevmean_obj": float("inf"),
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "trait": trait_name,
                        }
                        _append_csv(pd.DataFrame([row]), result_rows_path)
                        continue

                    step_ga_cfg = copy.deepcopy(ga_cfg)
                    step_ga_cfg.seed = repeat_seed + step_i * 7919

                    def fitness_fn(subset: np.ndarray) -> float:
                        return pev_mean(kernel_K, diag_K, subset, target_idx, lam=ridge_alpha)

                    best_subset, best_pev, ga_stats = run_ga(
                        n_candidates=N_source,
                        n_train=n_train,
                        fitness_fn=fitness_fn,
                        cfg=step_ga_cfg,
                        candidate_indices=cand_idx,
                    )

                    eval_result = _evaluate_ridge_subset(
                        train_idx=best_subset,
                        X_source=X_source,
                        y_source=y_source,
                        X_test=X_target,
                        y_test=y_target,
                        y_eval_test=y_eval_target,
                        alpha=ridge_alpha,
                        snp_cols=snp_cols,
                    )

                    row = {
                        "n_individuals": n_train,
                        "corr_eval": eval_result["corr_eval"],
                        "mse_adj": eval_result["mse_adj"],
                        "method": "pevmean_ga",
                        "order_seed": -1,
                        "pevmean_obj": float(best_pev),
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "trait": trait_name,
                    }
                    _append_csv(pd.DataFrame([row]), result_rows_path)

                    logger.info(
                        "  step %d/%d n_train=%d PEVmean=%.6f corr=%.4f",
                        step_i + 1, len(step_counts), n_train,
                        best_pev, eval_result["corr_eval"],
                    )

                # ---- Random baseline (same sizes) -----------------------------
                for r_order in range(n_random_orders):
                    ind_rng = np.random.default_rng(repeat_seed + 500_000 + r_order)
                    shuffled = ind_rng.permutation(N_source)

                    for step_i, n_train in enumerate(step_counts):
                        n_train = int(n_train)
                        chosen = shuffled[:n_train]

                        eval_result = _evaluate_ridge_subset(
                            train_idx=chosen,
                            X_source=X_source,
                            y_source=y_source,
                            X_test=X_target,
                            y_test=y_target,
                            y_eval_test=y_eval_target,
                            alpha=ridge_alpha,
                            snp_cols=snp_cols,
                        )

                        row = {
                            "n_individuals": n_train,
                            "corr_eval": eval_result["corr_eval"],
                            "mse_adj": eval_result["mse_adj"],
                            "method": "random_individual",
                            "order_seed": r_order,
                            "pevmean_obj": float("nan"),
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "trait": trait_name,
                        }
                        _append_csv(pd.DataFrame([row]), result_rows_path)

                    logger.info("  random baseline order %d done", r_order)

        # ---- Summary ----------------------------------------------------------
        if result_rows_path.exists():
            all_results = pd.read_csv(result_rows_path)
            summary = (
                all_results.groupby(
                    ["trait", "target_island", "method", "n_individuals"],
                    as_index=False,
                )
                .agg(
                    corr_mean=("corr_eval", "mean"),
                    corr_std=("corr_eval", "std"),
                    mse_mean=("mse_adj", "mean"),
                    n_rows=("corr_eval", "size"),
                )
            )
            summary_path = trait_output / "pevmean_ga_summary.csv"
            summary.to_csv(summary_path, index=False)
            logger.info("Summary saved to %s", summary_path)

        logger.info(
            "PEVmean-GA experiment complete for trait '%s'. Output: %s",
            trait_name, trait_output,
        )


if __name__ == "__main__":
    main()
