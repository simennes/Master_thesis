#!/usr/bin/env python
"""
Run PEVmean-GA training-set optimisation.

For each target island, select training sets of given sizes from the
candidate pool (all non-target islands) by minimising PEVmean — a
genotype-only criterion.  The script writes the selected individuals and
objective values only; downstream prediction models consume those subsets
in separate experiments.

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
with PEVmean objective and GA convergence metadata.

``pevmean_ga_selected_individuals.csv`` – one row per selected individual
for each PEVmean-GA-optimised subset (ringnumber + source island origin).
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
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.training_set_optimization.ga_subset import GAConfig, run_ga
from src.training_set_optimization.pevmean import (
    paper_style_lambda,
    pev_mean,
    pev_mean_batch,
    sum_pc_variances,
)
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


def _compute_dataset_pcs(
    X: np.ndarray,
    n_pcs: int,
    seed: int,
    standardize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit PCA on the full filtered X once and return PC scores + var ratio.

    Safe because PCA uses genotypes only -- no phenotype leakage from the
    target island. PCs are nested, so callers can slice Z[:, :r] for any
    r <= the number of fitted components.
    """
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    feasible_r = int(min(int(n_pcs), n_samples, n_features))
    if feasible_r < 1:
        raise ValueError(
            f"Cannot fit PCA: feasible_r={feasible_r} (n={n_samples}, p={n_features}, n_pcs={n_pcs})."
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
        pca.explained_variance_ratio_.astype(np.float32, copy=False),
        dt,
    )


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


def _assign_jobs_weighted(
    jobs: List[Dict[str, Any]],
    num_shards: int,
) -> List[List[Dict[str, Any]]]:
    """Greedy weighted partition of jobs across shards (LPT heuristic)."""
    if num_shards <= 1:
        return [jobs]

    bins: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
    loads = np.zeros(num_shards, dtype=np.float64)

    sorted_jobs = sorted(jobs, key=lambda j: float(j.get("weight", 1.0)), reverse=True)
    for job in sorted_jobs:
        tgt = int(np.argmin(loads))
        bins[tgt].append(job)
        loads[tgt] += float(job.get("weight", 1.0))

    return bins


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
                pevmean_mean=("pevmean_obj", "mean"),
                pevmean_std=("pevmean_obj", "std"),
                ga_elapsed_sum=("ga_elapsed_sec", "sum"),
                n_rows=("pevmean_obj", "size"),
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

        sel_parts: List[pd.DataFrame] = []
        for sel_path in sorted(
            shards_root.glob(f"shard_*/{trait_name}/pevmean_ga_selected_individuals.csv")
        ):
            if sel_path.exists():
                sel_parts.append(pd.read_csv(sel_path))

        if len(sel_parts) > 0:
            sel_merged = pd.concat(sel_parts, ignore_index=True)
            sel_out = trait_output / "pevmean_ga_selected_individuals.csv"
            sel_merged.to_csv(sel_out, index=False)
            logger.info(
                "Merged %d selection files for trait '%s' into %s (%d rows)",
                len(sel_parts),
                trait_name,
                sel_out,
                len(sel_merged),
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

    # ---- PEVmean shrinkage + PC count (marker-form objective) ----------------
    # See `src/training_set_optimization/pevmean.py` for the convention.
    # The PEVmean objective uses PC scores as features; no kernel is built.
    pevmean_cfg = cfg.get("pevmean", {})
    pevmean_n_pcs = int(pevmean_cfg.get("n_pcs", 100))
    pevmean_lambda_mode = str(pevmean_cfg.get("lambda_mode", "paper")).strip().lower()
    if pevmean_lambda_mode not in {"paper", "fixed"}:
        raise ValueError("pevmean.lambda_mode must be 'paper' or 'fixed'.")
    pevmean_lambda_fixed = pevmean_cfg.get("lambda_fixed")
    if pevmean_lambda_fixed is not None:
        pevmean_lambda_fixed = float(pevmean_lambda_fixed)
    pevmean_va_apriori = pevmean_cfg.get("va_apriori")
    if pevmean_va_apriori is not None:
        pevmean_va_apriori = float(pevmean_va_apriori)
    pevmean_sigma_e2_apriori = pevmean_cfg.get("sigma_e2_apriori")
    if pevmean_sigma_e2_apriori is not None:
        pevmean_sigma_e2_apriori = float(pevmean_sigma_e2_apriori)
    pca_seed = int(pevmean_cfg.get("pca_seed", cfg.get("seed", 14)))
    pca_standardize = bool(pevmean_cfg.get("standardize_for_pca", False))

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
    baselines_cfg = cfg.get("baselines", {})
    include_full_source_baseline = bool(baselines_cfg.get("include_full_source", True))
    n_train_sizes_raw = ga_raw.get("n_train_sizes", None)

    # ---- Parallelism (process-level fitness evaluation) ----------------------
    n_jobs = int(ga_raw.get("n_jobs", 1))
    parallel_min_n_train = int(ga_raw.get("parallel_min_n_train", 1000))
    if n_jobs > 1:
        logger.info(
            "Parallel fitness eval enabled: n_jobs=%d, min_n_train=%d",
            n_jobs, parallel_min_n_train,
        )

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

    if num_shards > 1:
        logger.info(
            "Shard %d/%d enabled; using weighted job distribution across (target, repeat, n_train).",
            shard_index,
            num_shards,
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

        # Restrict PCA to included islands so Z aligns with the source/target masks.
        included_mask = np.isin(locality, included_island_codes)
        if not np.all(included_mask):
            X = X[included_mask]
            y = y[included_mask]
            y_eval = y_eval[included_mask]
            ids = ids[included_mask]
            locality = locality[included_mask]

        # Fit PCA only for the PEVmean objective. Downstream models consume the
        # selected subsets later and choose their own feature representation.
        Z_full, var_ratio, pca_seconds = _compute_dataset_pcs(
            X=X, n_pcs=pevmean_n_pcs, seed=pca_seed, standardize=pca_standardize,
        )
        cumvar_pev = float(np.sum(var_ratio[:pevmean_n_pcs]))
        logger.info(
            "Trait=%s PCA fit on included islands (n=%d, p=%d, r=%d) in %.2fs "
            "(cumvar at PEVmean r=%d: %.3f)",
            trait_name, X.shape[0], X.shape[1], int(Z_full.shape[1]),
            pca_seconds, pevmean_n_pcs, cumvar_pev,
        )

        # Build PEVmean features and resolve trait-specific lambda.
        Z_pev = np.ascontiguousarray(Z_full[:, :pevmean_n_pcs], dtype=np.float64)
        if pevmean_lambda_mode == "paper":
            va = pevmean_va_apriori
            if va is None:
                raise ValueError(
                    f"pevmean.lambda_mode='paper' requires pevmean.va_apriori for trait '{trait_name}'."
                )
            va = float(va)
            sigma_e2_eff = pevmean_sigma_e2_apriori
            if sigma_e2_eff is None:
                sigma_e2_eff = float(np.var(np.asarray(y, dtype=np.float64), ddof=1))
            sum_var_pc = sum_pc_variances(Z_pev)
            pevmean_lambda = paper_style_lambda(sigma_e2_eff, va, sum_var_pc)
            logger.info(
                "Trait=%s paper-mode lambda: sigma_e^2=%.4f * sum_var_PC=%.2f / va=%.3f = %.3e",
                trait_name, sigma_e2_eff, sum_var_pc, va, pevmean_lambda,
            )
        else:
            if pevmean_lambda_fixed is None:
                raise ValueError(
                    "pevmean.lambda_mode='fixed' requires pevmean.lambda_fixed."
                )
            pevmean_lambda = float(pevmean_lambda_fixed)
            logger.info("Trait=%s fixed-mode lambda: %.3e", trait_name, pevmean_lambda)

        result_rows_path = trait_output / "pevmean_ga_results.csv"
        if result_rows_path.exists():
            result_rows_path.unlink()
        selected_rows_path = trait_output / "pevmean_ga_selected_individuals.csv"
        if selected_rows_path.exists():
            selected_rows_path.unlink()

        step_counts_by_target: Dict[int, np.ndarray] = {}
        jobs: List[Dict[str, Any]] = []

        for target_code in target_codes:
            source_codes = [c for c in included_island_codes if c != target_code]
            if len(source_codes) == 0:
                continue

            target_mask = (locality == target_code)
            source_mask = (~target_mask) & np.isin(locality, source_codes)
            locality_source = locality[source_mask]
            n_source = int(np.sum(source_mask))

            if n_train_sizes_raw is not None:
                step_counts = np.array(sorted(int(x) for x in n_train_sizes_raw), dtype=np.int64)
            else:
                n_per_island = np.array(
                    [(locality_source == c).sum() for c in source_codes],
                    dtype=np.int64,
                )
                step_counts = np.cumsum(np.sort(n_per_island)[::-1])

            step_counts = np.clip(step_counts, 1, n_source)
            step_counts_by_target[int(target_code)] = step_counts

            for repeat_idx in range(n_repeats):
                if include_full_source_baseline:
                    jobs.append({
                        "kind": "full",
                        "target_code": int(target_code),
                        "repeat_idx": int(repeat_idx),
                        "n_train": None,
                        "weight": 1.0,
                    })
                for n_train in step_counts:
                    n_train = int(n_train)
                    if n_train >= n_source:
                        continue
                    jobs.append({
                        "kind": "size",
                        "target_code": int(target_code),
                        "repeat_idx": int(repeat_idx),
                        "n_train": int(n_train),
                        "weight": float(max(1, n_train) ** 2),
                    })

        shard_bins = _assign_jobs_weighted(jobs, num_shards)
        shard_jobs = shard_bins[shard_index] if num_shards > 1 else jobs

        assigned_full = {
            (int(j["target_code"]), int(j["repeat_idx"]))
            for j in shard_jobs
            if j["kind"] == "full"
        }
        assigned_sizes: Dict[Tuple[int, int], set[int]] = {}
        for job in shard_jobs:
            if job["kind"] != "size":
                continue
            key = (int(job["target_code"]), int(job["repeat_idx"]))
            assigned_sizes.setdefault(key, set()).add(int(job["n_train"]))

        logger.info(
            "Shard %d/%d assigned %d/%d jobs (full=%d, size=%d)",
            shard_index,
            num_shards,
            len(shard_jobs),
            len(jobs),
            len(assigned_full),
            sum(len(v) for v in assigned_sizes.values()),
        )

        total_jobs = len(shard_jobs)
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

            ids_source = ids[source_mask]
            locality_source = locality[source_mask]

            N_source = int(np.sum(source_mask))
            N_target = int(np.sum(target_mask))

            logger.info(
                "Target %s (%s): n_source=%d, n_target=%d",
                target_code, target_name, N_source, N_target,
            )

            step_counts = step_counts_by_target.get(int(target_code), np.array([], dtype=np.int64))
            if len(step_counts) == 0:
                logger.info("Target %s (%s): no valid step sizes; skipping", target_code, target_name)
                continue

            logger.info("Training-set sizes: %s", step_counts.tolist())

            # ---- Per-target indices into Z_pev for marker-form PEVmean -------
            # source_pev_idx maps GA candidate row -> row in Z_pev.
            source_pev_idx = np.flatnonzero(source_mask).astype(np.int64)
            target_idx = np.flatnonzero(target_mask).astype(np.int64)

            cand_idx = np.arange(N_source, dtype=np.int64)

            for repeat_idx in range(n_repeats):
                key = (int(target_code), int(repeat_idx))
                run_full_baseline = key in assigned_full
                assigned_sizes_for_key = sorted(assigned_sizes.get(key, set()))
                if (not run_full_baseline) and (len(assigned_sizes_for_key) == 0):
                    continue

                repeat_seed = _make_repeat_seed(global_seed, int(target_code), int(repeat_idx))

                done_jobs += int(run_full_baseline) + len(assigned_sizes_for_key)
                logger.info(
                    "PEVmean-GA job %d/%d | trait=%s target=%s repeat=%d/%d | full=%s sizes=%s",
                    done_jobs, total_jobs, trait_name, target_code,
                    repeat_idx + 1, n_repeats,
                    run_full_baseline,
                    assigned_sizes_for_key,
                )

                # ---- Always compare against training on all individuals ------
                # This does not need GA: subset is the full candidate pool.
                if run_full_baseline and include_full_source_baseline:
                    full_subset = cand_idx
                    full_pev = float(pev_mean(
                        Z_pev,
                        source_pev_idx[full_subset],
                        target_idx,
                        lam=pevmean_lambda,
                    ))
                    # Full-source PEV is kept as objective metadata when requested.
                    all_pev_row = {
                        "n_individuals": int(N_source),
                        "method": "pevmean_ga",
                        "order_seed": -2,
                        "pevmean_obj": float(full_pev),
                        "pevmean_n_pcs": int(pevmean_n_pcs),
                        "pevmean_lambda": float(pevmean_lambda),
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "trait": trait_name,
                        "ga_generations": 0,
                        "ga_cache_size": 0,
                        "ga_elapsed_sec": 0.0,
                    }
                    _append_csv(pd.DataFrame([all_pev_row]), result_rows_path)

                # ---- PEVmean-GA for each training-set size --------------------
                for step_i, n_train in enumerate(step_counts):
                    n_train = int(n_train)

                    if n_train not in assigned_sizes_for_key:
                        continue

                    # All-individual case is handled explicitly above.
                    if n_train >= N_source:
                        continue

                    if n_train < 2:
                        row = {
                            "n_individuals": n_train,
                            "method": "pevmean_ga",
                            "order_seed": -1,
                            "pevmean_obj": float("inf"),
                            "pevmean_n_pcs": int(pevmean_n_pcs),
                            "pevmean_lambda": float(pevmean_lambda),
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "trait": trait_name,
                            "ga_generations": 0,
                            "ga_cache_size": 0,
                            "ga_elapsed_sec": 0.0,
                        }
                        _append_csv(pd.DataFrame([row]), result_rows_path)
                        continue

                    step_ga_cfg = copy.deepcopy(ga_cfg)
                    step_ga_cfg.seed = repeat_seed + step_i * 7919

                    def fitness_fn(subset: np.ndarray) -> float:
                        return pev_mean(
                            Z_pev,
                            source_pev_idx[subset],
                            target_idx,
                            lam=pevmean_lambda,
                        )

                    if n_jobs > 1 and n_train >= parallel_min_n_train:
                        def batch_fitness_fn(subsets):
                            return pev_mean_batch(
                                Z_pev,
                                [source_pev_idx[s] for s in subsets],
                                target_idx,
                                pevmean_lambda,
                                n_jobs=n_jobs,
                            )
                    else:
                        batch_fitness_fn = None

                    best_subset, best_pev, ga_stats = run_ga(
                        n_candidates=N_source,
                        n_train=n_train,
                        fitness_fn=fitness_fn,
                        cfg=step_ga_cfg,
                        candidate_indices=cand_idx,
                        batch_fitness_fn=batch_fitness_fn,
                    )

                    row = {
                        "n_individuals": n_train,
                        "method": "pevmean_ga",
                        "order_seed": -1,
                        "pevmean_obj": float(best_pev),
                        "pevmean_n_pcs": int(pevmean_n_pcs),
                        "pevmean_lambda": float(pevmean_lambda),
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "trait": trait_name,
                        "ga_generations": int(ga_stats.get("generations_run", 0)),
                        "ga_cache_size": int(ga_stats.get("cache_size", 0)),
                        "ga_elapsed_sec": float(ga_stats.get("elapsed_sec", float("nan"))),
                    }
                    _append_csv(pd.DataFrame([row]), result_rows_path)

                    selected_codes = locality_source[best_subset].astype(int)
                    selected_names = [island_label(c, code_to_label) for c in selected_codes]
                    selected_df = pd.DataFrame({
                        "trait": trait_name,
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "n_train_size": int(n_train),
                        "method": "pevmean_ga",
                        "pevmean_obj": float(best_pev),
                        "pevmean_n_pcs": int(pevmean_n_pcs),
                        "pevmean_lambda": float(pevmean_lambda),
                        "ringnumber": ids_source[best_subset],
                        "source_island": selected_codes,
                        "source_island_name": selected_names,
                    })
                    _append_csv(selected_df, selected_rows_path)

                    logger.info(
                        "  step %d/%d n_train=%d PEVmean=%.6f",
                        step_i + 1, len(step_counts), n_train,
                        best_pev,
                    )

        # ---- Summary ----------------------------------------------------------
        if result_rows_path.exists():
            all_results = pd.read_csv(result_rows_path)
            summary = (
                all_results.groupby(
                    ["trait", "target_island", "method", "n_individuals"],
                    as_index=False,
                )
                .agg(
                    pevmean_mean=("pevmean_obj", "mean"),
                    pevmean_std=("pevmean_obj", "std"),
                    ga_elapsed_sum=("ga_elapsed_sec", "sum"),
                    n_rows=("pevmean_obj", "size"),
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
