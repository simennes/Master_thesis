#!/usr/bin/env python
"""
Delta-LD island-ranked training-set selection for ridge.

Idea
----
For each target island, compute a Delta-LD distance to every candidate source
island (lower = more similar), rank source islands by that distance, and add
training data island-by-island in ranked order.

This script produces outputs compatible with the existing visualization
notebook structure (same core columns: target_island, method, n_individuals,
corr_eval, ...).

Usage
-----
Worker:
  python -m scripts.run_delta_ld_rank_weight --mode worker --config config/delta_ld_rank_weight_config.json

Merge shards:
  python -m scripts.run_delta_ld_rank_weight --mode merge --config config/delta_ld_rank_weight_config.json
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.training_set_optimization.runner import _evaluate_ridge_subset
from src.utils import set_seed

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


def _resolve_training_islands(
    training_islands_config: Optional[List[Any]],
    code_to_label: dict,
    present_codes: set,
    target_code: int,
) -> Optional[List[int]]:
    if training_islands_config is None:
        return None
    result = []
    for island_ref in training_islands_config:
        code = resolve_island_code(island_ref, code_to_label, present_codes)
        if code != target_code:
            result.append(code)
    return result if result else None


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


def _assign_jobs_weighted(jobs: List[Dict[str, Any]], num_shards: int) -> List[List[Dict[str, Any]]]:
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


def _make_repeat_seed(global_seed: int, target_code: int, repeat_idx: int) -> int:
    token = f"delta_ld|{int(global_seed)}|{int(target_code)}|{int(repeat_idx)}"
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % 2_147_483_647


def _choose_ld_snp_pool(
    total_snps: int,
    rng: np.random.Generator,
    *,
    base_pool: Optional[np.ndarray],
    max_ld_snps: int,
) -> np.ndarray:
    if base_pool is None:
        pool = np.arange(total_snps, dtype=np.int64)
    else:
        pool = np.asarray(base_pool, dtype=np.int64)

    if max_ld_snps > 0 and len(pool) > max_ld_snps:
        pool = np.sort(rng.choice(pool, size=int(max_ld_snps), replace=False).astype(np.int64))

    if len(pool) < 2:
        raise ValueError("Delta-LD requires at least 2 SNPs in the LD pool")
    return pool


def _sample_ld_pairs(
    n_pool: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_pool < 2:
        raise ValueError("Need at least two SNPs to sample LD pairs")
    if n_pairs < 1:
        raise ValueError("ld_num_pairs must be >= 1")

    a = rng.integers(0, n_pool, size=n_pairs, endpoint=False)
    b = rng.integers(0, n_pool - 1, size=n_pairs, endpoint=False)
    b = b + (b >= a)
    return a.astype(np.int64), b.astype(np.int64)


def _corr_for_pairs(X_sub: np.ndarray, a_idx: np.ndarray, b_idx: np.ndarray) -> np.ndarray:
    if X_sub.shape[0] < 3:
        return np.zeros(len(a_idx), dtype=np.float64)

    xa = X_sub[:, a_idx].astype(np.float64, copy=False)
    xb = X_sub[:, b_idx].astype(np.float64, copy=False)

    xa = xa - xa.mean(axis=0, keepdims=True)
    xb = xb - xb.mean(axis=0, keepdims=True)

    sa = xa.std(axis=0, ddof=1)
    sb = xb.std(axis=0, ddof=1)
    denom = sa * sb

    num = np.sum(xa * xb, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = num / np.maximum((X_sub.shape[0] - 1) * denom, 1e-12)

    corr = np.where(np.isfinite(corr), corr, 0.0)
    corr = np.clip(corr, -1.0, 1.0)
    return corr.astype(np.float64)


def _compute_island_ld_signatures(
    X: np.ndarray,
    locality: np.ndarray,
    island_codes: List[int],
    *,
    snp_pool: np.ndarray,
    pair_a_pool_idx: np.ndarray,
    pair_b_pool_idx: np.ndarray,
) -> Dict[int, np.ndarray]:
    pool_a = snp_pool[pair_a_pool_idx]
    pool_b = snp_pool[pair_b_pool_idx]
    signatures: Dict[int, np.ndarray] = {}
    for code in island_codes:
        mask = locality == int(code)
        Xi = X[mask]
        signatures[int(code)] = _corr_for_pairs(Xi, pool_a, pool_b)
    return signatures


def _delta_ld(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    d = sig_a - sig_b
    return float(np.sqrt(np.mean(d * d)))


def run_merge(config_path: Path) -> None:
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    exp_cfg = cfg.get("delta_ld_experiment", {})
    output_dir = Path(exp_cfg.get("output_dir", cfg["paths"].get("output_dir", "outputs/delta_ld_rank_weight")))
    trait_specs = _build_trait_specs(cfg)
    shards_root = output_dir / "shards"

    if not shards_root.exists():
        raise FileNotFoundError(f"No shard directory found: {shards_root}")

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        trait_output = output_dir / trait_name if len(trait_specs) > 1 else output_dir
        trait_output.mkdir(parents=True, exist_ok=True)

        merge_specs = [
            ("delta_ld_rank_weight_results.csv", True),
            ("delta_ld_ranked_selected_individuals.csv", False),
            ("delta_ld_island_distances.csv", False),
        ]

        merged_results: Optional[pd.DataFrame] = None

        for filename, is_results in merge_specs:
            parts: List[pd.DataFrame] = []
            for csv_path in sorted(shards_root.glob(f"shard_*/{trait_name}/{filename}")):
                if csv_path.exists():
                    parts.append(pd.read_csv(csv_path))

            if len(parts) == 0:
                logger.warning("No shard files found for trait '%s': %s", trait_name, filename)
                continue

            merged = pd.concat(parts, ignore_index=True)
            out_path = trait_output / filename
            merged.to_csv(out_path, index=False)
            logger.info("Merged %d shards into %s (%d rows)", len(parts), out_path, len(merged))

            if is_results:
                merged_results = merged

        if merged_results is not None and len(merged_results) > 0:
            summary = (
                merged_results.groupby(
                    ["trait", "target_island", "target_island_name", "analysis", "method", "n_individuals"],
                    as_index=False,
                )
                .agg(
                    corr_mean=("corr_eval", "mean"),
                    corr_std=("corr_eval", "std"),
                    mse_mean=("mse_adj", "mean"),
                    n_rows=("corr_eval", "size"),
                )
            )
            summary_csv = trait_output / "delta_ld_rank_weight_summary.csv"
            summary.to_csv(summary_csv, index=False)
            logger.info("Wrote summary for trait '%s'", trait_name)

    logger.info("Merge complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Delta-LD island-ranked ridge analysis")
    parser.add_argument("--mode", choices=["worker", "merge"], default="worker")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--target_islands", nargs="+", default=None)
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output CSV files instead of recreating them.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    import json
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.mode == "merge":
        run_merge(config_path)
        return

    exp_cfg = cfg.get("delta_ld_experiment", {})
    output_dir = Path(exp_cfg.get("output_dir", cfg["paths"].get("output_dir", "outputs/delta_ld_rank_weight")))
    output_dir.mkdir(parents=True, exist_ok=True)

    global_seed = int(cfg.get("seed", 42))
    set_seed(global_seed)

    model_cfg = cfg.get("model", {})
    ridge_alpha = float(model_cfg.get("alpha", 1e5))

    n_repeats = int(exp_cfg.get("n_repeats", 1))
    if n_repeats < 1:
        raise ValueError("delta_ld_experiment.n_repeats must be >= 1")

    n_random_reps = int(exp_cfg.get("n_random_reps", cfg.get("baselines", {}).get("n_random_orders", 0)))
    if n_random_reps < 0:
        raise ValueError("delta_ld_experiment.n_random_reps must be >= 0")

    training_islands_raw = exp_cfg.get("training_islands", None)
    max_ld_snps = int(exp_cfg.get("ld_num_snps", 1024))
    ld_num_pairs = int(exp_cfg.get("ld_num_pairs", 5000))

    use_snp_selection = bool(cfg.get("use_snp_selection", False))
    num_snps = cfg.get("num_snps", None)

    trait_specs = _build_trait_specs(cfg)

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

        present_codes = set(int(c) for c in np.unique(locality))
        included_raw = cfg.get("included_islands", None)
        if included_raw is not None:
            included_island_codes = [resolve_island_code(v, code_to_label, present_codes) for v in included_raw]
        else:
            included_island_codes = sorted(present_codes)

        raw_targets = args.target_islands if args.target_islands else cfg.get("target_islands", [0])
        target_codes = [resolve_island_code(t, code_to_label, present_codes) for t in raw_targets]

        fixed_snp_cols: Optional[np.ndarray] = None
        if use_snp_selection and num_snps is not None and int(num_snps) < X.shape[1]:
            rng_snp = np.random.default_rng(global_seed)
            fixed_snp_cols = np.sort(rng_snp.choice(X.shape[1], size=int(num_snps), replace=False).astype(np.int64))
            logger.info("Fixed SNP subset: n=%d", len(fixed_snp_cols))

        # Build one global LD fingerprint definition for this trait.
        rng_ld = np.random.default_rng(global_seed + 91_337)
        ld_snp_pool = _choose_ld_snp_pool(
            total_snps=int(X.shape[1]),
            rng=rng_ld,
            base_pool=fixed_snp_cols,
            max_ld_snps=max_ld_snps,
        )
        pair_a_pool_idx, pair_b_pool_idx = _sample_ld_pairs(
            n_pool=int(len(ld_snp_pool)),
            n_pairs=ld_num_pairs,
            rng=rng_ld,
        )

        ld_signatures = _compute_island_ld_signatures(
            X=X,
            locality=locality,
            island_codes=included_island_codes,
            snp_pool=ld_snp_pool,
            pair_a_pool_idx=pair_a_pool_idx,
            pair_b_pool_idx=pair_b_pool_idx,
        )
        logger.info(
            "Delta-LD signatures ready: %d islands, %d SNPs in pool, %d SNP pairs",
            len(ld_signatures),
            len(ld_snp_pool),
            len(pair_a_pool_idx),
        )

        results_path = trait_output / "delta_ld_rank_weight_results.csv"
        selected_path = trait_output / "delta_ld_ranked_selected_individuals.csv"
        island_dist_path = trait_output / "delta_ld_island_distances.csv"
        if not args.append:
            for p in [results_path, selected_path, island_dist_path]:
                if p.exists():
                    p.unlink()
        else:
            logger.info("Append mode enabled: preserving existing CSV outputs")

        jobs: List[Dict[str, Any]] = []
        source_codes_by_target: Dict[int, List[int]] = {}

        for target_code in target_codes:
            source_codes = [c for c in included_island_codes if c != target_code]
            if training_islands_raw is not None:
                training_codes = _resolve_training_islands(
                    training_islands_raw, code_to_label, present_codes, target_code
                )
                if training_codes is not None:
                    source_codes = [c for c in source_codes if c in training_codes]
            if len(source_codes) == 0:
                continue

            source_codes_by_target[int(target_code)] = list(source_codes)
            n_source = int(np.sum((locality != target_code) & np.isin(locality, source_codes)))
            if n_source < 2:
                continue

            # Work weight: full + per-island cumulative fits + random baselines on same steps.
            n_steps = len(source_codes)
            job_weight = float(max(1, n_source) * (1 + n_steps * (1 + n_random_reps)))
            for repeat_idx in range(n_repeats):
                jobs.append({
                    "target_code": int(target_code),
                    "repeat_idx": int(repeat_idx),
                    "weight": job_weight,
                })

        shard_bins = _assign_jobs_weighted(jobs, num_shards)
        shard_jobs = shard_bins[shard_index] if num_shards > 1 else jobs
        assigned = {(int(j["target_code"]), int(j["repeat_idx"])) for j in shard_jobs}

        logger.info(
            "Shard %d/%d assigned %d/%d jobs",
            shard_index,
            num_shards,
            len(shard_jobs),
            len(jobs),
        )

        done = 0
        total = len(shard_jobs)

        for target_code in target_codes:
            if int(target_code) not in source_codes_by_target:
                continue

            target_name = island_label(target_code, code_to_label)
            source_codes = source_codes_by_target[int(target_code)]

            target_mask = locality == target_code
            source_mask = (~target_mask) & np.isin(locality, source_codes)

            X_source = X[source_mask]
            y_source = y[source_mask]
            ids_source = ids[source_mask]
            locality_source = locality[source_mask]

            X_target = X[target_mask]
            y_target = y[target_mask]
            y_eval_target = y_eval[target_mask]

            N_source = len(X_source)
            if N_source < 2 or len(X_target) == 0:
                continue

            # Compute island-to-target Delta-LD and ranked source-island order.
            target_sig = ld_signatures[int(target_code)]
            island_rows: List[Dict[str, Any]] = []
            for src_code in source_codes:
                src_sig = ld_signatures[int(src_code)]
                delta = _delta_ld(target_sig, src_sig)
                n_src_island = int(np.sum(locality_source == int(src_code)))
                island_rows.append({
                    "source_island": int(src_code),
                    "source_island_name": island_label(int(src_code), code_to_label),
                    "delta_ld": float(delta),
                    "n_source_island": int(n_src_island),
                })

            island_df = pd.DataFrame(island_rows).sort_values(
                ["delta_ld", "source_island"],
                kind="mergesort",
            ).reset_index(drop=True)
            island_df["island_rank"] = np.arange(1, len(island_df) + 1, dtype=np.int64)

            island_df_out = island_df.copy()
            island_df_out["trait"] = trait_name
            island_df_out["target_island"] = int(target_code)
            island_df_out["target_island_name"] = str(target_name)
            _append_csv(island_df_out, island_dist_path)

            ordered_islands = island_df["source_island"].astype(int).tolist()

            # Build cumulative individual order by ranked islands.
            idx_blocks: List[np.ndarray] = []
            n_steps: List[int] = []
            island_rank_at_step: List[int] = []
            delta_ld_at_step: List[float] = []
            running = 0

            for rank_idx, src_code in enumerate(ordered_islands, start=1):
                block = np.flatnonzero(locality_source == int(src_code)).astype(np.int64)
                if len(block) == 0:
                    continue
                idx_blocks.append(block)
                running += int(len(block))
                n_steps.append(int(running))
                island_rank_at_step.append(int(rank_idx))
                delta_ld_at_step.append(float(island_df.loc[rank_idx - 1, "delta_ld"]))

            if len(idx_blocks) == 0:
                continue

            ordered_idx = np.concatenate(idx_blocks).astype(np.int64)
            step_counts = np.array(n_steps, dtype=np.int64)
            full_idx = np.arange(N_source, dtype=np.int64)

            for repeat_idx in range(n_repeats):
                if (int(target_code), int(repeat_idx)) not in assigned:
                    continue

                done += 1
                repeat_seed = _make_repeat_seed(global_seed, int(target_code), int(repeat_idx))
                logger.info(
                    "Job %d/%d | trait=%s target=%s repeat=%d/%d",
                    done,
                    total,
                    trait_name,
                    target_code,
                    repeat_idx + 1,
                    n_repeats,
                )

                # Full-source baseline
                full_eval = _evaluate_ridge_subset(
                    train_idx=full_idx,
                    X_source=X_source,
                    y_source=y_source,
                    X_test=X_target,
                    y_test=y_target,
                    y_eval_test=y_eval_target,
                    alpha=ridge_alpha,
                    snp_cols=fixed_snp_cols,
                )
                full_row = {
                    "analysis": "full_baseline",
                    "method": "full_source_unweighted",
                    "weight_scheme": "none",
                    "order_seed": -2,
                    "weighted_fit_used": False,
                    "n_individuals": int(N_source),
                    "corr_eval": float(full_eval["corr_eval"]),
                    "mse_adj": float(full_eval["mse_adj"]),
                    "delta_ld_obj": float("nan"),
                    "target_island": int(target_code),
                    "target_island_name": str(target_name),
                    "repeat": int(repeat_idx),
                    "repeat_seed": int(repeat_seed),
                    "trait": trait_name,
                }
                _append_csv(pd.DataFrame([full_row]), results_path)

                # Ranked island-by-island inclusion
                for step_i, n_train in enumerate(step_counts):
                    chosen = ordered_idx[: int(n_train)]
                    eval_result = _evaluate_ridge_subset(
                        train_idx=chosen,
                        X_source=X_source,
                        y_source=y_source,
                        X_test=X_target,
                        y_test=y_target,
                        y_eval_test=y_eval_target,
                        alpha=ridge_alpha,
                        snp_cols=fixed_snp_cols,
                    )

                    row = {
                        "analysis": "ranked_subset",
                        "method": "delta_ld_island_topk",
                        "weight_scheme": "none",
                        "order_seed": -1,
                        "weighted_fit_used": False,
                        "n_individuals": int(n_train),
                        "corr_eval": float(eval_result["corr_eval"]),
                        "mse_adj": float(eval_result["mse_adj"]),
                        "delta_ld_obj": float(delta_ld_at_step[step_i]),
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "trait": trait_name,
                    }
                    _append_csv(pd.DataFrame([row]), results_path)

                    chosen_island_ranks = np.full(len(chosen), island_rank_at_step[step_i], dtype=np.int64)
                    selected_df = pd.DataFrame({
                        "trait": trait_name,
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "n_train_size": int(n_train),
                        "method": "delta_ld_island_topk",
                        "ringnr": ids_source[chosen],
                        "ringnumber": ids_source[chosen],
                        "source_island": locality_source[chosen].astype(int),
                        "source_island_name": [island_label(int(c), code_to_label) for c in locality_source[chosen]],
                        "island_rank_inclusion": chosen_island_ranks,
                    })
                    _append_csv(selected_df, selected_path)

                # Optional random baseline over the same island-cumulative step sizes.
                for order_seed in range(n_random_reps):
                    rng = np.random.default_rng(repeat_seed + 500_000 + order_seed)
                    shuffled = rng.permutation(N_source)
                    for n_train in step_counts:
                        chosen = shuffled[: int(n_train)]
                        eval_result = _evaluate_ridge_subset(
                            train_idx=chosen,
                            X_source=X_source,
                            y_source=y_source,
                            X_test=X_target,
                            y_test=y_target,
                            y_eval_test=y_eval_target,
                            alpha=ridge_alpha,
                            snp_cols=fixed_snp_cols,
                        )
                        rand_row = {
                            "analysis": "ranked_subset",
                            "method": "random_individual",
                            "weight_scheme": "none",
                            "order_seed": int(order_seed),
                            "weighted_fit_used": False,
                            "n_individuals": int(n_train),
                            "corr_eval": float(eval_result["corr_eval"]),
                            "mse_adj": float(eval_result["mse_adj"]),
                            "delta_ld_obj": float("nan"),
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "trait": trait_name,
                        }
                        _append_csv(pd.DataFrame([rand_row]), results_path)

        if results_path.exists():
            all_results = pd.read_csv(results_path)
            summary = (
                all_results.groupby(
                    ["trait", "target_island", "target_island_name", "analysis", "method", "n_individuals"],
                    as_index=False,
                )
                .agg(
                    corr_mean=("corr_eval", "mean"),
                    corr_std=("corr_eval", "std"),
                    mse_mean=("mse_adj", "mean"),
                    n_rows=("corr_eval", "size"),
                )
            )
            summary_csv = trait_output / "delta_ld_rank_weight_summary.csv"
            summary.to_csv(summary_csv, index=False)

        logger.info("Trait '%s' complete. Output: %s", trait_name, trait_output)


if __name__ == "__main__":
    main()
