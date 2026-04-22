#!/usr/bin/env python
"""
Run avg-GRM similarity experiments for training-set selection and weighting.

This mirrors the PEVmean-GA island-wise setup but replaces GA/PEVmean with
simple genotype-only scores:
  avg_grm_i = mean_j GRM[i, j] for target/test individuals j.

Per target island (and repeat):
  A) Ranked subset analysis:
     - rank source individuals by avg_grm_i (descending)
     - evaluate ridge on top-k subsets for configured k grid

  B) Weighted analysis:
     - use all source individuals
     - compute weights from avg_grm_i with several schemes
     - normalize weights to mean 1
     - fit weighted ridge and evaluate on target island

Usage
-----
Worker:
  python -m scripts.run_avggrm_rank_weight --mode worker --config config/avggrm_rank_weight_config.json

Merge shards:
  python -m scripts.run_avggrm_rank_weight --mode merge --config config/avggrm_rank_weight_config.json
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
from sklearn.linear_model import Ridge

from src.avggrm_weighting import weights_from_scheme as _shared_weights_from_scheme
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
    """
    Resolve training_islands config to a list of island codes.
    If None, returns None (meaning use all except target).
    If specified, returns the resolved list of training island codes.
    """
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
    token = f"avggrm|{int(global_seed)}|{int(target_code)}|{int(repeat_idx)}"
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % 2_147_483_647


def _safe_minmax(x: np.ndarray) -> np.ndarray:
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def _normalize_mean_one(w: np.ndarray, floor: float = 1e-6, clip_max: Optional[float] = None) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, float(floor))
    mean_w = float(np.mean(w))
    if mean_w <= 0:
        w = np.ones_like(w, dtype=float)
    else:
        w = w / mean_w
    if clip_max is not None:
        w = np.minimum(w, float(clip_max))
        w = w / max(float(np.mean(w)), 1e-12)
    return w


def _weights_from_scheme(
    avg_grm: np.ndarray,
    ranks: np.ndarray,
    scheme_cfg: Dict[str, Any],
) -> Tuple[str, np.ndarray]:
    name = str(scheme_cfg.get("name", "uniform")).lower()
    return name, _shared_weights_from_scheme(avg_grm, ranks, scheme_cfg)


def _parse_weight_schemes(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = cfg.get("weight_schemes", ["uniform", "linear", "minmax", "exponential", "top-heavy"])
    schemes: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            schemes.append({"name": item})
        elif isinstance(item, dict) and "name" in item:
            schemes.append(item)
        else:
            raise ValueError("weight_schemes must contain scheme names or objects with a 'name' field")
    if len(schemes) == 0:
        raise ValueError("At least one weight scheme is required")
    return schemes


def run_merge(config_path: Path) -> None:
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    exp_cfg = cfg.get("avggrm_experiment", {})
    output_dir = Path(exp_cfg.get("output_dir", cfg["paths"].get("output_dir", "outputs/avggrm_rank_weight")))
    trait_specs = _build_trait_specs(cfg)
    shards_root = output_dir / "shards"

    if not shards_root.exists():
        raise FileNotFoundError(f"No shard directory found: {shards_root}")

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        trait_output = output_dir / trait_name if len(trait_specs) > 1 else output_dir
        trait_output.mkdir(parents=True, exist_ok=True)

        merge_specs = [
            ("avggrm_rank_weight_results.csv", True),
            ("avggrm_ranked_selected_individuals.csv", False),
            ("avggrm_weighted_individuals.csv", False),
            ("avggrm_individual_diagnostics.csv", False),
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
            summary_csv = trait_output / "avggrm_rank_weight_summary.csv"
            summary.to_csv(summary_csv, index=False)
            logger.info("Wrote summary files for trait '%s'", trait_name)

    logger.info("Merge complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Avg-GRM ranked + weighted training-set analysis")
    parser.add_argument("--mode", choices=["worker", "merge"], default="worker")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--target_islands", nargs="+", default=None, help="Override target_islands from config")
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
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

    exp_cfg = cfg.get("avggrm_experiment", {})
    output_dir = Path(exp_cfg.get("output_dir", cfg["paths"].get("output_dir", "outputs/avggrm_rank_weight")))
    output_dir.mkdir(parents=True, exist_ok=True)

    global_seed = int(cfg.get("seed", 42))
    set_seed(global_seed)

    model_cfg = cfg.get("model", {})
    ridge_alpha = float(model_cfg.get("alpha", 1e5))

    n_repeats = int(exp_cfg.get("n_repeats", 1))
    if n_repeats < 1:
        raise ValueError("avggrm_experiment.n_repeats must be >= 1")
    n_random_reps = int(exp_cfg.get("n_random_reps", cfg.get("baselines", {}).get("n_random_orders", 5)))
    if n_random_reps < 0:
        raise ValueError("avggrm_experiment.n_random_reps must be >= 0")

    n_train_sizes_raw = exp_cfg.get("n_train_sizes", None)
    training_islands_raw = exp_cfg.get("training_islands", None)
    compute_individual_diagnostics = bool(exp_cfg.get("compute_individual_diagnostics", False))
    weight_schemes = _parse_weight_schemes(exp_cfg)

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

        if GRM_df is None:
            raise ValueError(
                "GRM is required for avg-GRM experiment. Provide paths.grm_rds in config/trait paths."
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

        results_path = trait_output / "avggrm_rank_weight_results.csv"
        selected_path = trait_output / "avggrm_ranked_selected_individuals.csv"
        weights_path = trait_output / "avggrm_weighted_individuals.csv"
        individual_diag_path = trait_output / "avggrm_individual_diagnostics.csv"
        for p in [results_path, selected_path, weights_path, individual_diag_path]:
            if p.exists():
                p.unlink()

        jobs: List[Dict[str, Any]] = []
        steps_by_target: Dict[int, np.ndarray] = {}

        for target_code in target_codes:
            source_codes = [c for c in included_island_codes if c != target_code]
            
            # Apply training_islands filter if specified
            if training_islands_raw is not None:
                training_codes = _resolve_training_islands(
                    training_islands_raw, code_to_label, present_codes, target_code
                )
                if training_codes is not None:
                    source_codes = [c for c in source_codes if c in training_codes]
            
            if len(source_codes) == 0:
                continue

            target_mask = (locality == target_code)
            source_mask = (~target_mask) & np.isin(locality, source_codes)
            n_source = int(np.sum(source_mask))
            if n_source < 2:
                continue

            if n_train_sizes_raw is not None:
                step_counts = np.array(sorted(int(x) for x in n_train_sizes_raw), dtype=np.int64)
            else:
                locality_source = locality[source_mask]
                n_per_island = np.array([(locality_source == c).sum() for c in source_codes], dtype=np.int64)
                step_counts = np.cumsum(np.sort(n_per_island)[::-1])

            step_counts = np.unique(np.clip(step_counts, 2, n_source))
            steps_by_target[int(target_code)] = step_counts

            for repeat_idx in range(n_repeats):
                jobs.append({
                    "target_code": int(target_code),
                    "repeat_idx": int(repeat_idx),
                    "weight": float(max(1, n_source) * (1 + len(step_counts) * (1 + n_random_reps) + len(weight_schemes))),
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
            target_name = island_label(target_code, code_to_label)
            source_codes = [c for c in included_island_codes if c != target_code]

            # Apply training_islands filter (must match the job-building loop)
            if training_islands_raw is not None:
                training_codes = _resolve_training_islands(
                    training_islands_raw, code_to_label, present_codes, target_code
                )
                if training_codes is not None:
                    source_codes = [c for c in source_codes if c in training_codes]

            if len(source_codes) == 0:
                continue

            target_mask = (locality == target_code)
            source_mask = (~target_mask) & np.isin(locality, source_codes)

            X_source = X[source_mask]
            y_source = y[source_mask]
            ids_source = ids[source_mask]
            locality_source = locality[source_mask]

            X_target = X[target_mask]
            y_target = y[target_mask]
            y_eval_target = y_eval[target_mask]
            ids_target = ids[target_mask]

            N_source = len(X_source)
            if N_source < 2 or len(X_target) == 0:
                continue

            # Compute avg-GRM using only genotype-based GRM similarities.
            grm_block = GRM_df.loc[ids_source, ids_target].to_numpy(dtype=float)
            avg_grm = np.asarray(grm_block.mean(axis=1), dtype=float)

            order = np.argsort(-avg_grm, kind="mergesort")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)

            step_counts = steps_by_target.get(int(target_code), np.array([], dtype=np.int64)).copy()
            
            # Add positive avg_grm threshold size
            n_positive = int(np.sum(avg_grm > 0))
            if n_positive >= 2 and n_positive not in step_counts:
                step_counts = np.unique(np.append(step_counts, [n_positive])).astype(np.int64)
                step_counts = np.sort(step_counts)
            
            if len(step_counts) == 0:
                continue

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

                individual_diag_parts: List[pd.DataFrame] = []

                def _build_individual_diagnostics(
                    train_idx: np.ndarray,
                    analysis: str,
                    method: str,
                    order_seed: int,
                    weight_scheme: str,
                    weighted_fit_used: bool,
                    sample_weight: Optional[np.ndarray] = None,
                ) -> pd.DataFrame:
                    X_train = X_source[train_idx]
                    y_train = y_source[train_idx]
                    X_test_fit = X_target

                    if fixed_snp_cols is not None:
                        X_train = X_train[:, fixed_snp_cols]
                        X_test_fit = X_test_fit[:, fixed_snp_cols]

                    model = Ridge(alpha=max(float(ridge_alpha), 1e-12))
                    if sample_weight is None:
                        model.fit(X_train, y_train)
                    else:
                        model.fit(X_train, y_train, sample_weight=np.asarray(sample_weight, dtype=float))

                    y_pred_all = model.predict(X_test_fit)
                    grm_tgt_block = GRM_df.loc[ids_source[train_idx], ids_target].to_numpy(dtype=float)
                    avg_grm_per_tgt = grm_tgt_block.mean(axis=0)

                    return pd.DataFrame({
                        "trait": trait_name,
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "analysis": analysis,
                        "method": method,
                        "weight_scheme": weight_scheme,
                        "order_seed": int(order_seed),
                        "weighted_fit_used": bool(weighted_fit_used),
                        "n_individuals": int(len(train_idx)),
                        "individual_id": ids_target,
                        "avg_grm_to_training_set": avg_grm_per_tgt.astype(float),
                        "pred_mse": ((y_target - y_pred_all) ** 2).astype(float),
                        "pred_mse_eval": ((y_eval_target - y_pred_all) ** 2).astype(float),
                        "y_true": y_target.astype(float),
                        "y_true_eval": y_eval_target.astype(float),
                        "y_pred": y_pred_all.astype(float),
                    })

                full_idx = np.arange(N_source, dtype=np.int64)
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
                    "avg_grm_obj": float(np.mean(avg_grm)),
                    "target_island": int(target_code),
                    "target_island_name": str(target_name),
                    "repeat": int(repeat_idx),
                    "repeat_seed": int(repeat_seed),
                    "trait": trait_name,
                }
                _append_csv(pd.DataFrame([full_row]), results_path)
                if compute_individual_diagnostics:
                    individual_diag_parts.append(
                        _build_individual_diagnostics(
                            train_idx=full_idx,
                            analysis="full_baseline",
                            method="full_source_unweighted",
                            order_seed=-2,
                            weight_scheme="none",
                            weighted_fit_used=False,
                        )
                    )

                # A) Ranked subset top-k analysis.
                for k in step_counts:
                    n_train = int(min(int(k), N_source))
                    chosen = order[:n_train]

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
                        "method": "avggrm_topk",
                        "weight_scheme": "none",
                        "order_seed": -1,
                        "weighted_fit_used": False,
                        "n_individuals": int(n_train),
                        "corr_eval": float(eval_result["corr_eval"]),
                        "mse_adj": float(eval_result["mse_adj"]),
                        "avg_grm_obj": float(np.mean(avg_grm[chosen])) if n_train > 0 else float("nan"),
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "trait": trait_name,
                    }
                    _append_csv(pd.DataFrame([row]), results_path)
                    if compute_individual_diagnostics:
                        individual_diag_parts.append(
                            _build_individual_diagnostics(
                                train_idx=chosen,
                                analysis="ranked_subset",
                                method="avggrm_topk",
                                order_seed=-1,
                                weight_scheme="none",
                                weighted_fit_used=False,
                            )
                        )

                    selected_df = pd.DataFrame({
                        "trait": trait_name,
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "n_train_size": int(n_train),
                        "method": "avggrm_topk",
                        "ringnr": ids_source[chosen],
                        "ringnumber": ids_source[chosen],
                        "source_island": locality_source[chosen].astype(int),
                        "source_island_name": [island_label(int(c), code_to_label) for c in locality_source[chosen]],
                        "rank": ranks[chosen].astype(int),
                        "avg_grm": avg_grm[chosen].astype(float),
                    })
                    _append_csv(selected_df, selected_path)

                for order_seed in range(n_random_reps):
                    rng = np.random.default_rng(repeat_seed + 500_000 + order_seed)
                    shuffled = rng.permutation(N_source)
                    for k in step_counts:
                        n_train = int(min(int(k), N_source))
                        chosen = shuffled[:n_train]
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
                            "avg_grm_obj": float("nan"),
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "trait": trait_name,
                        }
                        _append_csv(pd.DataFrame([rand_row]), results_path)
                        if compute_individual_diagnostics:
                            individual_diag_parts.append(
                                _build_individual_diagnostics(
                                    train_idx=chosen,
                                    analysis="ranked_subset",
                                    method="random_individual",
                                    order_seed=int(order_seed),
                                    weight_scheme="none",
                                    weighted_fit_used=False,
                                )
                            )

                # B) Full-set weighted analysis.
                for scheme_cfg in weight_schemes:
                    scheme_name, w = _weights_from_scheme(avg_grm, ranks, scheme_cfg)
                    weighted_used = True

                    try:
                        eval_result = _evaluate_ridge_subset(
                            train_idx=full_idx,
                            X_source=X_source,
                            y_source=y_source,
                            X_test=X_target,
                            y_test=y_target,
                            y_eval_test=y_eval_target,
                            alpha=ridge_alpha,
                            snp_cols=fixed_snp_cols,
                            sample_weight=w,
                        )
                    except TypeError:
                        weighted_used = False
                        eval_result = _evaluate_ridge_subset(
                            train_idx=full_idx,
                            X_source=X_source,
                            y_source=y_source,
                            X_test=X_target,
                            y_test=y_target,
                            y_eval_test=y_eval_target,
                            alpha=ridge_alpha,
                            snp_cols=fixed_snp_cols,
                        )

                    row = {
                        "analysis": "weighted_full",
                        "method": f"avggrm_weight_{scheme_name}",
                        "weight_scheme": scheme_name,
                        "order_seed": -1,
                        "weighted_fit_used": bool(weighted_used),
                        "n_individuals": int(N_source),
                        "corr_eval": float(eval_result["corr_eval"]),
                        "mse_adj": float(eval_result["mse_adj"]),
                        "avg_grm_obj": float(np.mean(avg_grm)),
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "trait": trait_name,
                    }
                    _append_csv(pd.DataFrame([row]), results_path)

                    w_df = pd.DataFrame({
                        "trait": trait_name,
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "method": f"avggrm_weight_{scheme_name}",
                        "weight_scheme": scheme_name,
                        "ringnr": ids_source,
                        "ringnumber": ids_source,
                        "source_island": locality_source.astype(int),
                        "source_island_name": [island_label(int(c), code_to_label) for c in locality_source],
                        "rank": ranks.astype(int),
                        "avg_grm": avg_grm.astype(float),
                        "weight": w.astype(float),
                    })
                    _append_csv(w_df, weights_path)
                    if compute_individual_diagnostics:
                        individual_diag_parts.append(
                            _build_individual_diagnostics(
                                train_idx=full_idx,
                                analysis="weighted_full",
                                method=f"avggrm_weight_{scheme_name}",
                                order_seed=-1,
                                weight_scheme=scheme_name,
                                weighted_fit_used=bool(weighted_used),
                                sample_weight=w if weighted_used else None,
                            )
                        )

                # C) Store full-set selected individuals (avg_grm for entire source pool)
                full_selected_df = pd.DataFrame({
                    "trait": trait_name,
                    "target_island": int(target_code),
                    "target_island_name": str(target_name),
                    "repeat": int(repeat_idx),
                    "repeat_seed": int(repeat_seed),
                    "n_train_size": int(N_source),
                    "method": "avggrm_topk",
                    "ringnr": ids_source,
                    "ringnumber": ids_source,
                    "source_island": locality_source.astype(int),
                    "source_island_name": [island_label(int(c), code_to_label) for c in locality_source],
                    "rank": ranks.astype(int),
                    "avg_grm": avg_grm.astype(float),
                })
                _append_csv(full_selected_df, selected_path)

                # D) Individual-level diagnostics (optional): one row per target individual per fitted model
                if compute_individual_diagnostics and individual_diag_parts:
                    _append_csv(pd.concat(individual_diag_parts, ignore_index=True), individual_diag_path)

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
            summary_csv = trait_output / "avggrm_rank_weight_summary.csv"
            summary.to_csv(summary_csv, index=False)

        logger.info("Trait '%s' complete. Output: %s", trait_name, trait_output)


if __name__ == "__main__":
    main()
