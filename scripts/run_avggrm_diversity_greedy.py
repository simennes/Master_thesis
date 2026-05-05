#!/usr/bin/env python
"""
Run greedy avgGRM-diversity training-set selection.

For each target island, candidates are ranked by greedy maximization of

    avgGRM(train, target) - lambda_div * avgGRM(train, train).

The same selected subsets can then be evaluated with either Ridge or BPCRR-INLA.

Usage
-----
Worker:
  python -m scripts.run_avggrm_diversity_greedy --mode worker --config config/avggrm_diversity_greedy_config.json

Merge shards:
  python -m scripts.run_avggrm_diversity_greedy --mode merge --config config/avggrm_diversity_greedy_config.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scripts.run_avggrm_rank_weight import (
    _append_csv,
    _assign_jobs_weighted,
    _resolve_training_islands,
    resolve_island_code,
)
from scripts.run_bpcrr_inla_rank_select import (
    _evaluate_bpcrr_subset,
    _parse_bpcrr_prior_cfg,
    _parse_n_components_values,
    _prepare_one_step_covariates,
    _resolve_existing_path,
    _set_seed,
)
from src.avggrm_weighting import greedy_avggrm_diversity_order
from src.cv_utils import island_label
from src.data import load_data
from src.training_set_optimization.runner import _evaluate_ridge_subset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EXP_KEY = "avggrm_diversity_greedy_experiment"


def _format_label_value(value: float) -> str:
    s = f"{float(value):g}"
    return s.replace("-", "m").replace(".", "p")


def _make_repeat_seed(global_seed: int, target_code: int, repeat_idx: int) -> int:
    token = f"avggrm_diversity_greedy|{int(global_seed)}|{int(target_code)}|{int(repeat_idx)}"
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % 2_147_483_647


def _get_model_type(cfg: Dict[str, Any], override: Optional[str] = None) -> str:
    raw = override if override is not None else cfg.get("model", {}).get("type", "ridge")
    model_type = str(raw).strip().lower()
    aliases = {"bpcrr_inla": "bpcrr", "inla": "bpcrr"}
    model_type = aliases.get(model_type, model_type)
    if model_type not in {"ridge", "bpcrr"}:
        raise ValueError("model.type must be 'ridge' or 'bpcrr'")
    return model_type


def _model_override_cfg(cfg: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    overrides = cfg.get("model_overrides", {})
    if not isinstance(overrides, dict):
        return {}
    out = overrides.get(model_type, {})
    return out if isinstance(out, dict) else {}


def _build_trait_specs(cfg: Dict[str, Any], model_type: str) -> List[Dict[str, Any]]:
    traits_cfg = cfg.get("traits", None)
    model_overrides = _model_override_cfg(cfg, model_type)

    default_target = model_overrides.get("target_column", cfg.get("target_column", "y_adjusted"))
    default_eval = model_overrides.get("eval_target_column", cfg.get("eval_target_column", "y_mean"))
    default_standardize = bool(model_overrides.get("standardize_features", cfg.get("standardize_features", True)))
    default_min_count = int(model_overrides.get("min_count", cfg.get("min_count", 20)))

    if traits_cfg is None:
        return [{
            "name": "default",
            "paths": dict(cfg["paths"]),
            "target_column": default_target,
            "eval_target_column": default_eval,
            "standardize_features": default_standardize,
            "min_count": default_min_count,
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
            "target_column": t.get("target_column", default_target),
            "eval_target_column": t.get("eval_target_column", default_eval),
            "standardize_features": bool(t.get("standardize_features", default_standardize)),
            "min_count": int(t.get("min_count", default_min_count)),
        })
    return specs


def _parse_lambda_divs(exp_cfg: Dict[str, Any]) -> List[float]:
    raw = exp_cfg.get("lambda_divs", [0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
    if isinstance(raw, (int, float, str)):
        raw_values = [raw]
    else:
        raw_values = list(raw)

    values: List[float] = []
    seen = set()
    for item in raw_values:
        val = float(item)
        if not np.isfinite(val) or val < 0:
            raise ValueError("avggrm_diversity_greedy_experiment.lambda_divs must be finite values >= 0")
        key = f"{val:.12g}"
        if key not in seen:
            seen.add(key)
            values.append(val)

    if not values:
        raise ValueError("At least one lambda_div value is required")
    return values


def _compute_step_counts(
    n_train_sizes_raw: Optional[List[Any]],
    locality_source: np.ndarray,
    source_codes: List[int],
    n_source: int,
    include_full_size: bool,
) -> np.ndarray:
    if n_train_sizes_raw is not None:
        step_counts = np.array(sorted(int(x) for x in n_train_sizes_raw), dtype=np.int64)
    else:
        n_per_island = np.array([(locality_source == c).sum() for c in source_codes], dtype=np.int64)
        step_counts = np.cumsum(np.sort(n_per_island)[::-1])
    step_counts = np.unique(np.clip(step_counts, 2, n_source)).astype(np.int64)
    if not include_full_size:
        step_counts = step_counts[step_counts < n_source]
    return step_counts


def _resolve_trait_paths(config_path: Path, trait_paths: Dict[str, Any]) -> Dict[str, Any]:
    paths = dict(trait_paths)
    paths["npz"] = str(_resolve_existing_path(config_path, paths.get("npz"), required=True))
    paths["grm_rds"] = str(_resolve_existing_path(config_path, paths.get("grm_rds"), required=True))
    return paths


def _avg_train_train_block(grm: np.ndarray, idx: np.ndarray, include_diagonal: bool) -> float:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return float("nan")
    block = np.asarray(grm[np.ix_(idx, idx)], dtype=float)
    if include_diagonal or idx.size <= 1:
        return float(np.mean(block))
    offdiag_sum = float(np.sum(block) - np.trace(block))
    return offdiag_sum / float(idx.size * (idx.size - 1))


def _evaluate_subset(
    model_type: str,
    train_idx: np.ndarray,
    context: Dict[str, Any],
) -> List[Tuple[float, Dict[str, float]]]:
    if model_type == "ridge":
        result = _evaluate_ridge_subset(
            train_idx=train_idx,
            X_source=context["X_source"],
            y_source=context["y_source"],
            X_test=context["X_target"],
            y_test=context["y_target"],
            y_eval_test=context["y_eval_target"],
            alpha=float(context["ridge_alpha"]),
            snp_cols=context.get("fixed_snp_cols"),
        )
        return [(float("nan"), result)]

    out: List[Tuple[float, Dict[str, float]]] = []
    seen_components = set()
    z_dim = int(context["Z_source_full"].shape[1])
    for n_comp_req in context["bpcrr_n_components_values"]:
        n_comp = int(min(int(n_comp_req), z_dim))
        if n_comp < 1 or n_comp in seen_components:
            continue
        seen_components.add(n_comp)
        result = _evaluate_bpcrr_subset(
            train_idx=train_idx,
            Z_source=context["Z_source_full"][:, :n_comp],
            y_source=context["y_source"],
            Z_target=context["Z_target_full"][:, :n_comp],
            y_target=context["y_target"],
            y_eval_target=context["y_eval_target"],
            one_step_source=context.get("one_step_source"),
            one_step_target=context.get("one_step_target"),
            rr_prior_mode=str(context["bpcrr_prior_mode"]),
            rr_va_apriori=context.get("bpcrr_va_apriori"),
        )
        out.append((float(n_comp), result))
    return out


def run_merge(config_path: Path, model_type_override: Optional[str] = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_type = _get_model_type(cfg, model_type_override)
    exp_cfg = cfg.get(EXP_KEY, {})
    output_dir = Path(exp_cfg.get("output_dir", cfg["paths"].get("output_dir", "outputs/avggrm_diversity_greedy")))
    trait_specs = _build_trait_specs(cfg, model_type)
    shards_root = output_dir / "shards"

    if not shards_root.exists():
        raise FileNotFoundError(f"No shard directory found: {shards_root}")

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        trait_output = output_dir / trait_name if len(trait_specs) > 1 else output_dir
        trait_output.mkdir(parents=True, exist_ok=True)

        merge_specs = [
            ("avggrm_diversity_greedy_results.csv", True),
            ("avggrm_diversity_greedy_selected_individuals.csv", False),
        ]

        merged_results: Optional[pd.DataFrame] = None
        for filename, is_results in merge_specs:
            parts: List[pd.DataFrame] = []
            for csv_path in sorted(shards_root.glob(f"shard_*/{trait_name}/{filename}")):
                if csv_path.exists():
                    parts.append(pd.read_csv(csv_path))
            if not parts:
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
                    [
                        "trait",
                        "model_type",
                        "target_island",
                        "target_island_name",
                        "analysis",
                        "method",
                        "selection_method",
                        "lambda_label",
                        "lambda_div",
                        "n_components",
                        "n_individuals",
                    ],
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    corr_mean=("corr_eval", "mean"),
                    corr_std=("corr_eval", "std"),
                    mse_mean=("mse_adj", "mean"),
                    avg_grm_target_mean=("avg_grm_target_obj", "mean"),
                    avg_grm_train_train_mean=("avg_grm_train_train_obj", "mean"),
                    objective_mean=("objective_obj", "mean"),
                    n_rows=("corr_eval", "size"),
                )
            )
            summary_path = trait_output / "avggrm_diversity_greedy_summary.csv"
            summary.to_csv(summary_path, index=False)
            logger.info("Wrote summary for trait '%s'", trait_name)

    logger.info("Merge complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy avgGRM-diversity training-set analysis")
    parser.add_argument("--mode", choices=["worker", "merge"], default="worker")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--model_type", choices=["ridge", "bpcrr", "bpcrr_inla"], default=None)
    parser.add_argument("--target_islands", nargs="+", default=None)
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_type = _get_model_type(cfg, args.model_type)
    if args.mode == "merge":
        run_merge(config_path, args.model_type)
        return

    exp_cfg = cfg.get(EXP_KEY, {})
    output_dir = Path(exp_cfg.get("output_dir", cfg["paths"].get("output_dir", "outputs/avggrm_diversity_greedy")))
    output_dir.mkdir(parents=True, exist_ok=True)

    global_seed = int(cfg.get("seed", 42))
    _set_seed(global_seed)

    lambda_divs = _parse_lambda_divs(exp_cfg)
    n_repeats = int(exp_cfg.get("n_repeats", 1))
    if n_repeats < 1:
        raise ValueError(f"{EXP_KEY}.n_repeats must be >= 1")
    n_random_reps = int(exp_cfg.get("n_random_reps", cfg.get("baselines", {}).get("n_random_orders", 0)))
    if n_random_reps < 0:
        raise ValueError(f"{EXP_KEY}.n_random_reps must be >= 0")
    n_train_sizes_raw = exp_cfg.get("n_train_sizes", None)
    training_islands_raw = exp_cfg.get("training_islands", None)
    include_diagonal = bool(exp_cfg.get("diversity_include_diagonal", True))
    include_full_size = bool(exp_cfg.get("include_full_size", False))

    model_cfg = cfg.get("model", {})
    ridge_alpha = float(model_cfg.get("alpha", exp_cfg.get("ridge_alpha", 1e5)))
    use_snp_selection = bool(exp_cfg.get("use_snp_selection", cfg.get("use_snp_selection", False)))
    num_snps = exp_cfg.get("num_snps", cfg.get("num_snps", None))

    if model_type == "bpcrr":
        bpcrr_prior_mode, bpcrr_va_apriori = _parse_bpcrr_prior_cfg(exp_cfg)
        bpcrr_n_components_values = _parse_n_components_values(
            exp_cfg,
            key="bpcrr_n_components",
            default_value=int(exp_cfg.get("n_components", 20) if exp_cfg.get("n_components", 20) is not None else 20),
        )
    else:
        bpcrr_prior_mode = "default"
        bpcrr_va_apriori = None
        bpcrr_n_components_values = []

    one_step_cfg = exp_cfg.get("one_step", {})
    if isinstance(one_step_cfg, bool):
        one_step_enabled = bool(one_step_cfg)
    else:
        one_step_enabled = bool(one_step_cfg.get("enabled", False))

    trait_specs = _build_trait_specs(cfg, model_type)

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

        logger.info("Loading data for trait '%s' using model=%s", trait_name, model_type)
        trait_paths = _resolve_trait_paths(config_path, dict(trait_spec["paths"]))
        X, y, ids, GRM_df, locality, code_to_label, y_eval = load_data(
            paths=trait_paths,
            target_column=trait_spec["target_column"],
            standardize_features=trait_spec["standardize_features"],
            return_locality=True,
            min_count=trait_spec["min_count"],
            return_eval=True,
            eval_target_column=trait_spec["eval_target_column"],
        )
        if GRM_df is None:
            raise ValueError("GRM is required. Provide paths.grm_rds in the config.")

        one_step_covars = None
        if model_type == "bpcrr":
            cfg_for_one_step = dict(cfg)
            cfg_for_one_step["bpcrr_inla_experiment"] = exp_cfg
            one_step_covars = _prepare_one_step_covariates(
                config_path=config_path,
                cfg=cfg_for_one_step,
                ids=ids,
                locality_codes=locality,
                code_to_label=code_to_label,
            )
            if one_step_enabled and one_step_covars is None:
                raise RuntimeError("one_step is enabled but covariates could not be prepared")

        present_codes = set(int(c) for c in np.unique(locality))
        included_raw = cfg.get("included_islands", None)
        if included_raw is not None:
            included_island_codes = [resolve_island_code(v, code_to_label, present_codes) for v in included_raw]
        else:
            included_island_codes = sorted(present_codes)

        raw_targets = args.target_islands if args.target_islands else cfg.get("target_islands", [0])
        target_codes = [resolve_island_code(t, code_to_label, present_codes) for t in raw_targets]

        fixed_snp_cols: Optional[np.ndarray] = None
        if model_type == "ridge" and use_snp_selection and num_snps is not None and int(num_snps) < X.shape[1]:
            rng_snp = np.random.default_rng(global_seed)
            fixed_snp_cols = np.sort(rng_snp.choice(X.shape[1], size=int(num_snps), replace=False).astype(np.int64))
            np.save(trait_output / "fixed_snp_columns.npy", fixed_snp_cols)
            logger.info("Fixed SNP subset: n=%d", len(fixed_snp_cols))

        results_path = trait_output / "avggrm_diversity_greedy_results.csv"
        selected_path = trait_output / "avggrm_diversity_greedy_selected_individuals.csv"
        for p in [results_path, selected_path]:
            if p.exists():
                p.unlink()

        jobs: List[Dict[str, Any]] = []
        step_counts_by_target: Dict[int, np.ndarray] = {}

        n_comp_factor = max(1, len(bpcrr_n_components_values)) if model_type == "bpcrr" else 1
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

            target_mask = locality == target_code
            source_mask = (~target_mask) & np.isin(locality, source_codes)
            n_source = int(np.sum(source_mask))
            if n_source < 2:
                continue

            step_counts = _compute_step_counts(
                n_train_sizes_raw,
                locality[source_mask],
                source_codes,
                n_source,
                include_full_size=include_full_size,
            )
            if len(step_counts) == 0:
                continue
            step_counts_by_target[int(target_code)] = step_counts
            max_k = int(np.max(step_counts))
            sum_k = int(np.sum(step_counts))

            for repeat_idx in range(n_repeats):
                jobs.append({
                    "target_code": int(target_code),
                    "repeat_idx": int(repeat_idx),
                    "task": "baseline",
                    "weight": float(max(2, n_source) * n_comp_factor),
                })
                for lam in lambda_divs:
                    jobs.append({
                        "target_code": int(target_code),
                        "repeat_idx": int(repeat_idx),
                        "task": "lambda",
                        "lambda_div": float(lam),
                        "weight": float(max(2, max_k) + len(step_counts) * n_comp_factor * max(2, max_k)),
                    })
                for order_seed in range(n_random_reps):
                    jobs.append({
                        "target_code": int(target_code),
                        "repeat_idx": int(repeat_idx),
                        "task": "random",
                        "order_seed": int(order_seed),
                        "weight": float(max(2, sum_k) * n_comp_factor),
                    })

        shard_bins = _assign_jobs_weighted(jobs, num_shards)
        shard_jobs = shard_bins[shard_index] if num_shards > 1 else jobs

        assignments: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for job in shard_jobs:
            key = (int(job["target_code"]), int(job["repeat_idx"]))
            if key not in assignments:
                assignments[key] = {"baseline": False, "lambda_divs": [], "random_order_seeds": []}
            task = str(job.get("task"))
            if task == "baseline":
                assignments[key]["baseline"] = True
            elif task == "lambda":
                assignments[key]["lambda_divs"].append(float(job["lambda_div"]))
            elif task == "random":
                assignments[key]["random_order_seeds"].append(int(job["order_seed"]))

        for assignment in assignments.values():
            assigned_lambdas = {f"{float(x):.12g}" for x in assignment["lambda_divs"]}
            assignment["lambda_divs"] = [
                float(lam) for lam in lambda_divs if f"{float(lam):.12g}" in assigned_lambdas
            ]
            assignment["random_order_seeds"] = sorted(set(int(x) for x in assignment["random_order_seeds"]))

        active_keys = [
            key for key, a in assignments.items()
            if bool(a["baseline"]) or a["lambda_divs"] or a["random_order_seeds"]
        ]
        logger.info(
            "Shard %d/%d assigned %d/%d jobs across %d repeat groups",
            shard_index,
            num_shards,
            len(shard_jobs),
            len(jobs),
            len(active_keys),
        )

        done_groups = 0
        for target_code in target_codes:
            target_name = island_label(target_code, code_to_label)
            source_codes = [c for c in included_island_codes if c != target_code]
            if training_islands_raw is not None:
                training_codes = _resolve_training_islands(
                    training_islands_raw, code_to_label, present_codes, target_code
                )
                if training_codes is not None:
                    source_codes = [c for c in source_codes if c in training_codes]
            if len(source_codes) == 0:
                continue

            target_mask = locality == target_code
            source_mask = (~target_mask) & np.isin(locality, source_codes)
            if int(np.sum(source_mask)) < 2 or int(np.sum(target_mask)) == 0:
                continue

            X_source = X[source_mask]
            y_source = y[source_mask]
            ids_source = ids[source_mask]
            locality_source = locality[source_mask]

            X_target = X[target_mask]
            y_target = y[target_mask]
            y_eval_target = y_eval[target_mask]
            ids_target = ids[target_mask]

            n_source = int(len(X_source))
            step_counts = step_counts_by_target.get(int(target_code), np.array([], dtype=np.int64))
            if len(step_counts) == 0:
                continue

            grm_source_target = GRM_df.loc[ids_source, ids_target].to_numpy(dtype=float)
            avg_grm_to_target = np.asarray(grm_source_target.mean(axis=1), dtype=float)
            grm_source_source = GRM_df.loc[ids_source, ids_source].to_numpy(dtype=float)

            one_step_source = None
            one_step_target = None
            Z_source_full = None
            Z_target_full = None
            if model_type == "bpcrr":
                max_requested = int(max(bpcrr_n_components_values))
                max_feasible = int(min(max_requested, X_source.shape[0], X_source.shape[1]))
                if max_feasible < 1:
                    continue
                pca = PCA(n_components=max_feasible)
                Z_source_full = pca.fit_transform(X_source)
                Z_target_full = pca.transform(X_target)
                if one_step_covars is not None:
                    one_step_source = {
                        k: (None if v is None else np.asarray(v)[source_mask])
                        for k, v in one_step_covars.items()
                    }
                    one_step_target = {
                        k: (None if v is None else np.asarray(v)[target_mask])
                        for k, v in one_step_covars.items()
                    }

            context = {
                "X_source": X_source,
                "y_source": y_source,
                "X_target": X_target,
                "y_target": y_target,
                "y_eval_target": y_eval_target,
                "ridge_alpha": ridge_alpha,
                "fixed_snp_cols": fixed_snp_cols,
                "Z_source_full": Z_source_full,
                "Z_target_full": Z_target_full,
                "bpcrr_n_components_values": bpcrr_n_components_values,
                "bpcrr_prior_mode": bpcrr_prior_mode,
                "bpcrr_va_apriori": bpcrr_va_apriori,
                "one_step_source": one_step_source,
                "one_step_target": one_step_target,
            }

            for repeat_idx in range(n_repeats):
                key = (int(target_code), int(repeat_idx))
                assignment = assignments.get(key)
                if assignment is None:
                    continue
                if not bool(assignment["baseline"]) and not assignment["lambda_divs"] and not assignment["random_order_seeds"]:
                    continue

                done_groups += 1
                repeat_seed = _make_repeat_seed(global_seed, int(target_code), int(repeat_idx))
                logger.info(
                    "Group %d/%d | trait=%s model=%s target=%s repeat=%d/%d lambdas=%d random=%d",
                    done_groups,
                    len(active_keys),
                    trait_name,
                    model_type,
                    target_code,
                    repeat_idx + 1,
                    n_repeats,
                    len(assignment["lambda_divs"]),
                    len(assignment["random_order_seeds"]),
                )

                full_idx = np.arange(n_source, dtype=np.int64)
                if bool(assignment["baseline"]):
                    for n_comp, eval_result in _evaluate_subset(model_type, full_idx, context):
                        row = {
                            "trait": trait_name,
                            "model_type": model_type,
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "analysis": "full_baseline",
                            "method": "full_source_unweighted",
                            "selection_method": "none",
                            "lambda_label": "none",
                            "lambda_div": np.nan,
                            "order_seed": -2,
                            "n_components": n_comp,
                            "n_individuals": int(n_source),
                            "corr_eval": float(eval_result["corr_eval"]),
                            "mse_adj": float(eval_result["mse_adj"]),
                            "avg_grm_target_obj": float(np.mean(avg_grm_to_target)),
                            "avg_grm_train_train_obj": _avg_train_train_block(
                                grm_source_source, full_idx, include_diagonal
                            ),
                            "objective_obj": float("nan"),
                            "diversity_include_diagonal": bool(include_diagonal),
                        }
                        _append_csv(pd.DataFrame([row]), results_path)

                for order_seed in assignment["random_order_seeds"]:
                    rng = np.random.default_rng(repeat_seed + 500_000 + int(order_seed))
                    shuffled = rng.permutation(n_source)
                    for k in step_counts:
                        n_train = int(min(int(k), n_source))
                        chosen = shuffled[:n_train]
                        avg_target_obj = float(np.mean(avg_grm_to_target[chosen]))
                        avg_train_train_obj = _avg_train_train_block(
                            grm_source_source, chosen, include_diagonal
                        )
                        for n_comp, eval_result in _evaluate_subset(model_type, chosen, context):
                            row = {
                                "trait": trait_name,
                                "model_type": model_type,
                                "target_island": int(target_code),
                                "target_island_name": str(target_name),
                                "repeat": int(repeat_idx),
                                "repeat_seed": int(repeat_seed),
                                "analysis": "ranked_subset",
                                "method": "random_individual",
                                "selection_method": "random",
                                "lambda_label": "none",
                                "lambda_div": np.nan,
                                "order_seed": int(order_seed),
                                "n_components": n_comp,
                                "n_individuals": int(n_train),
                                "corr_eval": float(eval_result["corr_eval"]),
                                "mse_adj": float(eval_result["mse_adj"]),
                                "avg_grm_target_obj": avg_target_obj,
                                "avg_grm_train_train_obj": avg_train_train_obj,
                                "objective_obj": float("nan"),
                                "diversity_include_diagonal": bool(include_diagonal),
                            }
                            _append_csv(pd.DataFrame([row]), results_path)

                max_k = int(np.max(step_counts))
                for lambda_div in assignment["lambda_divs"]:
                    lambda_label = f"lambda_{_format_label_value(lambda_div)}"
                    trace = greedy_avggrm_diversity_order(
                        avg_grm_to_target=avg_grm_to_target,
                        train_train_grm=grm_source_source,
                        lambda_div=float(lambda_div),
                        max_size=max_k,
                        include_diagonal=include_diagonal,
                    )
                    order = np.asarray(trace["order"], dtype=np.int64)
                    ranks_full = np.full(n_source, -1, dtype=np.int64)
                    ranks_full[order] = np.arange(1, len(order) + 1, dtype=np.int64)

                    for k in step_counts:
                        n_train = int(min(int(k), len(order)))
                        if n_train < 2:
                            continue
                        chosen = order[:n_train]
                        trace_pos = n_train - 1
                        objective_obj = float(trace["objective_after"][trace_pos])
                        avg_target_obj = float(trace["avg_grm_train_target_after"][trace_pos])
                        avg_train_train_obj = float(trace["avg_grm_train_train_after"][trace_pos])

                        method_name = f"{model_type}_avggrm_diversity_greedy"
                        for n_comp, eval_result in _evaluate_subset(model_type, chosen, context):
                            row = {
                                "trait": trait_name,
                                "model_type": model_type,
                                "target_island": int(target_code),
                                "target_island_name": str(target_name),
                                "repeat": int(repeat_idx),
                                "repeat_seed": int(repeat_seed),
                                "analysis": "ranked_subset",
                                "method": method_name,
                                "selection_method": "avggrm_diversity_greedy",
                                "lambda_label": lambda_label,
                                "lambda_div": float(lambda_div),
                                "order_seed": -1,
                                "n_components": n_comp,
                                "n_individuals": int(n_train),
                                "corr_eval": float(eval_result["corr_eval"]),
                                "mse_adj": float(eval_result["mse_adj"]),
                                "avg_grm_target_obj": avg_target_obj,
                                "avg_grm_train_train_obj": avg_train_train_obj,
                                "objective_obj": objective_obj,
                                "diversity_include_diagonal": bool(include_diagonal),
                            }
                            _append_csv(pd.DataFrame([row]), results_path)

                        pos = np.arange(n_train, dtype=np.int64)
                        selected_df = pd.DataFrame({
                            "trait": trait_name,
                            "model_type": model_type,
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "n_train_size": int(n_train),
                            "method": method_name,
                            "selection_method": "avggrm_diversity_greedy",
                            "lambda_label": lambda_label,
                            "lambda_div": float(lambda_div),
                            "ringnr": ids_source[chosen],
                            "ringnumber": ids_source[chosen],
                            "source_island": locality_source[chosen].astype(int),
                            "source_island_name": [
                                island_label(int(c), code_to_label) for c in locality_source[chosen]
                            ],
                            "rank": ranks_full[chosen].astype(int),
                            "avg_grm_to_target": avg_grm_to_target[chosen].astype(float),
                            "avg_grm_added_to_selected_before": trace["avg_grm_added_to_selected_before"][pos],
                            "self_grm": trace["self_grm"][pos],
                            "objective_after_add": trace["objective_after"][pos],
                            "marginal_gain_at_add": trace["marginal_gain"][pos],
                            "objective_at_k": objective_obj,
                            "avg_grm_target_at_k": avg_target_obj,
                            "avg_grm_train_train_at_k": avg_train_train_obj,
                            "diversity_include_diagonal": bool(include_diagonal),
                        })
                        _append_csv(selected_df, selected_path)

        if results_path.exists():
            all_results = pd.read_csv(results_path)
            summary = (
                all_results.groupby(
                    [
                        "trait",
                        "model_type",
                        "target_island",
                        "target_island_name",
                        "analysis",
                        "method",
                        "selection_method",
                        "lambda_label",
                        "lambda_div",
                        "n_components",
                        "n_individuals",
                    ],
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    corr_mean=("corr_eval", "mean"),
                    corr_std=("corr_eval", "std"),
                    mse_mean=("mse_adj", "mean"),
                    avg_grm_target_mean=("avg_grm_target_obj", "mean"),
                    avg_grm_train_train_mean=("avg_grm_train_train_obj", "mean"),
                    objective_mean=("objective_obj", "mean"),
                    n_rows=("corr_eval", "size"),
                )
            )
            summary.to_csv(trait_output / "avggrm_diversity_greedy_summary.csv", index=False)

        logger.info("Trait '%s' complete. Output: %s", trait_name, trait_output)


if __name__ == "__main__":
    main()
