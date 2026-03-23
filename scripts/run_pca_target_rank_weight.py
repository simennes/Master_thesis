#!/usr/bin/env python
"""
PCA-on-target-island training-set selection and weighting.

Conceptually equivalent to avg-GRM top-k but uses a PCA distance metric:
  1. Compute PCA on the *target island* genotypes.
  2. Project every source individual onto those PCs.
  3. Distance = Euclidean distance from each projected source individual
     to the target-island centre in PC space.
  4. Rank source individuals by distance (ascending — closer is better).
  5. Evaluate ridge on top-k subsets and weighted full-set fits.

Usage
-----
Worker:
  python -m scripts.run_pca_target_rank_weight --mode worker --config config/pca_target_rank_weight_config.json

Merge shards:
  python -m scripts.run_pca_target_rank_weight --mode merge --config config/pca_target_rank_weight_config.json
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

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


# ---------------------------------------------------------------------------
# Helpers (shared with avg-GRM script)
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
    token = f"pca_target|{int(global_seed)}|{int(target_code)}|{int(repeat_idx)}"
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
    score: np.ndarray,
    ranks: np.ndarray,
    scheme_cfg: Dict[str, Any],
) -> Tuple[str, np.ndarray]:
    """Compute sample weights from a score vector (higher = better)."""
    name = str(scheme_cfg.get("name", "uniform")).lower()
    floor = float(scheme_cfg.get("floor", 1e-6))
    clip_max = scheme_cfg.get("clip_max", None)
    s = _safe_minmax(score)
    n = len(score)
    if name == "uniform":
        w = np.ones(n, dtype=float)
    elif name == "linear":
        a = float(scheme_cfg.get("min_weight", 0.25))
        b = float(scheme_cfg.get("max_weight", 1.75))
        w = a + (b - a) * s
    elif name == "minmax":
        eps = float(scheme_cfg.get("eps", 0.05))
        w = eps + s
    elif name == "exponential":
        beta = float(scheme_cfg.get("beta", 3.0))
        w = np.exp(beta * s)
    elif name == "top-heavy":
        top_frac = float(scheme_cfg.get("top_frac", 0.2))
        high = float(scheme_cfg.get("high", 3.0))
        low = float(scheme_cfg.get("low", 1.0))
        top_n = max(1, int(np.ceil(top_frac * n)))
        w = np.full(n, low, dtype=float)
        order = np.argsort(ranks)
        w[order[:top_n]] = high
    else:
        raise ValueError(f"Unknown weight scheme: {name}")
    return name, _normalize_mean_one(w, floor=floor, clip_max=clip_max)


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


def _parse_n_components_values(exp_cfg: Dict[str, Any]) -> List[int]:
    """Accept an int or list-like config for PCA component counts."""
    raw = exp_cfg.get("n_components", 50)
    if raw is None:
        raw = 50

    if isinstance(raw, (list, tuple)):
        values = [int(v) for v in raw]
    else:
        values = [int(raw)]

    values = sorted(set(v for v in values if int(v) > 0))
    if not values:
        raise ValueError("pca_target_experiment.n_components must contain at least one positive integer")
    return values


# ---------------------------------------------------------------------------
# PCA-on-target distance
# ---------------------------------------------------------------------------

def compute_pca_target_distances(
    X_target: np.ndarray,
    X_source: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """
    Fit PCA on target-island genotypes, project source individuals,
    and return Euclidean distance of each source individual to the
    target centre in PC space.
    """
    n_components = min(n_components, X_target.shape[0], X_target.shape[1])
    pca = PCA(n_components=n_components)
    Z_target = pca.fit_transform(X_target)  # (n_target, n_components)
    target_centre = Z_target.mean(axis=0)   # (n_components,)
    Z_source = pca.transform(X_source)      # (n_source, n_components)
    distances = np.linalg.norm(Z_source - target_centre, axis=1)
    return distances


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def run_merge(config_path: Path) -> None:
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    exp_cfg = cfg.get("pca_target_experiment", {})
    output_dir = Path(exp_cfg.get("output_dir", cfg["paths"].get("output_dir", "outputs/pca_target_rank_weight")))
    trait_specs = _build_trait_specs(cfg)
    shards_root = output_dir / "shards"

    if not shards_root.exists():
        raise FileNotFoundError(f"No shard directory found: {shards_root}")

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        trait_output = output_dir / trait_name if len(trait_specs) > 1 else output_dir
        trait_output.mkdir(parents=True, exist_ok=True)

        merge_specs = [
            ("pca_target_rank_weight_results.csv", True),
            ("pca_target_ranked_selected_individuals.csv", False),
            ("pca_target_weighted_individuals.csv", False),
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
                    ["trait", "target_island", "target_island_name", "analysis", "method", "n_components", "n_individuals"],
                    as_index=False,
                )
                .agg(
                    corr_mean=("corr_eval", "mean"),
                    corr_std=("corr_eval", "std"),
                    mse_mean=("mse_adj", "mean"),
                    n_rows=("corr_eval", "size"),
                )
            )
            summary_csv = trait_output / "pca_target_rank_weight_summary.csv"
            summary.to_csv(summary_csv, index=False)
            logger.info("Wrote summary for trait '%s'", trait_name)

    logger.info("Merge complete.")


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PCA-on-target ranked + weighted training-set analysis")
    parser.add_argument("--mode", choices=["worker", "merge"], default="worker")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--target_islands", nargs="+", default=None)
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

    exp_cfg = cfg.get("pca_target_experiment", {})
    output_dir = Path(exp_cfg.get("output_dir", cfg["paths"].get("output_dir", "outputs/pca_target_rank_weight")))
    output_dir.mkdir(parents=True, exist_ok=True)

    global_seed = int(cfg.get("seed", 42))
    set_seed(global_seed)

    model_cfg = cfg.get("model", {})
    ridge_alpha = float(model_cfg.get("alpha", 1e5))

    n_repeats = int(exp_cfg.get("n_repeats", 1))
    n_random_reps = int(exp_cfg.get("n_random_reps", cfg.get("baselines", {}).get("n_random_orders", 5)))
    n_components_values = _parse_n_components_values(exp_cfg)
    n_train_sizes_raw = exp_cfg.get("n_train_sizes", None)
    training_islands_raw = exp_cfg.get("training_islands", None)
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

        results_path = trait_output / "pca_target_rank_weight_results.csv"
        selected_path = trait_output / "pca_target_ranked_selected_individuals.csv"
        weights_path = trait_output / "pca_target_weighted_individuals.csv"
        for p in [results_path, selected_path, weights_path]:
            if p.exists():
                p.unlink()

        # Build jobs
        jobs: List[Dict[str, Any]] = []
        steps_by_target: Dict[int, np.ndarray] = {}

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

            for n_components in n_components_values:
                for repeat_idx in range(n_repeats):
                    jobs.append({
                        "target_code": int(target_code),
                        "n_components": int(n_components),
                        "repeat_idx": int(repeat_idx),
                        "weight": float(max(1, n_source) * (1 + len(step_counts) * (1 + n_random_reps) + len(weight_schemes))),
                    })

        shard_bins = _assign_jobs_weighted(jobs, num_shards)
        shard_jobs = shard_bins[shard_index] if num_shards > 1 else jobs
        assigned = {
            (int(j["target_code"]), int(j["n_components"]), int(j["repeat_idx"]))
            for j in shard_jobs
        }

        logger.info("Shard %d/%d assigned %d/%d jobs", shard_index, num_shards, len(shard_jobs), len(jobs))

        done = 0
        total = len(shard_jobs)

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

            step_counts = steps_by_target.get(int(target_code), np.array([], dtype=np.int64)).copy()
            if len(step_counts) == 0:
                continue

            for n_components in n_components_values:
                # --- PCA on target island, project source, compute distances ---
                distances = compute_pca_target_distances(X_target, X_source, int(n_components))

                # Rank by distance ascending (closer = better = rank 1)
                order = np.argsort(distances, kind="mergesort")
                ranks = np.empty_like(order)
                ranks[order] = np.arange(1, len(order) + 1)

                # Similarity score (inverse distance) for weighting — higher is better
                score = -distances

                for repeat_idx in range(n_repeats):
                    if (int(target_code), int(n_components), int(repeat_idx)) not in assigned:
                        continue

                    done += 1
                    repeat_seed = _make_repeat_seed(global_seed, int(target_code), int(repeat_idx))
                    logger.info(
                        "Job %d/%d | trait=%s target=%s repeat=%d/%d n_components=%d",
                        done, total, trait_name, target_code, repeat_idx + 1, n_repeats, n_components,
                    )

                    # Full-source baseline
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
                        "pca_dist_obj": float(np.mean(distances)),
                        "n_components": int(n_components),
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "trait": trait_name,
                    }
                    _append_csv(pd.DataFrame([full_row]), results_path)

                    # A) Ranked subset top-k analysis
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
                            "method": "pca_target_topk",
                            "weight_scheme": "none",
                            "order_seed": -1,
                            "weighted_fit_used": False,
                            "n_individuals": int(n_train),
                            "corr_eval": float(eval_result["corr_eval"]),
                            "mse_adj": float(eval_result["mse_adj"]),
                            "pca_dist_obj": float(np.mean(distances[chosen])) if n_train > 0 else float("nan"),
                            "n_components": int(n_components),
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "trait": trait_name,
                        }
                        _append_csv(pd.DataFrame([row]), results_path)

                        selected_df = pd.DataFrame({
                            "trait": trait_name,
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "n_train_size": int(n_train),
                            "method": "pca_target_topk",
                            "n_components": int(n_components),
                            "ringnr": ids_source[chosen],
                            "ringnumber": ids_source[chosen],
                            "source_island": locality_source[chosen].astype(int),
                            "source_island_name": [island_label(int(c), code_to_label) for c in locality_source[chosen]],
                            "rank": ranks[chosen].astype(int),
                            "pca_dist": distances[chosen].astype(float),
                        })
                        _append_csv(selected_df, selected_path)

                    # Random baselines
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
                                "pca_dist_obj": float("nan"),
                                "n_components": int(n_components),
                                "target_island": int(target_code),
                                "target_island_name": str(target_name),
                                "repeat": int(repeat_idx),
                                "repeat_seed": int(repeat_seed),
                                "trait": trait_name,
                            }
                            _append_csv(pd.DataFrame([rand_row]), results_path)

                    # B) Full-set weighted analysis
                    for scheme_cfg in weight_schemes:
                        scheme_name, w = _weights_from_scheme(score, ranks, scheme_cfg)
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
                            "method": f"pca_target_weight_{scheme_name}",
                            "weight_scheme": scheme_name,
                            "order_seed": -1,
                            "weighted_fit_used": bool(weighted_used),
                            "n_individuals": int(N_source),
                            "corr_eval": float(eval_result["corr_eval"]),
                            "mse_adj": float(eval_result["mse_adj"]),
                            "pca_dist_obj": float(np.mean(distances)),
                            "n_components": int(n_components),
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
                            "method": f"pca_target_weight_{scheme_name}",
                            "weight_scheme": scheme_name,
                            "n_components": int(n_components),
                            "ringnr": ids_source,
                            "ringnumber": ids_source,
                            "source_island": locality_source.astype(int),
                            "source_island_name": [island_label(int(c), code_to_label) for c in locality_source],
                            "rank": ranks.astype(int),
                            "pca_dist": distances.astype(float),
                            "weight": w.astype(float),
                        })
                        _append_csv(w_df, weights_path)

                    # C) Store full-set selected individuals
                    full_selected_df = pd.DataFrame({
                        "trait": trait_name,
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "n_train_size": int(N_source),
                        "method": "pca_target_topk",
                        "n_components": int(n_components),
                        "ringnr": ids_source,
                        "ringnumber": ids_source,
                        "source_island": locality_source.astype(int),
                        "source_island_name": [island_label(int(c), code_to_label) for c in locality_source],
                        "rank": ranks.astype(int),
                        "pca_dist": distances.astype(float),
                    })
                    _append_csv(full_selected_df, selected_path)

        if results_path.exists():
            all_results = pd.read_csv(results_path)
            summary = (
                all_results.groupby(
                    ["trait", "target_island", "target_island_name", "analysis", "method", "n_components", "n_individuals"],
                    as_index=False,
                )
                .agg(
                    corr_mean=("corr_eval", "mean"),
                    corr_std=("corr_eval", "std"),
                    mse_mean=("mse_adj", "mean"),
                    n_rows=("corr_eval", "size"),
                )
            )
            summary_csv = trait_output / "pca_target_rank_weight_summary.csv"
            summary.to_csv(summary_csv, index=False)

        logger.info("Trait '%s' complete. Output: %s", trait_name, trait_output)


if __name__ == "__main__":
    main()
