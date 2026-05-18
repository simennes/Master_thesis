#!/usr/bin/env python
"""
Run BPCRR-INLA weighted full-source analysis.

Per target island and repeat:
    A) Full-source unweighted baseline (once)
    B) Weighted full-source analysis for configured avggrm-based schemes

This runner is intentionally weighting-only: it does not evaluate top-k subset
sizes. Use scripts.run_bpcrr_inla_rank_select for ranked subset studies.

Usage
-----
Worker:
  python -m scripts.run_bpcrr_inla_rank_weight --mode worker --config config/bpcrr_inla_rank_weight_config.json

Merge shards:
  python -m scripts.run_bpcrr_inla_rank_weight --mode merge --config config/bpcrr_inla_rank_weight_config.json
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scripts.run_bpcrr_inla_rank_select import (
    _append_csv,
    _build_trait_specs,
    _inla_bpcrr_predict,
    _long_format_initial_slice,
    _long_format_subset,
    _make_repeat_seed,
    _parse_bpcrr_prior_cfg,
    _parse_n_components_values,
    _prepare_one_step_covariates,
    _resolve_existing_path,
    _resolve_training_islands,
    _set_seed,
    _pearson_corr,
    resolve_island_code,
)
from src.cv_utils import island_label
from src.data import load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _format_label_value(value: float) -> str:
    s = f"{float(value):g}"
    return s.replace("-", "m").replace(".", "p")


def _scheme_label(scheme_cfg: Dict[str, Any]) -> str:
    label = scheme_cfg.get("label", None)
    if label is not None:
        s = str(label).strip()
        if s:
            return s

    name = str(scheme_cfg.get("name", "uniform")).lower()
    if name == "exponential":
        beta = float(scheme_cfg.get("beta", 3.0))
        return f"exponential_b{_format_label_value(beta)}"
    return name


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
    scheme_name = _scheme_label(scheme_cfg)
    floor = float(scheme_cfg.get("floor", 1e-6))
    clip_max = scheme_cfg.get("clip_max", None)

    s = _safe_minmax(avg_grm)
    n = len(avg_grm)

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

    return scheme_name, _normalize_mean_one(w, floor=floor, clip_max=clip_max)


def _parse_weight_schemes(exp_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = exp_cfg.get("weight_schemes", ["uniform", "linear", "minmax", "exponential", "top-heavy"])
    schemes: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            schemes.append({"name": item, "label": str(item)})
        elif isinstance(item, dict) and "name" in item:
            schemes.append(item)
        else:
            raise ValueError("weight_schemes must contain names or objects with a 'name' field")
    if len(schemes) == 0:
        raise ValueError("At least one weight scheme is required")

    labels = [_scheme_label(cfg) for cfg in schemes]
    dup_labels = sorted({lab for lab in labels if labels.count(lab) > 1})
    if dup_labels:
        raise ValueError(
            "weight_schemes must have unique labels. "
            f"Duplicate labels: {dup_labels}. Set a distinct 'label' per scheme."
        )

    return schemes


def _evaluate_bpcrr_subset(
    train_idx: np.ndarray,
    Z_source: np.ndarray,
    y_source: np.ndarray,
    Z_target: np.ndarray,
    y_target: np.ndarray,
    y_eval_target: np.ndarray,
    one_step_source: Optional[Dict[str, np.ndarray]] = None,
    one_step_target: Optional[Dict[str, np.ndarray]] = None,
    rr_prior_mode: str = "default",
    rr_va_apriori: Optional[float] = None,
    train_weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if len(train_idx) < 2:
        return {"corr_eval": 0.0, "mse_adj": float("inf")}

    sample_weights = None
    if train_weights is not None:
        train_weights = np.asarray(train_weights, dtype=float)
        sample_weights = train_weights[np.asarray(train_idx, dtype=np.int64)]

    one_step_train_subset = _long_format_subset(
        one_step_source,
        np.asarray(train_idx, dtype=np.int64),
        n_z_rows=Z_source.shape[0],
    )
    pred = _inla_bpcrr_predict(
        Z_train=Z_source[train_idx],
        y_train=y_source[train_idx],
        Z_test=Z_target,
        train_weights=sample_weights,
        one_step_train=one_step_train_subset,
        one_step_test=one_step_target,
        rr_prior_mode=rr_prior_mode,
        rr_va_apriori=rr_va_apriori,
    )

    corr_eval = float(_pearson_corr(pred, y_eval_target))
    if not np.isfinite(corr_eval):
        corr_eval = 0.0
    mse_adj = float(np.mean((pred - y_target) ** 2))
    return {"corr_eval": corr_eval, "mse_adj": mse_adj}


def run_merge(config_path: Path) -> None:
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    exp_cfg = cfg.get("bpcrr_inla_weight_experiment", {})
    output_dir = Path(exp_cfg.get("output_dir", cfg["paths"].get("output_dir", "outputs/bpcrr_inla_rank_weight")))
    trait_specs = _build_trait_specs(cfg)
    shards_root = output_dir / "shards"

    if not shards_root.exists():
        raise FileNotFoundError(f"No shard directory found: {shards_root}")

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        trait_output = output_dir / trait_name if len(trait_specs) > 1 else output_dir
        trait_output.mkdir(parents=True, exist_ok=True)

        merge_specs = [
            ("bpcrr_inla_rank_weight_results.csv", True),
            ("bpcrr_inla_weighted_individuals.csv", False),
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
                    [
                        "trait",
                        "target_island",
                        "target_island_name",
                        "analysis",
                        "method",
                        "weight_scheme",
                        "n_components",
                        "n_individuals",
                    ],
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    corr_mean=("corr_eval", "mean"),
                    corr_std=("corr_eval", "std"),
                    mse_mean=("mse_adj", "mean"),
                    n_rows=("corr_eval", "size"),
                )
            )
            summary_csv = trait_output / "bpcrr_inla_rank_weight_summary.csv"
            summary.to_csv(summary_csv, index=False)
            logger.info("Wrote summary for trait '%s'", trait_name)

    logger.info("Merge complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="BPCRR-INLA weighted full-source training-set analysis")
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

    exp_cfg = cfg.get("bpcrr_inla_weight_experiment", {})
    output_dir = Path(exp_cfg.get("output_dir", cfg["paths"].get("output_dir", "outputs/bpcrr_inla_rank_weight")))
    output_dir.mkdir(parents=True, exist_ok=True)

    global_seed = int(cfg.get("seed", 42))
    _set_seed(global_seed)

    n_repeats = int(exp_cfg.get("n_repeats", 1))
    if n_repeats < 1:
        raise ValueError("bpcrr_inla_weight_experiment.n_repeats must be >= 1")

    # This runner is weighting-only and ignores top-k ranked subset controls.
    if exp_cfg.get("n_train_sizes", None) is not None or int(exp_cfg.get("n_random_reps", 0)) > 0:
        logger.info("Ignoring n_train_sizes/n_random_reps: weighting-only mode uses full source set only")

    training_islands_raw = exp_cfg.get("training_islands", None)
    weight_schemes = _parse_weight_schemes(exp_cfg)
    bpcrr_prior_mode, bpcrr_va_apriori = _parse_bpcrr_prior_cfg(exp_cfg)
    bpcrr_n_components_values = _parse_n_components_values(
        exp_cfg,
        key="bpcrr_n_components",
        default_value=int(exp_cfg.get("n_components", 20) if exp_cfg.get("n_components", 20) is not None else 20),
    )

    one_step_cfg = exp_cfg.get("one_step", {})
    if isinstance(one_step_cfg, bool):
        one_step_enabled = bool(one_step_cfg)
    else:
        one_step_enabled = bool(one_step_cfg.get("enabled", False))

    trait_specs = _build_trait_specs(cfg)

    shard_index = args.shard_index
    if shard_index is None:
        shard_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    num_shards = args.num_shards
    if num_shards is None:
        env_num_shards = os.environ.get("SWEEP_NUM_SHARDS", None)
        if env_num_shards is not None:
            num_shards = int(env_num_shards)
        else:
            num_shards = max(1, len(weight_schemes))

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}]")

    if num_shards > len(weight_schemes):
        logger.warning(
            "num_shards=%d is larger than number of weight schemes=%d; some shards will be idle",
            num_shards,
            len(weight_schemes),
        )

    scheme_indices_for_shard = [i for i in range(len(weight_schemes)) if i % num_shards == shard_index]
    scheme_labels_for_shard = [_scheme_label(weight_schemes[i]) for i in scheme_indices_for_shard]
    run_baseline_this_shard = shard_index == 0

    logger.info(
        "Shard %d/%d running schemes=%s baseline=%s",
        shard_index,
        num_shards,
        scheme_labels_for_shard,
        str(run_baseline_this_shard),
    )

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        if num_shards > 1:
            trait_output = output_dir / "shards" / f"shard_{shard_index:03d}" / trait_name
        else:
            trait_output = output_dir / trait_name if len(trait_specs) > 1 else output_dir
        trait_output.mkdir(parents=True, exist_ok=True)

        logger.info("Loading data for trait '%s'", trait_name)
        trait_paths = dict(trait_spec["paths"])
        trait_paths["npz"] = str(_resolve_existing_path(config_path, trait_paths.get("npz"), required=True))
        trait_paths["grm_rds"] = str(_resolve_existing_path(config_path, trait_paths.get("grm_rds"), required=True))

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
            raise ValueError("GRM is required for BPCRR weighted analysis. Provide paths.grm_rds in config.")

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

        results_path = trait_output / "bpcrr_inla_rank_weight_results.csv"
        weights_path = trait_output / "bpcrr_inla_weighted_individuals.csv"
        for p in [results_path, weights_path]:
            if p.exists():
                p.unlink()

        done = 0
        total = max(1, len(target_codes) * n_repeats)

        for target_code in target_codes:
            target_name = island_label(target_code, code_to_label)
            source_codes = [c for c in included_island_codes if c != target_code]
            if training_islands_raw is not None:
                training_codes = _resolve_training_islands(training_islands_raw, code_to_label, present_codes, target_code)
                if training_codes is not None:
                    source_codes = [c for c in source_codes if c in training_codes]

            if len(source_codes) == 0:
                continue

            target_mask = locality == target_code
            source_mask = (~target_mask) & np.isin(locality, source_codes)

            X_source = X[source_mask]
            y_source = y[source_mask]
            ids_source = ids[source_mask]
            locality_source = locality[source_mask]

            X_target = X[target_mask]
            y_target = y[target_mask]
            y_eval_target = y_eval[target_mask]
            ids_target = ids[target_mask]

            if len(X_source) < 2 or len(X_target) == 0:
                continue

            max_requested = int(max(bpcrr_n_components_values))
            max_feasible = int(min(max_requested, X_source.shape[0], X_source.shape[1]))
            if max_feasible < 1:
                continue

            pca = PCA(n_components=max_feasible)
            Z_source_full = pca.fit_transform(X_source)
            Z_target_full = pca.transform(X_target)

            one_step_source = None
            one_step_target = None
            if one_step_covars is not None:
                source_inds_global = np.where(source_mask)[0]
                target_inds_global = np.where(target_mask)[0]
                one_step_source = _long_format_initial_slice(
                    one_step_covars, source_inds_global, n_total_inds=len(ids),
                )
                one_step_target = _long_format_initial_slice(
                    one_step_covars, target_inds_global, n_total_inds=len(ids),
                )

            full_idx = np.arange(len(X_source), dtype=np.int64)
            grm_block = GRM_df.loc[ids_source, ids_target].to_numpy(dtype=float)
            avg_grm = np.asarray(grm_block.mean(axis=1), dtype=float)
            order = np.argsort(-avg_grm, kind="mergesort")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)

            for repeat_idx in range(n_repeats):
                done += 1
                repeat_seed = _make_repeat_seed(global_seed, int(target_code), int(repeat_idx))
                logger.info(
                    "Repeat %d/%d | trait=%s target=%s repeat=%d/%d",
                    done,
                    total,
                    trait_name,
                    target_code,
                    repeat_idx + 1,
                    n_repeats,
                )

                for n_comp_req in bpcrr_n_components_values:
                    n_comp = int(min(int(n_comp_req), Z_source_full.shape[1]))
                    if n_comp < 1:
                        continue

                    Z_source = Z_source_full[:, :n_comp]
                    Z_target = Z_target_full[:, :n_comp]
                    eval_kwargs = {
                        "rr_prior_mode": bpcrr_prior_mode,
                        "rr_va_apriori": bpcrr_va_apriori,
                    }

                    # A) Full-source unweighted baseline (written once on shard 0).
                    if run_baseline_this_shard:
                        full_eval = _evaluate_bpcrr_subset(
                            train_idx=full_idx,
                            Z_source=Z_source,
                            y_source=y_source,
                            Z_target=Z_target,
                            y_target=y_target,
                            y_eval_target=y_eval_target,
                            one_step_source=one_step_source,
                            one_step_target=one_step_target,
                            **eval_kwargs,
                        )
                        full_row = {
                            "analysis": "full_baseline",
                            "method": "full_source_unweighted",
                            "weight_scheme": "none",
                            "order_seed": -2,
                            "weighted_fit_used": False,
                            "n_individuals": int(len(X_source)),
                            "corr_eval": float(full_eval["corr_eval"]),
                            "mse_adj": float(full_eval["mse_adj"]),
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "trait": trait_name,
                            "n_components": int(n_comp),
                            "avg_grm_obj": float(np.mean(avg_grm)),
                        }
                        _append_csv(pd.DataFrame([full_row]), results_path)

                    # B) Full-source weighted analysis for this shard's schemes.
                    for scheme_idx in scheme_indices_for_shard:
                        scheme_cfg = weight_schemes[scheme_idx]
                        scheme_name, w = _weights_from_scheme(avg_grm, ranks, scheme_cfg)
                        weighted_used = True
                        try:
                            eval_result = _evaluate_bpcrr_subset(
                                train_idx=full_idx,
                                Z_source=Z_source,
                                y_source=y_source,
                                Z_target=Z_target,
                                y_target=y_target,
                                y_eval_target=y_eval_target,
                                one_step_source=one_step_source,
                                one_step_target=one_step_target,
                                train_weights=w,
                                **eval_kwargs,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Weighted BPCRR fit failed for scheme=%s; falling back to unweighted. Error: %s",
                                scheme_name,
                                exc,
                            )
                            weighted_used = False
                            eval_result = _evaluate_bpcrr_subset(
                                train_idx=full_idx,
                                Z_source=Z_source,
                                y_source=y_source,
                                Z_target=Z_target,
                                y_target=y_target,
                                y_eval_target=y_eval_target,
                                one_step_source=one_step_source,
                                one_step_target=one_step_target,
                                **eval_kwargs,
                            )

                        row = {
                            "analysis": "weighted_full",
                            "method": f"bpcrr_weight_{scheme_name}",
                            "weight_scheme": scheme_name,
                            "order_seed": -1,
                            "weighted_fit_used": bool(weighted_used),
                            "n_individuals": int(len(X_source)),
                            "corr_eval": float(eval_result["corr_eval"]),
                            "mse_adj": float(eval_result["mse_adj"]),
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "trait": trait_name,
                            "n_components": int(n_comp),
                            "avg_grm_obj": float(np.mean(avg_grm)),
                        }
                        _append_csv(pd.DataFrame([row]), results_path)

                        w_df = pd.DataFrame({
                            "trait": trait_name,
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "method": f"bpcrr_weight_{scheme_name}",
                            "weight_scheme": scheme_name,
                            "n_components": int(n_comp),
                            "ringnr": ids_source,
                            "ringnumber": ids_source,
                            "source_island": locality_source.astype(int),
                            "source_island_name": [island_label(int(c), code_to_label) for c in locality_source],
                            "rank": ranks.astype(int),
                            "avg_grm": avg_grm.astype(float),
                            "weight": w.astype(float),
                        })
                        _append_csv(w_df, weights_path)

        if results_path.exists():
            all_results = pd.read_csv(results_path)
            summary = (
                all_results.groupby(
                    [
                        "trait",
                        "target_island",
                        "target_island_name",
                        "analysis",
                        "method",
                        "weight_scheme",
                        "n_components",
                        "n_individuals",
                    ],
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    corr_mean=("corr_eval", "mean"),
                    corr_std=("corr_eval", "std"),
                    mse_mean=("mse_adj", "mean"),
                    n_rows=("corr_eval", "size"),
                )
            )
            summary_csv = trait_output / "bpcrr_inla_rank_weight_summary.csv"
            summary.to_csv(summary_csv, index=False)

        logger.info("Trait '%s' complete. Output: %s", trait_name, trait_output)


if __name__ == "__main__":
    main()
