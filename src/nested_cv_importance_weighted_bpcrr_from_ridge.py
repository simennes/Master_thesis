from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.cv_utils import island_label
from src.data import load_data
from src.importance_weighting import compute_pc_logistic_importance_weights
from src.nested_cv_avggrm_weighted_unified import (
    _apply_include_islands_filter,
    _bpcrr_one_step_enabled,
    _build_bpcrr_fold_cache,
    _evaluate_bpcrr_from_fold_cache,
    _extract_bpcrr_cfg,
    _parse_bpcrr_prior_settings,
    _parse_selected_splits,
    _prepare_bpcrr_one_step_covariates,
    _result_output_path,
    _write_summary,
)
from src.utils import _select_top_snps_by_abs_corr, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _source_ridge_results_path(config: Dict[str, Any], config_path: Optional[Path]) -> Path:
    raw_path = (
        config.get("source_ridge_results")
        or config.get("source_ridge_results_path")
        or config.get("ridge_results_path")
    )
    if not raw_path:
        raise ValueError(
            "Config must set source_ridge_results to a merged ridge importance-weighted results JSON."
        )

    path = Path(str(raw_path))
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    if config_path is not None:
        repo_relative = config_path.parent.parent / path
        if repo_relative.exists():
            return repo_relative

    return cwd_path


def _load_source_ridge_payload(config: Dict[str, Any], config_path: Optional[Path]) -> tuple[Dict[str, Any], Path]:
    path = _source_ridge_results_path(config, config_path)
    if not path.exists():
        raise FileNotFoundError(f"Source ridge results file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload, path


def _bpcrr_n_components(config: Dict[str, Any]) -> int:
    bpcrr_cfg = _extract_bpcrr_cfg(config)
    if bpcrr_cfg.get("n_components") is not None:
        n_components = int(bpcrr_cfg["n_components"])
    elif config.get("search_space", {}).get("bpcrr", {}).get("n_components_choices"):
        choices = [int(x) for x in config["search_space"]["bpcrr"]["n_components_choices"]]
        n_components = int(choices[0])
    else:
        n_components = int(config.get("n_components", 500))

    if n_components < 1:
        raise ValueError("BPCRR n_components must be >= 1.")
    return n_components


def _best_ridge_by_fold(source_payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for item in source_payload.get("best_params_per_fold", []):
        fold = item.get("fold")
        if fold is not None:
            out[int(fold)] = item
    return out


def _ridge_metrics_by_fold(source_payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for item in source_payload.get("per_fold_metrics", []):
        fold = item.get("fold")
        if fold is not None:
            out[int(fold)] = item
    return out


def _split_result_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    marker = "_splits_"
    if marker not in stem:
        return (10**9, path.name)
    split_text = stem.split(marker, 1)[1].removesuffix("_results")
    first_split = split_text.split("_", 1)[0]
    try:
        return (int(first_split), path.name)
    except ValueError:
        return (10**9, path.name)


def _build_summary(
    *,
    outer_results: list[float],
    selected_set: Optional[set[int]],
    unique_islands: np.ndarray,
    best_params_per_fold: list[dict[str, Any]],
    per_fold_metrics: list[dict[str, Any]],
    source_ridge_path: Path,
    source_ridge_payload: Dict[str, Any],
    n_components: int,
    rr_prior_mode: str,
    rr_va_apriori: Optional[float],
    one_step_enabled: bool,
) -> Dict[str, Any]:
    return {
        "mode": "bpcrr",
        "model_type": "bpcrr",
        "weighting_family": "importance_weighting_from_ridge",
        "cv_strategy": "leave_island_out",
        "outer_test_corr": outer_results,
        "outer_test_corr_mean": float(np.mean(outer_results)) if outer_results else None,
        "outer_test_corr_std": float(np.std(outer_results)) if outer_results else None,
        "inner_strategy": "ridge_winner_reuse",
        "outer_splits": int(len(selected_set)) if selected_set else int(len(unique_islands)),
        "selected_splits": sorted(selected_set) if selected_set else None,
        "best_params_per_fold": best_params_per_fold,
        "per_fold_metrics": per_fold_metrics,
        "importance_weighting_method_choices": list(
            source_ridge_payload.get("importance_weighting_method_choices", [])
        ),
        "source_ridge_results": str(source_ridge_path),
        "source_ridge_outer_test_corr_mean": source_ridge_payload.get("outer_test_corr_mean"),
        "bpcrr_n_components": int(n_components),
        "prior_mode": str(rr_prior_mode),
        "va_apriori": None if rr_va_apriori is None else float(rr_va_apriori),
        "one_step_enabled": bool(one_step_enabled),
    }


def run_merge(config: Dict[str, Any], config_path: Optional[Path] = None) -> tuple[Dict[str, Any], str]:
    base = config["base_train"]
    source_payload, source_path = _load_source_ridge_payload(config, config_path)
    n_components = _bpcrr_n_components(config)
    rr_prior_mode, rr_va_apriori = _parse_bpcrr_prior_settings(config)
    one_step_enabled = _bpcrr_one_step_enabled(config)

    cv_cfg = config.get("cv", {})
    selected_raw = config.get("selected_splits", None)
    if selected_raw is None:
        selected_raw = cv_cfg.get("selected_splits", None)
    selected_set = _parse_selected_splits(selected_raw)

    out_dir = Path(base["paths"].get("output_dir", "outputs/nested_cv"))
    out_name = str(base["paths"].get("output_name", "bpcrr_importance_from_ridge_nested"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if selected_set:
        partial_paths = [
            out_dir / f"{out_name}_splits_{int(split_idx)}_results.json"
            for split_idx in sorted(selected_set)
        ]
        missing = [path for path in partial_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing shard result files: {missing}")
    else:
        partial_paths = sorted(out_dir.glob(f"{out_name}_splits_*_results.json"), key=_split_result_sort_key)
        if not partial_paths:
            raise FileNotFoundError(
                f"No shard result files found matching pattern: {out_dir / (out_name + '_splits_*_results.json')}"
            )

    merged_best_params: list[dict[str, Any]] = []
    merged_fold_metrics: list[dict[str, Any]] = []
    completed_splits: list[int] = []

    for partial_path in partial_paths:
        with open(partial_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        merged_best_params.extend(payload.get("best_params_per_fold", []))
        merged_fold_metrics.extend(payload.get("per_fold_metrics", []))
        completed_splits.extend(int(x.get("fold")) for x in payload.get("per_fold_metrics", []))

    merged_best_params.sort(key=lambda item: int(item.get("fold", 0)))
    merged_fold_metrics.sort(key=lambda item: int(item.get("fold", 0)))
    outer_results = [float(item["test_corr"]) for item in merged_fold_metrics]

    summary = _build_summary(
        outer_results=outer_results,
        selected_set=selected_set,
        unique_islands=np.arange(len(merged_fold_metrics), dtype=int),
        best_params_per_fold=merged_best_params,
        per_fold_metrics=merged_fold_metrics,
        source_ridge_path=source_path,
        source_ridge_payload=source_payload,
        n_components=n_components,
        rr_prior_mode=rr_prior_mode,
        rr_va_apriori=rr_va_apriori,
        one_step_enabled=one_step_enabled,
    )
    summary["merge_info"] = {
        "num_files_merged": len(partial_paths),
        "merged_from": [str(path) for path in partial_paths],
        "completed_splits": sorted(completed_splits),
    }

    out_path = _result_output_path(base["paths"], selected_set=None, model_type="bpcrr")
    _write_summary(summary, out_path)
    logger.info(
        "MERGE DONE. Mean OUTER r = %.4f +- %.4f",
        float(summary["outer_test_corr_mean"]),
        float(summary["outer_test_corr_std"]),
    )
    logger.info("Saved merged summary to: %s", out_path)
    return summary, out_path


def run_importance_weighted_bpcrr_from_ridge(
    config: Dict[str, Any],
    config_path: Optional[Path] = None,
) -> tuple[Dict[str, Any], str]:
    base = config["base_train"]
    cv_cfg = config.get("cv", {})
    source_payload, source_path = _load_source_ridge_payload(config, config_path)
    best_ridge = _best_ridge_by_fold(source_payload)
    ridge_metrics = _ridge_metrics_by_fold(source_payload)

    seed = int(base.get("seed", config.get("seed", 42)))
    set_seed(seed)

    n_components = _bpcrr_n_components(config)
    rr_prior_mode, rr_va_apriori = _parse_bpcrr_prior_settings(config)
    one_step_enabled = _bpcrr_one_step_enabled(config)

    X, y, ids, grm_df, locality, code_to_label, y_eval = load_data(
        base["paths"],
        target_column=base.get("target_column", config.get("target_column", "y_mean")),
        standardize_features=base.get("standardize_features", config.get("standardize_features", False)),
        return_locality=True,
        min_count=int(base.get("min_count", config.get("min_count", 20))),
        return_eval=True,
        eval_target_column=base.get("eval_target_column", config.get("eval_target_column", "y_mean")),
    )
    if y_eval is None:
        y_eval = y.copy()

    X, y, y_eval, ids, locality, grm_df = _apply_include_islands_filter(
        X=X,
        y=y,
        y_eval=y_eval,
        ids=ids,
        locality=locality,
        code_to_label=code_to_label,
        grm_df=grm_df,
        include_islands=cv_cfg.get("include_islands"),
    )
    del grm_df

    one_step_covars = None
    if one_step_enabled:
        one_step_covars = _prepare_bpcrr_one_step_covariates(
            config=config,
            config_path=config_path,
            base_paths=base["paths"],
            ids=ids,
            locality_codes=locality,
            code_to_label=code_to_label,
        )
        if one_step_covars is None:
            raise RuntimeError("BPCRR one_step is enabled but covariates could not be prepared.")

    selected_raw = config.get("selected_splits", None)
    if selected_raw is None:
        selected_raw = cv_cfg.get("selected_splits", None)
    selected_set = _parse_selected_splits(selected_raw)
    if selected_set:
        logger.info("Running only selected outer splits: %s (1-based)", sorted(selected_set))

    strategy = str(cv_cfg.get("strategy", "leave_island_out")).lower()
    if strategy != "leave_island_out":
        raise ValueError("This runner is LOIO-only. Set cv.strategy='leave_island_out'.")

    outer_results: list[float] = []
    best_params_per_fold: list[dict[str, Any]] = []
    per_fold_metrics: list[dict[str, Any]] = []

    unique_islands = np.unique(locality)
    for outer_idx, isl in enumerate(unique_islands, start=1):
        if selected_set and outer_idx not in selected_set:
            continue
        if outer_idx not in best_ridge:
            raise FileNotFoundError(f"Source ridge results do not contain best_params for fold {outer_idx}.")

        idx_outer_train = np.where(locality != isl)[0]
        idx_outer_test = np.where(locality == isl)[0]
        isl_name = island_label(int(isl), code_to_label)
        logger.info("OUTER %d: test_size=%d island=%s (%s)", outer_idx, len(idx_outer_test), isl, isl_name)

        source_best_row = best_ridge[outer_idx]
        source_fold_metrics = ridge_metrics.get(outer_idx, {})
        source_params = dict(source_best_row.get("best_params", {}))
        weight_spec = dict(source_params.get("weighting", {"name": "uniform"}))
        source_weight_name = str(weight_spec.get("name", "uniform")).lower()

        if source_fold_metrics and source_fold_metrics.get("test_island") is not None:
            source_test_island = int(source_fold_metrics["test_island"])
            if source_test_island != int(isl):
                logger.warning(
                    "Fold %d test island differs between source ridge (%s) and BPCRR data (%s).",
                    outer_idx,
                    source_test_island,
                    int(isl),
                )

        snp_cols = None
        if bool(source_params.get("use_snp_selection", False)):
            num_snps = source_params.get("num_snps")
            if num_snps is None:
                raise ValueError(f"Fold {outer_idx}: source ridge used SNP selection but num_snps is missing.")
            k = min(int(num_snps), X.shape[1])
            snp_cols = _select_top_snps_by_abs_corr(X[idx_outer_train], y[idx_outer_train], k)

        weight_result = compute_pc_logistic_importance_weights(
            X=X,
            train_idx=idx_outer_train,
            target_idx=idx_outer_test,
            weight_cfg=weight_spec,
            feature_cols=snp_cols,
        )
        train_weights = None if source_weight_name == "uniform" else weight_result["weights"]

        fold_cache = _build_bpcrr_fold_cache(
            train_idx=idx_outer_train,
            target_idx=idx_outer_test,
            X=X,
            one_step_covars=one_step_covars,
            grm_mat=None,
            max_n_components=n_components,
        )
        if fold_cache is None:
            raise RuntimeError(f"Failed to build BPCRR fold cache for split {outer_idx}.")

        eval_result = _evaluate_bpcrr_from_fold_cache(
            fold_cache=fold_cache,
            y=y,
            y_eval=y_eval,
            n_components=n_components,
            train_weights=train_weights,
            rr_prior_mode=rr_prior_mode,
            rr_va_apriori=rr_va_apriori,
        )

        full_best = {
            "model_type": "bpcrr",
            "n_components": int(n_components),
            "prior_mode": str(rr_prior_mode),
            "va_apriori": None if rr_va_apriori is None else float(rr_va_apriori),
            "one_step_enabled": bool(one_step_enabled),
            "weighting": _jsonable(weight_spec),
            "source_ridge_fold": int(outer_idx),
            "source_ridge_alpha": source_params.get("alpha"),
            "source_ridge_use_snp_selection": bool(source_params.get("use_snp_selection", False)),
            "source_ridge_num_snps": source_params.get("num_snps"),
            "source_ridge_mean_inner_r": source_best_row.get("mean_inner_r"),
            "source_ridge_outer_test_corr": source_fold_metrics.get("test_corr"),
        }
        best_params_per_fold.append(
            {
                "fold": int(outer_idx),
                "best_params": full_best,
                "mean_inner_r": None,
                "source_ridge_mean_inner_r": source_best_row.get("mean_inner_r"),
                "source_ridge_outer_test_corr": source_fold_metrics.get("test_corr"),
            }
        )

        r_test = float(eval_result["corr_eval"])
        outer_results.append(r_test)
        per_fold_metrics.append(
            {
                "fold": int(outer_idx),
                "test_corr": r_test,
                "test_size": int(len(idx_outer_test)),
                "test_island": int(isl),
                "test_island_name": str(isl_name),
                "n_components": int(n_components),
                "prior_mode": str(rr_prior_mode),
                "va_apriori": None if rr_va_apriori is None else float(rr_va_apriori),
                "one_step_enabled": bool(one_step_enabled),
                "weighting": _jsonable(weight_spec),
                "effective_sample_size": float(weight_result["effective_sample_size"]),
                "pre_shrink_effective_sample_size": None
                if weight_result.get("pre_shrink_effective_sample_size") is None
                else float(weight_result["pre_shrink_effective_sample_size"]),
                "importance_n_components_used": int(weight_result["n_components_used"]),
                "source_ridge_fold": int(outer_idx),
                "source_ridge_outer_test_corr": source_fold_metrics.get("test_corr"),
                "source_ridge_mean_inner_r": source_best_row.get("mean_inner_r"),
            }
        )
        logger.info("OUTER %d TEST r = %.4f", outer_idx, r_test)

    out_path = _result_output_path(base["paths"], selected_set=selected_set, model_type="bpcrr")
    summary = _build_summary(
        outer_results=outer_results,
        selected_set=selected_set,
        unique_islands=unique_islands,
        best_params_per_fold=best_params_per_fold,
        per_fold_metrics=per_fold_metrics,
        source_ridge_path=source_path,
        source_ridge_payload=source_payload,
        n_components=n_components,
        rr_prior_mode=rr_prior_mode,
        rr_va_apriori=rr_va_apriori,
        one_step_enabled=one_step_enabled,
    )
    _write_summary(summary, out_path)
    logger.info(
        "DONE. Mean OUTER r = %.4f +- %.4f",
        float(summary["outer_test_corr_mean"]),
        float(summary["outer_test_corr_std"]),
    )
    logger.info("Saved summary to: %s", out_path)
    return summary, out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train BPCRR outer folds using importance-weighting hyperparameters "
            "copied from a merged ridge importance-weighted nested-CV result."
        )
    )
    parser.add_argument("--mode", choices=["worker", "merge"], default="worker")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--selected_splits",
        type=str,
        default=None,
        help="Optional: JSON list or comma-separated 1-based outer split indices to run.",
    )

    args = parser.parse_args()
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.mode == "merge":
        run_merge(cfg, config_path=config_path)
        return

    selected_splits = args.selected_splits
    if selected_splits is None:
        selected_splits = os.environ.get("SLURM_ARRAY_TASK_ID")
    if selected_splits is not None:
        s = selected_splits.strip()
        if s.lower() not in ("false", "none", "", "0"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    cfg.setdefault("cv", {})["selected_splits"] = [int(x) for x in parsed]
                else:
                    cfg.setdefault("cv", {})["selected_splits"] = [int(parsed)]
            except Exception:
                cfg.setdefault("cv", {})["selected_splits"] = [int(x) for x in s.split(",") if x.strip()]

    run_importance_weighted_bpcrr_from_ridge(cfg, config_path=config_path)


if __name__ == "__main__":
    main()
