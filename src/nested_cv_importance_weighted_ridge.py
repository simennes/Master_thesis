from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna

from src.cv_utils import island_label, make_inner_loio_splits
from src.data import load_data
from src.importance_weighting import (
    compute_pc_logistic_importance_weights,
    suggest_importance_weighting_params,
)
from src.training_set_optimization.runner import _evaluate_ridge_subset
from src.utils import _select_top_snps_by_abs_corr, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _default_output_name() -> str:
    return "ridge_importance_weighted_nested"


def _suggest_ridge_params(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    ridge_cfg = search_space.get("ridge", {})
    feature_cfg = search_space.get("feature_selection", {})

    alpha_choices = ridge_cfg.get("alpha_choices")
    if alpha_choices is not None:
        alpha = float(trial.suggest_categorical("alpha", [float(x) for x in alpha_choices]))
    elif "alpha_range" in ridge_cfg:
        alpha_range = ridge_cfg.get("alpha_range", [1e-2, 1e6])
        alpha = float(
            trial.suggest_float(
                "alpha",
                float(alpha_range[0]),
                float(alpha_range[1]),
                log=bool(ridge_cfg.get("alpha_log", True)),
            )
        )
    else:
        alpha_range = ridge_cfg.get("alpha_loguniform", [1e-2, 1e6])
        alpha = float(trial.suggest_float("alpha", float(alpha_range[0]), float(alpha_range[1]), log=True))

    use_snp_selection = bool(
        trial.suggest_categorical(
            "use_snp_selection",
            feature_cfg.get("use_snp_selection_choices", [False, True]),
        )
    )

    num_snps = None
    if use_snp_selection:
        num_snps = int(trial.suggest_int("num_snps", *feature_cfg.get("num_snps_range", (2000, 60000))))

    return {
        "alpha": alpha,
        "use_snp_selection": use_snp_selection,
        "num_snps": num_snps,
    }


def _parse_selected_splits(raw_selected: Any) -> Optional[set[int]]:
    selected_splits: Optional[list[int]]

    if isinstance(raw_selected, (list, tuple, np.ndarray)):
        try:
            selected_splits = [int(x) for x in raw_selected]
        except Exception:
            selected_splits = None
    elif isinstance(raw_selected, str):
        s = raw_selected.strip().lower()
        if s in ("false", "none", "", "0"):
            selected_splits = None
        else:
            try:
                parsed = json.loads(raw_selected)
                if isinstance(parsed, list):
                    selected_splits = [int(x) for x in parsed]
                else:
                    selected_splits = None
            except Exception:
                try:
                    selected_splits = [int(x) for x in raw_selected.split(",") if x.strip()]
                except Exception:
                    selected_splits = None
    else:
        selected_splits = None

    return set(selected_splits) if selected_splits else None


def _apply_include_islands_filter(
    X: np.ndarray,
    y: np.ndarray,
    y_eval: np.ndarray,
    ids: Optional[np.ndarray],
    locality: np.ndarray,
    code_to_label: Optional[Dict[int, str]],
    grm_df,
    include_islands: Any,
):
    if not include_islands:
        return X, y, y_eval, ids, locality, grm_df

    if isinstance(include_islands, (list, tuple, set, np.ndarray)):
        include_list = list(include_islands)
    else:
        include_list = [include_islands]
    include_list = [x.item() if isinstance(x, np.generic) else x for x in include_list]

    label_to_code = {str(v): int(k) for k, v in (code_to_label or {}).items()}
    present_codes = set(np.unique(locality).astype(int).tolist())

    include_codes: set[int] = set()
    for val in include_list:
        sval = str(val)
        if sval in label_to_code:
            include_codes.add(int(label_to_code[sval]))
            continue
        try:
            ival = int(val)
            if ival in present_codes:
                include_codes.add(ival)
        except Exception:
            pass

    if not include_codes:
        available = [f"{c}:{(code_to_label or {}).get(int(c), '?')}" for c in sorted(present_codes)]
        raise ValueError(
            f"include_islands={include_islands} did not match any samples after mapping. "
            f"Available codes/labels: {available}"
        )

    mask = np.isin(locality.astype(int), np.fromiter(include_codes, dtype=int))
    idx = np.where(mask)[0]
    if idx.size == 0:
        available = [f"{c}:{(code_to_label or {}).get(int(c), '?')}" for c in sorted(present_codes)]
        raise ValueError(
            f"include_islands={include_islands} filtered out all samples. "
            f"Matched codes={sorted(include_codes)}. Available codes/labels: {available}"
        )

    X = X[idx]
    y = y[idx]
    y_eval = y_eval[idx]
    ids = ids[idx] if ids is not None else None
    locality = locality[idx]
    if grm_df is not None:
        grm_df = grm_df.iloc[idx, idx]

    kept_codes = sorted(set(locality.astype(int).tolist()))
    kept_labels = [(code_to_label or {}).get(int(c), str(c)) for c in kept_codes]
    logger.info(
        "Filtered to %d samples from islands (codes->labels): %s based on include_islands=%s",
        idx.size,
        ", ".join(f"{c}->{lbl}" for c, lbl in zip(kept_codes, kept_labels)),
        include_islands,
    )

    return X, y, y_eval, ids, locality, grm_df


def _build_summary(
    outer_results: list[float],
    selected_set: Optional[set[int]],
    unique_islands: np.ndarray,
    best_params_per_fold: list[dict[str, Any]],
    per_fold_metrics: list[dict[str, Any]],
    weighting_method_choices: list[str],
    trial_history_per_fold: Optional[list[dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "mode": "ridge",
        "model_type": "ridge",
        "weighting_family": "importance_weighting",
        "cv_strategy": "leave_island_out",
        "outer_test_corr": outer_results,
        "outer_test_corr_mean": float(np.mean(outer_results)) if outer_results else None,
        "outer_test_corr_std": float(np.std(outer_results)) if outer_results else None,
        "inner_strategy": "leave_island_out",
        "outer_splits": int(len(selected_set)) if selected_set else int(len(unique_islands)),
        "selected_splits": sorted(selected_set) if selected_set else None,
        "best_params_per_fold": best_params_per_fold,
        "per_fold_metrics": per_fold_metrics,
        "trial_history_per_fold": trial_history_per_fold or [],
        "importance_weighting_method_choices": weighting_method_choices,
    }


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


def _study_trial_history(study: optuna.Study, fold: int, island: Any, island_name: str) -> dict[str, Any]:
    trials = []
    for trial in study.trials:
        trials.append(
            {
                "number": int(trial.number),
                "state": str(trial.state.name),
                "value": None if trial.value is None else float(trial.value),
                "params": _jsonable(dict(trial.params)),
                "weighting": _jsonable(trial.user_attrs.get("weight_spec")),
                "mean_inner_effective_sample_size": _jsonable(trial.user_attrs.get("mean_inner_ess")),
                "mean_inner_effective_sample_size_threshold": _jsonable(
                    trial.user_attrs.get("mean_inner_ess_threshold")
                ),
                "effective_sample_size_rejected": bool(trial.user_attrs.get("ess_rejected", False)),
            }
        )

    return {
        "fold": int(fold),
        "test_island": None if island is None else int(island),
        "test_island_name": str(island_name),
        "n_trials": int(len(trials)),
        "trials": trials,
    }


def _write_summary(summary: Dict[str, Any], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _result_output_path(base_paths: Dict[str, Any], selected_set: Optional[set[int]]) -> str:
    out_dir = base_paths.get("output_dir", "outputs/nested_cv")
    out_name = base_paths.get("output_name", _default_output_name())
    if selected_set:
        suffix = "splits_" + "_".join(str(i) for i in sorted(selected_set))
        out_name = f"{out_name}_{suffix}"
    return os.path.join(out_dir, f"{out_name}_results.json")


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


def run_merge(config: Dict[str, Any], config_path: Optional[Path] = None) -> tuple[Dict[str, Any], str]:
    del config_path

    base = config["base_train"]
    cv_cfg = config.get("cv", {})
    selected_raw = config.get("selected_splits", None)
    if selected_raw is None:
        selected_raw = cv_cfg.get("selected_splits", None)
    selected_set = _parse_selected_splits(selected_raw)

    out_dir = Path(base["paths"].get("output_dir", "outputs/nested_cv"))
    out_name = str(base["paths"].get("output_name", _default_output_name()))
    out_dir.mkdir(parents=True, exist_ok=True)

    partial_paths: list[Path] = []
    if selected_set:
        for split_idx in sorted(selected_set):
            candidate = out_dir / f"{out_name}_splits_{int(split_idx)}_results.json"
            if not candidate.exists():
                raise FileNotFoundError(f"Expected shard result not found: {candidate}")
            partial_paths.append(candidate)
    else:
        partial_paths = sorted(out_dir.glob(f"{out_name}_splits_*_results.json"), key=_split_result_sort_key)
        if not partial_paths:
            raise FileNotFoundError(
                f"No shard result files found matching pattern: {out_dir / (out_name + '_splits_*_results.json')}"
            )

    merged_best_params: list[dict[str, Any]] = []
    merged_fold_metrics: list[dict[str, Any]] = []
    merged_trial_history: list[dict[str, Any]] = []
    method_choices: Optional[list[str]] = None
    completed_splits: list[int] = []

    for partial_path in partial_paths:
        with open(partial_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        merged_best_params.extend(payload.get("best_params_per_fold", []))
        merged_fold_metrics.extend(payload.get("per_fold_metrics", []))
        merged_trial_history.extend(payload.get("trial_history_per_fold", []))
        if method_choices is None:
            method_choices = list(payload.get("importance_weighting_method_choices", []))
        completed_splits.extend(int(x.get("fold")) for x in payload.get("per_fold_metrics", []))

    merged_best_params.sort(key=lambda item: int(item.get("fold", 0)))
    merged_fold_metrics.sort(key=lambda item: int(item.get("fold", 0)))
    merged_trial_history.sort(key=lambda item: int(item.get("fold", 0)))
    outer_results = [float(item["test_corr"]) for item in merged_fold_metrics]

    unique_islands = np.arange(len(merged_fold_metrics), dtype=int)
    summary = _build_summary(
        outer_results=outer_results,
        selected_set=selected_set,
        unique_islands=unique_islands,
        best_params_per_fold=merged_best_params,
        per_fold_metrics=merged_fold_metrics,
        weighting_method_choices=list(method_choices or []),
        trial_history_per_fold=merged_trial_history,
    )
    summary["merge_info"] = {
        "num_files_merged": len(partial_paths),
        "merged_from": [str(path) for path in partial_paths],
        "completed_splits": sorted(completed_splits),
        "trial_history_folds": len(merged_trial_history),
        "total_logged_trials": int(sum(len(item.get("trials", [])) for item in merged_trial_history)),
    }

    out_path = _result_output_path(base["paths"], selected_set=None)
    _write_summary(summary, out_path)

    mean_r = summary["outer_test_corr_mean"]
    std_r = summary["outer_test_corr_std"]
    if mean_r is not None and std_r is not None:
        logger.info("MERGE DONE. Mean OUTER r = %.4f +- %.4f", mean_r, std_r)
    else:
        logger.info("MERGE DONE. No outer folds were merged or results are empty.")
    logger.info("Saved merged summary to: %s", out_path)
    return summary, out_path


def run_nested_cv_importance_weighted_ridge(
    config: Dict[str, Any],
    config_path: Optional[Path] = None,
) -> tuple[Dict[str, Any], str]:
    del config_path

    base = config["base_train"]
    search_space = config.get("search_space", {})
    weighting_space = search_space.get("importance_weighting", {})
    cv_cfg = config.get("cv", {})

    seed = int(base.get("seed", config.get("seed", 42)))
    set_seed(seed)

    X, y, ids, grm_df, locality, code_to_label, y_eval = load_data(
        base["paths"],
        target_column=base.get("target_column", config.get("target_column", "y_adjusted")),
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

    selected_raw = config.get("selected_splits", None)
    if selected_raw is None:
        selected_raw = cv_cfg.get("selected_splits", None)
    selected_set = _parse_selected_splits(selected_raw)
    n_trials = int(config.get("n_trials", 100))

    raw_method_choices = weighting_space.get("method_choices", ["uniform", "pc_logistic"])
    weighting_method_choices = [str(x).lower() for x in raw_method_choices]

    unique_islands = np.unique(locality)
    strategy = str(cv_cfg.get("strategy", "leave_island_out"))

    best_params_per_fold: list[dict[str, Any]] = []
    per_fold_metrics: list[dict[str, Any]] = []
    trial_history_per_fold: list[dict[str, Any]] = []
    outer_results: list[float] = []

    if selected_set:
        logger.info("Running only selected outer splits: %s (1-based)", sorted(selected_set))

    for outer_idx, isl in enumerate(unique_islands):
        idx_outer_train = np.where(locality != isl)[0]
        idx_outer_test = np.where(locality == isl)[0]
        if selected_set and (outer_idx + 1) not in selected_set:
            continue

        isl_name = island_label(int(isl), code_to_label)
        logger.info("OUTER %d: test_size=%d island=%s (%s)", outer_idx + 1, len(idx_outer_test), isl, isl_name)

        inner_plan = make_inner_loio_splits(locality, idx_outer_train)

        def objective(trial: optuna.Trial) -> float:
            model_params = _suggest_ridge_params(trial, search_space)
            weight_spec = suggest_importance_weighting_params(trial, weighting_space)

            logger.info(
                "Trial %d | outer=%d | alpha=%.3e use_snp_selection=%s num_snps=%s weight=%s",
                trial.number,
                outer_idx + 1,
                float(model_params["alpha"]),
                bool(model_params["use_snp_selection"]),
                model_params["num_snps"],
                weight_spec,
            )

            r_vals: list[float] = []
            ess_vals: list[float] = []
            ess_threshold_vals: list[float] = []

            for in_tr, in_va, in_isl in inner_plan:
                if in_tr.size < 2 or in_va.size == 0:
                    logger.warning(
                        "Skipping inner fold with train=%d val=%d (outer=%d, inner_island=%s)",
                        in_tr.size,
                        in_va.size,
                        outer_idx + 1,
                        in_isl,
                    )
                    continue

                snp_cols = None
                if model_params["use_snp_selection"]:
                    k = min(int(model_params["num_snps"]), X.shape[1])
                    snp_cols = _select_top_snps_by_abs_corr(X[in_tr], y[in_tr], k)

                weight_result = compute_pc_logistic_importance_weights(
                    X=X,
                    train_idx=in_tr,
                    target_idx=in_va,
                    weight_cfg=weight_spec,
                    feature_cols=snp_cols,
                )
                train_weights = None if weight_spec["name"] == "uniform" else weight_result["weights"]

                eval_result = _evaluate_ridge_subset(
                    train_idx=in_tr,
                    X_source=X,
                    y_source=y,
                    X_test=X[in_va],
                    y_test=y[in_va],
                    y_eval_test=y_eval[in_va],
                    alpha=model_params["alpha"],
                    snp_cols=snp_cols,
                    sample_weight=train_weights,
                )

                r_vals.append(float(eval_result["corr_eval"]))
                ess_vals.append(float(weight_result["effective_sample_size"]))
                ess_threshold_vals.append(
                    float(len(in_tr)) * float(weighting_space.get("min_effective_sample_size_frac", 0.0))
                )

            mean_ess = float(np.mean(ess_vals)) if ess_vals else None
            mean_ess_threshold = float(np.mean(ess_threshold_vals)) if ess_threshold_vals else None
            ess_rejected = (
                weight_spec["name"] != "uniform"
                and mean_ess is not None
                and mean_ess_threshold is not None
                and mean_ess_threshold > 0.0
                and mean_ess < mean_ess_threshold
            )
            trial.set_user_attr("mean_inner_ess", mean_ess)
            trial.set_user_attr("mean_inner_ess_threshold", mean_ess_threshold)
            trial.set_user_attr("ess_rejected", ess_rejected)
            if ess_rejected:
                logger.info(
                    "Trial %d rejected by ESS guard: mean_ess=%.2f threshold=%.2f",
                    trial.number,
                    float(mean_ess),
                    float(mean_ess_threshold),
                )
                return -1.0
            return float(np.mean(r_vals)) if r_vals else 0.0

        study = optuna.create_study(
            direction="maximize",
            study_name=f"ridge_importance_inner_outer{outer_idx}",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=bool(config.get("show_progress_bar", True)),
        )

        best = study.best_params
        best_weight_spec = dict(study.best_trial.user_attrs.get("weight_spec", {"name": "uniform"}))
        use_snp_selection = bool(best.get("use_snp_selection", False))
        num_snps = int(best["num_snps"]) if use_snp_selection and best.get("num_snps") is not None else None
        best_mean_inner_ess = study.best_trial.user_attrs.get("mean_inner_ess")
        best_mean_inner_ess_threshold = study.best_trial.user_attrs.get("mean_inner_ess_threshold")

        full_best = {
            "model_type": "ridge",
            "alpha": float(best["alpha"]),
            "use_snp_selection": use_snp_selection,
            "num_snps": num_snps,
            "weighting": best_weight_spec,
            "mean_inner_effective_sample_size": None if best_mean_inner_ess is None else float(best_mean_inner_ess),
            "mean_inner_effective_sample_size_threshold": None
            if best_mean_inner_ess_threshold is None
            else float(best_mean_inner_ess_threshold),
        }

        logger.info(
            "OUTER %d best (inner mean r=%.4f): %s",
            outer_idx + 1,
            study.best_value,
            full_best,
        )

        best_params_per_fold.append(
            {
                "fold": int(outer_idx + 1),
                "best_params": full_best,
                "mean_inner_r": float(study.best_value),
            }
        )
        trial_history_per_fold.append(
            _study_trial_history(
                study=study,
                fold=outer_idx + 1,
                island=isl,
                island_name=isl_name,
            )
        )

        snp_cols = None
        if use_snp_selection:
            k = min(int(num_snps), X.shape[1])
            snp_cols = _select_top_snps_by_abs_corr(X[idx_outer_train], y[idx_outer_train], k)

        final_weight_result = compute_pc_logistic_importance_weights(
            X=X,
            train_idx=idx_outer_train,
            target_idx=idx_outer_test,
            weight_cfg=best_weight_spec,
            feature_cols=snp_cols,
        )
        final_train_weights = None if best_weight_spec.get("name", "uniform") == "uniform" else final_weight_result["weights"]

        eval_result = _evaluate_ridge_subset(
            train_idx=idx_outer_train,
            X_source=X,
            y_source=y,
            X_test=X[idx_outer_test],
            y_test=y[idx_outer_test],
            y_eval_test=y_eval[idx_outer_test],
            alpha=float(best["alpha"]),
            snp_cols=snp_cols,
            sample_weight=final_train_weights,
        )

        per_fold_metrics.append(
            {
                "fold": int(outer_idx + 1),
                "test_corr": float(eval_result["corr_eval"]),
                "test_size": int(len(idx_outer_test)),
                "test_island": None if isl is None else int(isl),
                "test_island_name": str(isl_name),
                "alpha": float(best["alpha"]),
                "use_snp_selection": use_snp_selection,
                "num_snps": num_snps,
                "weighting": best_weight_spec,
                "effective_sample_size": float(final_weight_result["effective_sample_size"]),
                "pre_shrink_effective_sample_size": None
                if final_weight_result.get("pre_shrink_effective_sample_size") is None
                else float(final_weight_result["pre_shrink_effective_sample_size"]),
                "n_components_used": int(final_weight_result["n_components_used"]),
            }
        )

        r_test = float(eval_result["corr_eval"])
        logger.info("OUTER %d TEST r = %.4f", outer_idx + 1, r_test)
        outer_results.append(r_test)

    out_path = _result_output_path(base["paths"], selected_set=selected_set)
    summary = _build_summary(
        outer_results=outer_results,
        selected_set=selected_set,
        unique_islands=unique_islands,
        best_params_per_fold=best_params_per_fold,
        per_fold_metrics=per_fold_metrics,
        weighting_method_choices=weighting_method_choices,
        trial_history_per_fold=trial_history_per_fold,
    )
    summary["cv_strategy"] = strategy
    _write_summary(summary, out_path)

    mean_r = summary["outer_test_corr_mean"]
    std_r = summary["outer_test_corr_std"]
    if mean_r is not None and std_r is not None:
        logger.info("DONE. Mean OUTER r = %.4f +- %.4f", mean_r, std_r)
    else:
        logger.info("DONE. No outer folds were evaluated or results are empty.")
    logger.info("Saved summary to: %s", out_path)
    return summary, out_path


def main():
    parser = argparse.ArgumentParser(
        description="Nested CV (Ridge) with PCA + logistic importance-weight hyperparameter tuning"
    )
    parser.add_argument("--mode", choices=["worker", "merge"], default="worker")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--selected_splits",
        type=str,
        default=None,
        help="Optional: JSON list or comma-separated 1-based outer split indices to run (e.g., '[10,11]' or '10,11'). Use 'false' to disable.",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.selected_splits is not None:
        s = args.selected_splits.strip()
        if s.lower() not in ("false", "none", "", "0"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    cfg.setdefault("cv", {})["selected_splits"] = [int(x) for x in parsed]
                else:
                    vals = [int(x) for x in s.split(",") if x.strip()]
                    cfg.setdefault("cv", {})["selected_splits"] = vals
            except Exception:
                try:
                    vals = [int(x) for x in s.split(",") if x.strip()]
                    cfg.setdefault("cv", {})["selected_splits"] = vals
                except Exception as exc:
                    raise ValueError(
                        "--selected_splits must be a JSON list or comma-separated integers, or 'false'."
                    ) from exc

    if args.mode == "merge":
        run_merge(cfg, config_path=config_path)
        return

    run_nested_cv_importance_weighted_ridge(cfg, config_path=config_path)


if __name__ == "__main__":
    main()
