from __future__ import annotations

import argparse
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
from sklearn.decomposition import PCA

from scripts.run_bpcrr_inla_rank_select import _inla_bpcrr_predict, _prepare_one_step_covariates
from src.avggrm_weighting import (
    compute_avggrm_weights,
    parse_top_k_related_islands,
    rank_inner_validation_islands_by_avg_grm,
    suggest_weighting_params,
)
from src.cv_utils import island_label, make_inner_loio_splits
from src.data import load_data
from src.training_set_optimization.runner import _evaluate_ridge_subset
from src.utils import _pearson_corr, _select_top_snps_by_abs_corr, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _normalize_model_type(raw_value: Any) -> str:
    model_type = str(raw_value or "ridge").strip().lower()
    aliases = {
        "ridge": "ridge",
        "bpcrr": "bpcrr",
        "bpcrr_inla": "bpcrr",
    }
    if model_type not in aliases:
        raise ValueError("model.type must be one of ['ridge', 'bpcrr'].")
    return aliases[model_type]


def _default_output_name(model_type: str) -> str:
    if model_type == "bpcrr":
        return "bpcrr_avggrm_weighted_nested"
    return "ridge_avggrm_weighted_nested"


def _suggest_ridge_params(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    ridge_cfg = search_space.get("ridge", {})
    feature_cfg = search_space.get("feature_selection", {})

    alpha_choices = ridge_cfg.get("alpha_choices", None)
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


def _suggest_bpcrr_params(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    bpcrr_cfg = search_space.get("bpcrr", {})

    n_components_choices = bpcrr_cfg.get("n_components_choices", None)
    if n_components_choices is not None:
        n_components = int(trial.suggest_categorical("n_components", [int(x) for x in n_components_choices]))
    else:
        range_cfg = bpcrr_cfg.get("n_components_range", [20, 500])
        n_components = int(
            trial.suggest_int(
                "n_components",
                int(range_cfg[0]),
                int(range_cfg[1]),
                step=int(bpcrr_cfg.get("n_components_step", 1)),
            )
        )

    return {
        "n_components": n_components,
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


def _extract_bpcrr_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = config.get("model", {})
    merged: Dict[str, Any] = {}
    if isinstance(config.get("bpcrr"), dict):
        merged.update(config["bpcrr"])
    if isinstance(model_cfg.get("bpcrr"), dict):
        merged.update(model_cfg["bpcrr"])

    legacy = config.get("bpcrr_inla_experiment", {})
    if "prior_mode" not in merged and legacy.get("bpcrr_prior_mode") is not None:
        merged["prior_mode"] = legacy.get("bpcrr_prior_mode")
    if "va_apriori" not in merged and merged.get("v_a") is not None:
        merged["va_apriori"] = merged.get("v_a")
    if "va_apriori" not in merged and legacy.get("bpcrr_va_apriori") is not None:
        merged["va_apriori"] = legacy.get("bpcrr_va_apriori")
    if "one_step" not in merged and legacy.get("one_step") is not None:
        merged["one_step"] = legacy.get("one_step")

    return merged


def _parse_bpcrr_prior_settings(config: Dict[str, Any]) -> tuple[str, Optional[float]]:
    legacy = deepcopy(config.get("bpcrr_inla_experiment", {}))
    bpcrr_cfg = _extract_bpcrr_cfg(config)

    prior_mode = bpcrr_cfg.get("prior_mode", legacy.get("bpcrr_prior_mode", "default"))
    prior_mode = str(prior_mode).strip().lower()
    if prior_mode == "fixed":
        prior_mode = "fixed_va"
    legacy["bpcrr_prior_mode"] = prior_mode

    va_apriori = bpcrr_cfg.get("va_apriori", legacy.get("bpcrr_va_apriori", None))
    if va_apriori is not None:
        legacy["bpcrr_va_apriori"] = float(va_apriori)

    valid_modes = {"default", "fixed_va"}
    if prior_mode not in valid_modes:
        raise ValueError("BPCRR prior_mode must be one of ['default', 'fixed', 'fixed_va'].")

    if prior_mode == "fixed_va" and legacy.get("bpcrr_va_apriori", None) is None:
        raise ValueError("BPCRR prior_mode='fixed'/'fixed_va' requires va_apriori (or v_a) > 0.")

    rr_prior_mode = str(legacy["bpcrr_prior_mode"])
    rr_va_apriori = None if legacy.get("bpcrr_va_apriori", None) is None else float(legacy["bpcrr_va_apriori"])
    return rr_prior_mode, rr_va_apriori


def _bpcrr_one_step_enabled(config: Dict[str, Any]) -> bool:
    one_step_cfg = _extract_bpcrr_cfg(config).get("one_step", {})
    if isinstance(one_step_cfg, bool):
        return bool(one_step_cfg)
    return bool(one_step_cfg.get("enabled", False))


def _prepare_bpcrr_one_step_covariates(
    config: Dict[str, Any],
    config_path: Optional[Path],
    base_paths: Dict[str, Any],
    ids: np.ndarray,
    locality_codes: np.ndarray,
    code_to_label: Dict[int, str],
) -> Optional[Dict[str, np.ndarray]]:
    helper_cfg = {
        "paths": dict(base_paths),
        "bpcrr_inla_experiment": {
            "one_step": _extract_bpcrr_cfg(config).get("one_step", {}),
        },
    }
    resolved_config_path = config_path if config_path is not None else (Path.cwd() / "_inline_nested_cv_config.json")
    return _prepare_one_step_covariates(
        config_path=resolved_config_path,
        cfg=helper_cfg,
        ids=ids,
        locality_codes=locality_codes,
        code_to_label=code_to_label,
    )


def _evaluate_bpcrr_from_indices(
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    y_eval: np.ndarray,
    n_components: int,
    train_weights: Optional[np.ndarray] = None,
    one_step_covars: Optional[Dict[str, np.ndarray]] = None,
    rr_prior_mode: str = "default",
    rr_va_apriori: Optional[float] = None,
) -> Dict[str, float]:
    if len(train_idx) < 2 or len(target_idx) == 0:
        return {"corr_eval": 0.0, "mse_adj": float("inf")}

    max_feasible = int(min(int(n_components), len(train_idx), X.shape[1]))
    if max_feasible < 1:
        return {"corr_eval": 0.0, "mse_adj": float("inf")}

    pca = PCA(n_components=max_feasible)
    Z_train = pca.fit_transform(X[train_idx])
    Z_target = pca.transform(X[target_idx])

    sample_weights = None
    if train_weights is not None:
        sample_weights = np.asarray(train_weights, dtype=float)
        if sample_weights.shape[0] != len(train_idx):
            raise ValueError("train_weights length must match train_idx length for BPCRR nested CV.")

    one_step_train = None
    one_step_target = None
    if one_step_covars is not None:
        one_step_train = {
            k: (None if v is None else np.asarray(v)[train_idx])
            for k, v in one_step_covars.items()
        }
        one_step_target = {
            k: (None if v is None else np.asarray(v)[target_idx])
            for k, v in one_step_covars.items()
        }

    pred = _inla_bpcrr_predict(
        Z_train=Z_train,
        y_train=y[train_idx],
        Z_test=Z_target,
        train_weights=sample_weights,
        one_step_train=one_step_train,
        one_step_test=one_step_target,
        rr_prior_mode=rr_prior_mode,
        rr_va_apriori=rr_va_apriori,
    )

    corr_eval = float(_pearson_corr(pred, y_eval[target_idx]))
    if not np.isfinite(corr_eval):
        corr_eval = 0.0
    mse_adj = float(np.mean((pred - y[target_idx]) ** 2))
    return {"corr_eval": corr_eval, "mse_adj": mse_adj}


def _build_summary(
    model_type: str,
    strategy: str,
    outer_results: list[float],
    selected_set: Optional[set[int]],
    unique_islands: np.ndarray,
    best_params_per_fold: list[dict[str, Any]],
    per_fold_metrics: list[dict[str, Any]],
    scheme_choices: list[str],
    inner_top_k_related_islands: Optional[int],
) -> Dict[str, Any]:
    return {
        "mode": model_type,
        "model_type": model_type,
        "cv_strategy": strategy,
        "outer_test_corr": outer_results,
        "outer_test_corr_mean": float(np.mean(outer_results)) if outer_results else None,
        "outer_test_corr_std": float(np.std(outer_results)) if outer_results else None,
        "inner_strategy": "leave_island_out",
        "outer_splits": int(len(selected_set)) if selected_set else int(len(unique_islands)),
        "selected_splits": sorted(selected_set) if selected_set else None,
        "best_params_per_fold": best_params_per_fold,
        "per_fold_metrics": per_fold_metrics,
        "weighting_scheme_choices": scheme_choices,
        "inner_top_k_related_islands": inner_top_k_related_islands,
    }


def _write_summary(summary: Dict[str, Any], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _result_output_path(base_paths: Dict[str, Any], selected_set: Optional[set[int]], model_type: str) -> str:
    out_dir = base_paths.get("output_dir", "outputs/nested_cv")
    out_name = base_paths.get("output_name", _default_output_name(model_type))
    if selected_set:
        suffix = "splits_" + "_".join(str(i) for i in sorted(selected_set))
        out_name = f"{out_name}_{suffix}"
    return os.path.join(out_dir, f"{out_name}_results.json")


def run_merge(config: Dict[str, Any], config_path: Optional[Path] = None) -> tuple[Dict[str, Any], str]:
    del config_path  # reserved for future use

    base = config["base_train"]
    cv_cfg = config.get("cv", {})
    model_type = _normalize_model_type(config.get("model", {}).get("type", "ridge"))
    selected_raw = config.get("selected_splits", None)
    if selected_raw is None:
        selected_raw = cv_cfg.get("selected_splits", None)
    selected_set = _parse_selected_splits(selected_raw)
    inner_top_k_related_islands = parse_top_k_related_islands(cv_cfg.get("inner_top_k_related_islands"))

    out_dir = Path(base["paths"].get("output_dir", "outputs/nested_cv"))
    out_name = str(base["paths"].get("output_name", _default_output_name(model_type)))
    out_dir.mkdir(parents=True, exist_ok=True)

    partial_paths: list[Path] = []
    if selected_set:
        for split_idx in sorted(selected_set):
            candidate = out_dir / f"{out_name}_splits_{int(split_idx)}_results.json"
            if not candidate.exists():
                raise FileNotFoundError(f"Expected shard result not found: {candidate}")
            partial_paths.append(candidate)
    else:
        partial_paths = sorted(out_dir.glob(f"{out_name}_splits_*_results.json"))
        if not partial_paths:
            raise FileNotFoundError(
                f"No shard result files found matching pattern: {out_dir / (out_name + '_splits_*_results.json')}"
            )

    merged_best_params: list[dict[str, Any]] = []
    merged_fold_metrics: list[dict[str, Any]] = []
    scheme_choices: Optional[list[str]] = None
    strategy: Optional[str] = None
    completed_splits: list[int] = []

    for partial_path in partial_paths:
        with open(partial_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        merged_best_params.extend(payload.get("best_params_per_fold", []))
        merged_fold_metrics.extend(payload.get("per_fold_metrics", []))
        if scheme_choices is None:
            scheme_choices = list(payload.get("weighting_scheme_choices", []))
        if strategy is None:
            strategy = str(payload.get("cv_strategy", "leave_island_out"))
        completed_splits.extend(int(x.get("fold")) for x in payload.get("per_fold_metrics", []))

    merged_best_params.sort(key=lambda item: int(item.get("fold", 0)))
    merged_fold_metrics.sort(key=lambda item: int(item.get("fold", 0)))
    outer_results = [float(item["test_corr"]) for item in merged_fold_metrics]

    unique_islands = np.arange(len(merged_fold_metrics), dtype=int)
    summary = _build_summary(
        model_type=model_type,
        strategy=str(strategy or "leave_island_out"),
        outer_results=outer_results,
        selected_set=selected_set,
        unique_islands=unique_islands,
        best_params_per_fold=merged_best_params,
        per_fold_metrics=merged_fold_metrics,
        scheme_choices=list(scheme_choices or []),
        inner_top_k_related_islands=inner_top_k_related_islands,
    )
    summary["merge_info"] = {
        "num_files_merged": len(partial_paths),
        "merged_from": [str(path) for path in partial_paths],
        "completed_splits": sorted(completed_splits),
    }

    out_path = _result_output_path(base["paths"], selected_set=None, model_type=model_type)
    _write_summary(summary, out_path)

    mean_r = summary["outer_test_corr_mean"]
    std_r = summary["outer_test_corr_std"]
    if mean_r is not None and std_r is not None:
        logger.info("MERGE DONE. Mean OUTER r = %.4f +- %.4f", mean_r, std_r)
    else:
        logger.info("MERGE DONE. No outer folds were merged or results are empty.")
    logger.info("Saved merged summary to: %s", out_path)
    return summary, out_path


def run_nested_cv_avggrm_weighted_unified(
    config: Dict[str, Any],
    config_path: Optional[Path] = None,
) -> tuple[Dict[str, Any], str]:
    base = config["base_train"]
    model_type = _normalize_model_type(config.get("model", {}).get("type", "ridge"))
    search_space = config.get("search_space", {})
    weighting_space = search_space.get("weighting", {})
    cv_cfg = config.get("cv", {})
    inner_top_k_related_islands = parse_top_k_related_islands(cv_cfg.get("inner_top_k_related_islands"))

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

    scheme_choices = [
        str(x).lower()
        for x in weighting_space.get("scheme_choices", ["uniform", "linear", "minmax", "exponential", "top-heavy"])
    ]
    non_uniform = {s for s in scheme_choices if s != "uniform"}
    if non_uniform and grm_df is None:
        raise ValueError(
            "AvgGRM weighting requested but GRM was not loaded. "
            "Set base_train.paths.grm_rds (or paths.grm_rds) in your config, "
            "or limit search_space.weighting.scheme_choices to ['uniform']."
        )

    grm_mat = None if grm_df is None else grm_df.to_numpy(dtype=np.float64)
    if inner_top_k_related_islands is not None and grm_mat is None:
        raise ValueError(
            "cv.inner_top_k_related_islands requires a GRM matrix. "
            "Set base_train.paths.grm_rds (or paths.grm_rds) in your config."
        )

    strategy = str(cv_cfg.get("strategy", "leave_island_out")).lower()
    if strategy != "leave_island_out":
        raise ValueError("This runner is LOIO-only. Set cv.strategy='leave_island_out'.")

    selected_raw = config.get("selected_splits", None)
    if selected_raw is None:
        selected_raw = cv_cfg.get("selected_splits", None)
    selected_set = _parse_selected_splits(selected_raw)
    if selected_set:
        logger.info("Running only selected outer splits: %s (1-based)", sorted(selected_set))

    rr_prior_mode = "default"
    rr_va_apriori = None
    one_step_covars = None
    one_step_enabled = False
    if model_type == "bpcrr":
        rr_prior_mode, rr_va_apriori = _parse_bpcrr_prior_settings(config)
        one_step_covars = _prepare_bpcrr_one_step_covariates(
            config=config,
            config_path=config_path,
            base_paths=base["paths"],
            ids=ids,
            locality_codes=locality,
            code_to_label=code_to_label,
        )
        one_step_enabled = _bpcrr_one_step_enabled(config)
        if one_step_enabled and one_step_covars is None:
            raise RuntimeError("BPCRR one_step is enabled but covariates could not be prepared.")
        logger.info(
            "BPCRR prior mode: %s%s",
            rr_prior_mode,
            "" if rr_va_apriori is None else f" (va_apriori={rr_va_apriori:.6g})",
        )

    n_trials = int(config.get("n_trials", 100))
    outer_results: list[float] = []
    best_params_per_fold: list[dict[str, Any]] = []
    per_fold_metrics: list[dict[str, Any]] = []

    unique_islands = np.unique(locality)
    for outer_idx, isl in enumerate(unique_islands):
        idx_outer_train = np.where(locality != isl)[0]
        idx_outer_test = np.where(locality == isl)[0]
        if selected_set and (outer_idx + 1) not in selected_set:
            continue

        isl_name = island_label(isl, code_to_label)
        logger.info("OUTER %d: test_size=%d island=%s (%s)", outer_idx + 1, len(idx_outer_test), isl, isl_name)

        inner_isls = np.unique(locality[idx_outer_train])
        inner_names = [island_label(int(i), code_to_label) for i in inner_isls]
        pairs = ", ".join(f"{int(i)}({n})" for i, n in zip(inner_isls, inner_names))
        logger.info("OUTER %d: inner LOIO validation islands: %s", outer_idx + 1, pairs)

        inner_plan = make_inner_loio_splits(locality, idx_outer_train)
        inner_validation_rankings: list[dict[str, Any]] = []
        requested_inner_top_k = inner_top_k_related_islands
        if inner_top_k_related_islands is not None:
            if grm_mat is None:
                raise RuntimeError("GRM matrix is required for cv.inner_top_k_related_islands")
            inner_validation_rankings = rank_inner_validation_islands_by_avg_grm(
                grm_mat=grm_mat,
                locality=locality,
                idx_outer_train=idx_outer_train,
                idx_outer_test=idx_outer_test,
                code_to_label=code_to_label,
            )
            effective_inner_top_k = min(int(inner_top_k_related_islands), len(inner_validation_rankings))
            if effective_inner_top_k < int(inner_top_k_related_islands):
                logger.info(
                    "OUTER %d: requested top %d related inner islands, but only %d are available; using all available.",
                    outer_idx + 1,
                    int(inner_top_k_related_islands),
                    len(inner_validation_rankings),
                )
            selected_inner_islands = {item["island"] for item in inner_validation_rankings[:effective_inner_top_k]}
            inner_plan = [split for split in inner_plan if int(split[2]) in selected_inner_islands]
            selected_desc = ", ".join(
                f"{item['island']}({item['island_name']}, avgGRM={item['avg_grm_to_outer_test']:.4f})"
                for item in inner_validation_rankings[:effective_inner_top_k]
            )
            logger.info(
                "OUTER %d: using top %d related inner validation islands for tuning: %s",
                outer_idx + 1,
                effective_inner_top_k,
                selected_desc,
            )
        else:
            effective_inner_top_k = None

        def objective(trial: optuna.Trial) -> float:
            if model_type == "bpcrr":
                model_params = _suggest_bpcrr_params(trial, search_space)
                logger.info(
                    "Trial %d | outer=%d | n_components=%s prior_mode=%s weight_search=on",
                    trial.number,
                    outer_idx + 1,
                    model_params["n_components"],
                    rr_prior_mode,
                )
            else:
                model_params = _suggest_ridge_params(trial, search_space)
                logger.info(
                    "Trial %d | outer=%d | alpha=%.3e use_snp_selection=%s num_snps=%s weight_search=on",
                    trial.number,
                    outer_idx + 1,
                    float(model_params["alpha"]),
                    bool(model_params["use_snp_selection"]),
                    model_params["num_snps"],
                )

            weight_spec = suggest_weighting_params(trial, weighting_space)
            logger.info("Trial %d | outer=%d | weight=%s", trial.number, outer_idx + 1, weight_spec)

            r_vals: list[float] = []
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

                train_weights = None
                if weight_spec["name"] != "uniform":
                    if grm_mat is None:
                        raise RuntimeError("GRM matrix is required for non-uniform AvgGRM weighting")
                    _, _, train_weights = compute_avggrm_weights(
                        grm_mat=grm_mat,
                        train_idx=in_tr,
                        target_idx=in_va,
                        scheme_cfg=weight_spec,
                    )

                if model_type == "bpcrr":
                    eval_result = _evaluate_bpcrr_from_indices(
                        train_idx=in_tr,
                        target_idx=in_va,
                        X=X,
                        y=y,
                        y_eval=y_eval,
                        n_components=int(model_params["n_components"]),
                        train_weights=train_weights,
                        one_step_covars=one_step_covars,
                        rr_prior_mode=rr_prior_mode,
                        rr_va_apriori=rr_va_apriori,
                    )
                else:
                    snp_cols = None
                    if model_params["use_snp_selection"]:
                        k = min(int(model_params["num_snps"]), X.shape[1])
                        snp_cols = _select_top_snps_by_abs_corr(X[in_tr], y[in_tr], k)

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

            return float(np.mean(r_vals)) if r_vals else 0.0

        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_type}_inner_outer{outer_idx}",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=bool(config.get("show_progress_bar", True)),
        )

        best = study.best_params
        best_weight_spec = dict(study.best_trial.user_attrs.get("weight_spec", {"name": "uniform"}))

        if model_type == "bpcrr":
            full_best = {
                "model_type": "bpcrr",
                "n_components": int(best["n_components"]),
                "prior_mode": rr_prior_mode,
                "va_apriori": rr_va_apriori,
                "one_step_enabled": bool(one_step_enabled),
                "weighting": best_weight_spec,
            }
        else:
            use_snp_selection = bool(best.get("use_snp_selection", False))
            num_snps = int(best["num_snps"]) if use_snp_selection and best.get("num_snps") is not None else None
            full_best = {
                "model_type": "ridge",
                "alpha": float(best["alpha"]),
                "use_snp_selection": use_snp_selection,
                "num_snps": num_snps,
                "weighting": best_weight_spec,
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
                "inner_validation_top_k_related_islands": effective_inner_top_k,
                "inner_validation_islands": inner_validation_rankings[:effective_inner_top_k]
                if effective_inner_top_k is not None
                else None,
            }
        )

        final_train_weights = None
        if best_weight_spec.get("name", "uniform") != "uniform":
            if grm_mat is None:
                raise RuntimeError("GRM matrix is required for non-uniform AvgGRM weighting")
            _, _, final_train_weights = compute_avggrm_weights(
                grm_mat=grm_mat,
                train_idx=idx_outer_train,
                target_idx=idx_outer_test,
                scheme_cfg=best_weight_spec,
            )

        if model_type == "bpcrr":
            eval_result = _evaluate_bpcrr_from_indices(
                train_idx=idx_outer_train,
                target_idx=idx_outer_test,
                X=X,
                y=y,
                y_eval=y_eval,
                n_components=int(best["n_components"]),
                train_weights=final_train_weights,
                one_step_covars=one_step_covars,
                rr_prior_mode=rr_prior_mode,
                rr_va_apriori=rr_va_apriori,
            )
            per_fold_metrics.append(
                {
                    "fold": int(outer_idx + 1),
                    "test_corr": float(eval_result["corr_eval"]),
                    "test_size": int(len(idx_outer_test)),
                    "test_island": None if isl is None else int(isl),
                    "test_island_name": str(isl_name),
                    "n_components": int(best["n_components"]),
                    "prior_mode": rr_prior_mode,
                    "va_apriori": rr_va_apriori,
                    "one_step_enabled": bool(one_step_enabled),
                    "weighting": best_weight_spec,
                    "inner_validation_top_k_related_islands_requested": requested_inner_top_k,
                    "inner_validation_top_k_related_islands_used": effective_inner_top_k,
                    "inner_validation_islands": inner_validation_rankings[:effective_inner_top_k]
                    if effective_inner_top_k is not None
                    else None,
                }
            )
        else:
            use_snp_selection = bool(best.get("use_snp_selection", False))
            num_snps = int(best["num_snps"]) if use_snp_selection and best.get("num_snps") is not None else None
            snp_cols = None
            if use_snp_selection:
                k = min(int(num_snps), X.shape[1])
                snp_cols = _select_top_snps_by_abs_corr(X[idx_outer_train], y[idx_outer_train], k)

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
                    "inner_validation_top_k_related_islands_requested": requested_inner_top_k,
                    "inner_validation_top_k_related_islands_used": effective_inner_top_k,
                    "inner_validation_islands": inner_validation_rankings[:effective_inner_top_k]
                    if effective_inner_top_k is not None
                    else None,
                }
            )

        r_test = float(eval_result["corr_eval"])
        logger.info("OUTER %d TEST r = %.4f", outer_idx + 1, r_test)
        outer_results.append(r_test)

    out_path = _result_output_path(base["paths"], selected_set=selected_set, model_type=model_type)
    summary = _build_summary(
        model_type=model_type,
        strategy=strategy,
        outer_results=outer_results,
        selected_set=selected_set,
        unique_islands=unique_islands,
        best_params_per_fold=best_params_per_fold,
        per_fold_metrics=per_fold_metrics,
        scheme_choices=scheme_choices,
        inner_top_k_related_islands=inner_top_k_related_islands,
    )
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
        description="Unified nested CV (ridge or BPCRR) with AvgGRM-weight hyperparameter tuning"
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

    if args.mode == "merge":
        run_merge(cfg, config_path=config_path)
        return

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

    run_nested_cv_avggrm_weighted_unified(cfg, config_path=config_path)


if __name__ == "__main__":
    main()
