from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import torch

from src.avggrm_weighting import (
    avg_grm_train_to_target,
    parse_top_k_related_islands,
    ranks_from_desc_scores,
    rank_inner_validation_islands_by_avg_grm,
    suggest_weighting_params,
    weights_from_scheme,
)
from src.cv_utils import island_label, make_inner_loio_splits
from src.data import load_data
from src.hyperparams import suggest_params
from src.importance_weighting import (
    compute_pc_logistic_importance_weights,
    effective_sample_size,
    suggest_importance_weighting_params,
)
from src.models import TrainParams, make_model
from src.nested_cv_avggrm_weighted import (
    _apply_include_islands_filter,
    _parse_selected_splits,
    _train_epochs_weighted,
)
from src.pc_feature import maybe_apply_pca, pca_active, suggest_pc_count
from src.utils import (
    _optimizer,
    _pearson_corr,
    _select_top_snps_by_abs_corr,
    decode_choice,
    set_seed,
)


VALID_WEIGHTING_MODES = ("uniform", "avggrm", "importance")


def _resolve_weighting_mode(config: Dict[str, Any]) -> str:
    """Return one of: uniform | avggrm | importance. Defaults to 'importance'."""
    mode = config.get("weighting_mode")
    if mode is None:
        mode = config.get("base_train", {}).get("weighting_mode")
    if mode is None:
        return "importance"  # back-compat: behave like before this change
    mode = str(mode).strip().lower()
    if mode not in VALID_WEIGHTING_MODES:
        raise ValueError(
            f"weighting_mode must be one of {list(VALID_WEIGHTING_MODES)}, got {mode!r}"
        )
    return mode


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _default_output_name() -> str:
    return "mlp_importance_weighted_nested"


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


def _write_summary(summary: Dict[str, Any], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _build_summary(
    *,
    outer_results: list[float],
    selected_set: Optional[set[int]],
    unique_islands: np.ndarray,
    best_params_per_fold: list[dict[str, Any]],
    per_fold_metrics: list[dict[str, Any]],
    weighting_method_choices: list[str],
    trial_history_per_fold: Optional[list[dict[str, Any]]] = None,
    inner_top_k_related_islands: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "mode": "mlp",
        "model_type": "mlp",
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
        "inner_top_k_related_islands": inner_top_k_related_islands,
    }


def _study_trial_history(study: optuna.Study, fold: int, island: Any, island_name: str) -> dict[str, Any]:
    trials = []
    for trial in study.trials:
        params = dict(trial.params)
        if "hidden_dims" in params:
            try:
                params["hidden_dims"] = decode_choice(params["hidden_dims"])
            except Exception:
                pass
        trials.append(
            {
                "number": int(trial.number),
                "state": str(trial.state.name),
                "value": None if trial.value is None else float(trial.value),
                "params": _jsonable(params),
                "weighting": _jsonable(trial.user_attrs.get("weight_spec")),
                "inner_fold_scores": _jsonable(trial.user_attrs.get("inner_fold_scores")),
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


def _decode_best_params(best: dict[str, Any]) -> dict[str, Any]:
    best_decoded = dict(best)
    if "hidden_dims" in best_decoded:
        try:
            best_decoded["hidden_dims"] = decode_choice(best_decoded["hidden_dims"])
        except Exception:
            pass
    return best_decoded


def _train_params_from_best(best: dict[str, Any]) -> TrainParams:
    hidden_dims = best.get("hidden_dims")
    if isinstance(hidden_dims, str):
        hidden_dims = json.loads(hidden_dims)
    return TrainParams(
        lr=float(best.get("lr")),
        weight_decay=float(best.get("weight_decay")),
        epochs=int(best.get("epochs")),
        loss_name=str(best.get("loss")),
        optimizer=str(best.get("optimizer")),
        hidden_dims=hidden_dims,
        dropout=float(best.get("dropout")),
        batch_norm=bool(best.get("batch_norm")),
    )


def _importance_weighting_method_choices(weighting_space: Dict[str, Any]) -> list[str]:
    raw_method_choices = weighting_space.get("method_choices", ["uniform", "pc_logistic"])
    method_choices = [str(x).lower() for x in raw_method_choices]
    if not method_choices:
        raise ValueError("search_space.importance_weighting.method_choices must contain at least one method")
    return method_choices


def _is_uniform_only_importance_weighting(method_choices: list[str]) -> bool:
    return all(method == "uniform" for method in method_choices)


def _uniform_importance_weight_result(n_train: int) -> dict[str, Any]:
    weights = np.ones(int(n_train), dtype=float)
    return {
        "weights": weights,
        "raw_weights": weights.copy(),
        "target_prob_train": np.full(int(n_train), 0.5, dtype=float),
        "effective_sample_size": effective_sample_size(weights),
        "n_components_used": 0,
        "pre_shrink_effective_sample_size": None,
    }


def _importance_weight_result(
    *,
    X: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    weight_cfg: Dict[str, Any],
    feature_cols: Optional[np.ndarray] = None,
    precomputed_pcs: bool = False,
) -> dict[str, Any]:
    if str(weight_cfg.get("name", "uniform")).lower() == "uniform":
        return _uniform_importance_weight_result(len(train_idx))

    return compute_pc_logistic_importance_weights(
        X=X,
        train_idx=train_idx,
        target_idx=target_idx,
        weight_cfg=weight_cfg,
        feature_cols=feature_cols,
        precomputed_pcs=precomputed_pcs,
    )


def _avggrm_weight_result(
    *,
    grm_mat: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    weight_cfg: Dict[str, Any],
) -> dict[str, Any]:
    """Compute avgGRM-weighting result with the same shape as importance result."""
    if str(weight_cfg.get("name", "uniform")).lower() == "uniform":
        return _uniform_importance_weight_result(len(train_idx))

    avg_grm = avg_grm_train_to_target(grm_mat, train_idx, target_idx)
    ranks = ranks_from_desc_scores(avg_grm)
    weights = weights_from_scheme(avg_grm, ranks, weight_cfg)
    return {
        "weights": weights,
        "raw_weights": weights.copy(),
        "target_prob_train": np.full(len(train_idx), float("nan"), dtype=float),
        "effective_sample_size": effective_sample_size(weights),
        "n_components_used": 0,
        "pre_shrink_effective_sample_size": None,
    }


def _load_outer_split_ids(config: Dict[str, Any]) -> list[int]:
    base = config["base_train"]
    cv_cfg = config.get("cv", {})

    paths = dict(base["paths"])
    paths.pop("grm_rds", None)

    X, y, ids, _, locality, code_to_label, y_eval = load_data(
        paths,
        target_column=base.get("target_column", config.get("target_column", "y_adjusted")),
        standardize_features=base.get("standardize_features", config.get("standardize_features", False)),
        return_locality=True,
        min_count=int(base.get("min_count", config.get("min_count", 20))),
        return_eval=True,
        eval_target_column=base.get("eval_target_column", config.get("eval_target_column", "y_mean")),
    )
    if y_eval is None:
        y_eval = y.copy()

    _, _, _, _, locality, _ = _apply_include_islands_filter(
        X=X,
        y=y,
        y_eval=y_eval,
        ids=ids,
        locality=locality,
        code_to_label=code_to_label,
        grm_df=None,
        include_islands=cv_cfg.get("include_islands"),
    )

    selected_raw = config.get("selected_splits", None)
    if selected_raw is None:
        selected_raw = cv_cfg.get("selected_splits", None)
    selected_set = _parse_selected_splits(selected_raw)

    unique_islands = np.unique(locality)
    return [idx + 1 for idx in range(len(unique_islands)) if not selected_set or (idx + 1) in selected_set]


def _parse_gpu_id_tokens(raw_value: Optional[str]) -> list[str]:
    if raw_value is None:
        return []

    raw = str(raw_value).strip()
    if not raw or raw.lower() in {"none", "(null)", "null", "n/a"}:
        return []

    tokens: list[str] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if token.isdigit():
            tokens.append(token)
            continue

        if "-" in token:
            left, _, right = token.partition("-")
            if left.isdigit() and right.isdigit():
                start = int(left)
                stop = int(right)
                if start <= stop:
                    tokens.extend(str(i) for i in range(start, stop + 1))
                    continue

        tokens.append(token)

    return tokens


def _visible_gpu_ids() -> list[str]:
    for env_name in ("CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        raw = os.environ.get(env_name)
        tokens = _parse_gpu_id_tokens(raw)
        if tokens:
            logger.info("Resolved GPU worker ids from %s=%s", env_name, ",".join(tokens))
            return tokens

    device_count = torch.cuda.device_count()
    if device_count > 1:
        logger.warning(
            "CUDA_VISIBLE_DEVICES, SLURM_STEP_GPUS, and SLURM_JOB_GPUS were all unset. "
            "Falling back to the first %d visible CUDA devices; on shared nodes this may not match your SLURM allocation.",
            device_count,
        )
    return [str(i) for i in range(device_count)]


def _run_parallel_outer_splits(config: Dict[str, Any], config_path: str):
    parallel_cfg = config.get("parallel_outer_splits", {})
    if not bool(parallel_cfg.get("enabled", True)):
        run_nested_cv_importance_weighted_mlp(config)
        return

    split_ids = _load_outer_split_ids(config)
    gpu_ids = _visible_gpu_ids()
    max_workers_cfg = parallel_cfg.get("max_concurrent_gpus", None)
    if max_workers_cfg is not None:
        try:
            gpu_ids = gpu_ids[: max(1, int(max_workers_cfg))]
        except Exception:
            pass

    if len(gpu_ids) <= 1 or len(split_ids) <= 1:
        run_nested_cv_importance_weighted_mlp(config)
        return

    logger.info(
        "Parallel outer-split execution enabled: %d split(s) across %d GPU worker(s)",
        len(split_ids),
        min(len(split_ids), len(gpu_ids)),
    )

    pending = list(split_ids)
    active: list[dict[str, Any]] = []

    def launch(split_id: int, gpu_id: str):
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        child_env["NESTED_CV_DISABLE_PARALLEL_OUTER"] = "1"
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        cmd = [
            sys.executable,
            "-m",
            "src.nested_cv_importance_weighted_mlp",
            "--config",
            config_path,
            "--selected_splits",
            str(split_id),
            "--worker_mode",
        ]
        out_path = _result_output_path(config["base_train"]["paths"], {int(split_id)})
        logger.info("Launching outer split %d on GPU %s", split_id, gpu_id)
        proc = subprocess.Popen(cmd, env=child_env, cwd=os.getcwd())
        active.append({"proc": proc, "split_id": split_id, "gpu_id": gpu_id, "out_path": out_path})

    while pending or active:
        active_gpu_ids = {str(job["gpu_id"]) for job in active}
        free_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id not in active_gpu_ids]
        while pending and free_gpu_ids:
            launch(pending.pop(0), free_gpu_ids.pop(0))

        time.sleep(1.0)
        still_running: list[dict[str, Any]] = []
        for job in active:
            return_code = job["proc"].poll()
            if return_code is None:
                still_running.append(job)
                continue

            split_id = int(job["split_id"])
            gpu_id = str(job["gpu_id"])
            if return_code != 0:
                for other in active:
                    if other["proc"].poll() is None:
                        other["proc"].terminate()
                raise RuntimeError(
                    f"Parallel outer worker failed for split {split_id} on GPU {gpu_id} "
                    f"with exit code {return_code}."
                )

            if not os.path.exists(job["out_path"]):
                raise FileNotFoundError(
                    f"Expected worker result for split {split_id} at '{job['out_path']}', but it was not created."
                )

            logger.info("Completed outer split %d on GPU %s", split_id, gpu_id)

        active = still_running

    summary, out_path = run_merge(config)
    summary["parallel_outer_execution"] = {
        "enabled": True,
        "gpu_workers": min(len(split_ids), len(gpu_ids)),
        "completed_splits": [int(item["fold"]) for item in summary.get("per_fold_metrics", [])],
    }
    _write_summary(summary, out_path)


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
    inner_top_k_related_islands: Optional[int] = None

    for partial_path in partial_paths:
        with open(partial_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        merged_best_params.extend(payload.get("best_params_per_fold", []))
        merged_fold_metrics.extend(payload.get("per_fold_metrics", []))
        merged_trial_history.extend(payload.get("trial_history_per_fold", []))
        if method_choices is None:
            method_choices = list(payload.get("importance_weighting_method_choices", []))
        if inner_top_k_related_islands is None:
            inner_top_k_related_islands = payload.get("inner_top_k_related_islands")
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
        inner_top_k_related_islands=inner_top_k_related_islands,
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


def run_nested_cv_importance_weighted_mlp(
    config: Dict[str, Any],
    config_path: Optional[Path] = None,
) -> tuple[Dict[str, Any], str]:
    del config_path

    base = config["base_train"]
    search_space = config.get("search_space", {})
    weighting_mode = _resolve_weighting_mode(config)
    weighting_space = (
        search_space.get("importance_weighting", {})
        if weighting_mode == "importance"
        else search_space.get("weighting", {})
    )
    if weighting_mode == "importance":
        weighting_method_choices = _importance_weighting_method_choices(weighting_space)
        uniform_only_weighting = _is_uniform_only_importance_weighting(weighting_method_choices)
    elif weighting_mode == "avggrm":
        weighting_method_choices = [
            str(x).lower()
            for x in weighting_space.get(
                "scheme_choices",
                ["uniform", "linear", "minmax", "exponential", "top-heavy"],
            )
        ]
        uniform_only_weighting = all(m == "uniform" for m in weighting_method_choices)
    else:  # uniform
        weighting_method_choices = ["uniform"]
        uniform_only_weighting = True
    cv_cfg = config.get("cv", {})
    inner_top_k_related_islands = parse_top_k_related_islands(cv_cfg.get("inner_top_k_related_islands"))

    seed = int(base.get("seed", config.get("seed", 42)))
    set_seed(seed)

    logger.info("weighting_mode = %s (choices=%s)", weighting_mode, weighting_method_choices)
    if weighting_mode == "uniform" or uniform_only_weighting:
        logger.info("Weights will be uniform; weight-search hyperparameters are not sampled.")

    data_paths = dict(base["paths"])
    needs_grm = (inner_top_k_related_islands is not None) or (weighting_mode == "avggrm")
    if not needs_grm:
        data_paths.pop("grm_rds", None)

    X, y, ids, grm_df, locality, code_to_label, y_eval = load_data(
        data_paths,
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

    pca_state = maybe_apply_pca(X, config)
    if pca_active(pca_state):
        X = pca_state["Z"]
        logger.info(
            "Using PC features for MLP: r_fit=%d (cumvar=%.4f). "
            "Trials will pick top-k PCs from this set.",
            pca_state["n_pcs_fit"], pca_state.get("cumvar_explained", float("nan")),
        )

    grm_mat = None
    if grm_df is not None:
        grm_mat = grm_df.to_numpy(dtype=np.float64)
    if inner_top_k_related_islands is not None and grm_mat is None:
        raise ValueError(
            "cv.inner_top_k_related_islands requires a GRM matrix. "
            "Set base_train.paths.grm_rds (or paths.grm_rds) in your config."
        )
    if weighting_mode == "avggrm" and grm_mat is None:
        raise ValueError(
            "weighting_mode='avggrm' requires a GRM matrix. "
            "Set base_train.paths.grm_rds (or paths.grm_rds) in your config."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    strategy = str(cv_cfg.get("strategy", "leave_island_out")).lower()
    if strategy != "leave_island_out":
        raise ValueError("This runner is LOIO-only. Set cv.strategy='leave_island_out'.")

    selected_raw = config.get("selected_splits", None)
    if selected_raw is None:
        selected_raw = cv_cfg.get("selected_splits", None)
    selected_set = _parse_selected_splits(selected_raw)
    if selected_set:
        logger.info("Running only selected outer splits: %s (1-based)", sorted(selected_set))

    n_trials = int(config.get("n_trials", 100))
    enable_pruning = bool(config.get("enable_pruning", True))
    pruner = (
        optuna.pruners.MedianPruner(n_warmup_steps=int(config.get("pruner_warmup_epochs", 5)))
        if enable_pruning
        else optuna.pruners.NopPruner()
    )

    outer_results: list[float] = []
    best_params_per_fold: list[dict[str, Any]] = []
    per_fold_metrics: list[dict[str, Any]] = []
    trial_history_per_fold: list[dict[str, Any]] = []

    unique_islands = np.unique(locality)
    for outer_idx, isl in enumerate(unique_islands):
        idx_outer_train = np.where(locality != isl)[0]
        idx_outer_test = np.where(locality == isl)[0]
        if selected_set and (outer_idx + 1) not in selected_set:
            continue

        isl_name = island_label(int(isl), code_to_label)
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
            tp = suggest_params(trial, search_space)
            if uniform_only_weighting or weighting_mode == "uniform":
                weight_spec = {"name": "uniform"}
                trial.set_user_attr("weight_spec", weight_spec)
            elif weighting_mode == "avggrm":
                weight_spec = suggest_weighting_params(trial, weighting_space)
            else:  # importance
                weight_spec = suggest_importance_weighting_params(trial, weighting_space)

            n_pcs_trial: Optional[int] = None
            if pca_active(pca_state):
                n_pcs_trial = suggest_pc_count(trial, pca_state)

            hidden_repr = list(tp.hidden_dims) if tp.hidden_dims else None
            logger.info(
                "Trial %d | outer=%d | hidden=%s epochs=%s lr=%.3e wd=%.3e n_pcs=%s weight=%s",
                trial.number,
                outer_idx + 1,
                hidden_repr,
                tp.epochs,
                tp.lr,
                tp.weight_decay,
                n_pcs_trial if n_pcs_trial is not None else "n/a",
                weight_spec,
            )

            r_vals: list[float] = []
            inner_fold_scores: list[dict[str, Any]] = []
            ess_vals: list[float] = []
            ess_threshold_vals: list[float] = []

            for step_idx, (in_tr, in_va, in_isl) in enumerate(inner_plan):
                if in_tr.size < 2 or in_va.size == 0:
                    logger.warning(
                        "Skipping inner fold with train=%d val=%d (outer=%d, inner_island=%s)",
                        in_tr.size,
                        in_va.size,
                        outer_idx + 1,
                        in_isl,
                    )
                    continue

                cols: Any = slice(None)
                feature_cols = None
                if n_pcs_trial is not None:
                    k = min(int(n_pcs_trial), X.shape[1])
                    feature_cols = np.arange(k, dtype=np.int64)
                    cols = feature_cols
                elif bool(trial.params.get("use_snp_selection", False)):
                    k = int(trial.params.get("num_snps", X.shape[1]))
                    feature_cols = _select_top_snps_by_abs_corr(X[in_tr], y[in_tr], min(k, X.shape[1]))
                    cols = feature_cols

                if weighting_mode == "avggrm":
                    weight_result = _avggrm_weight_result(
                        grm_mat=grm_mat,
                        train_idx=in_tr,
                        target_idx=in_va,
                        weight_cfg=weight_spec,
                    )
                else:
                    weight_result = _importance_weight_result(
                        X=X,
                        train_idx=in_tr,
                        target_idx=in_va,
                        weight_cfg=weight_spec,
                        feature_cols=feature_cols,
                        precomputed_pcs=pca_active(pca_state),
                    )
                train_weights = None if weight_spec.get("name", "uniform") == "uniform" else weight_result["weights"]

                X_tr, X_va = X[in_tr][:, cols], X[in_va][:, cols]
                x_tr = torch.from_numpy(X_tr).to(device)
                y_tr_t = torch.from_numpy(y[in_tr]).to(device).float()
                x_va = torch.from_numpy(X_va).to(device)

                model = make_model(in_dim=X_tr.shape[1], tp=tp).to(device)
                opt = _optimizer(tp.optimizer, model.parameters(), tp.lr, tp.weight_decay)

                _train_epochs_weighted(
                    model=model,
                    x=x_tr,
                    y=y_tr_t,
                    epochs=tp.epochs,
                    opt=opt,
                    loss_name=tp.loss_name,
                    sample_weight=train_weights,
                )

                model.eval()
                with torch.no_grad():
                    yhat_va = model(x_va).detach().cpu().numpy().ravel()
                inner_r = _pearson_corr(y_eval[in_va], yhat_va)
                r_vals.append(inner_r)
                inner_fold_scores.append(
                    {
                        "inner_island": None if in_isl is None else int(in_isl),
                        "inner_island_name": island_label(int(in_isl), code_to_label),
                        "validation_size": int(len(in_va)),
                        "score": float(inner_r),
                        "effective_sample_size": float(weight_result["effective_sample_size"]),
                        "n_components_used": int(weight_result["n_components_used"]),
                    }
                )
                ess_vals.append(float(weight_result["effective_sample_size"]))
                ess_threshold_vals.append(
                    float(len(in_tr)) * float(weighting_space.get("min_effective_sample_size_frac", 0.0))
                )

                should_prune = False
                if r_vals:
                    trial.report(float(np.mean(r_vals)), step=step_idx)
                    should_prune = bool(trial.should_prune())

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if should_prune:
                    raise optuna.TrialPruned()

            mean_ess = float(np.mean(ess_vals)) if ess_vals else None
            mean_ess_threshold = float(np.mean(ess_threshold_vals)) if ess_threshold_vals else None
            ess_rejected = (
                weight_spec.get("name", "uniform") != "uniform"
                and mean_ess is not None
                and mean_ess_threshold is not None
                and mean_ess_threshold > 0.0
                and mean_ess < mean_ess_threshold
            )
            trial.set_user_attr("mean_inner_ess", mean_ess)
            trial.set_user_attr("mean_inner_ess_threshold", mean_ess_threshold)
            trial.set_user_attr("ess_rejected", ess_rejected)
            trial.set_user_attr("inner_fold_scores", inner_fold_scores)
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
            study_name=f"mlp_importance_inner_outer{outer_idx}",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=pruner,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=bool(config.get("show_progress_bar", True)),
        )

        best = study.best_params
        best_decoded = _decode_best_params(best)
        best_weight_spec = dict(study.best_trial.user_attrs.get("weight_spec", {"name": "uniform"}))
        best_mean_inner_ess = study.best_trial.user_attrs.get("mean_inner_ess")
        best_mean_inner_ess_threshold = study.best_trial.user_attrs.get("mean_inner_ess_threshold")

        full_best = dict(best_decoded)
        full_best["model_type"] = "mlp"
        full_best["weighting"] = best_weight_spec
        full_best["mean_inner_effective_sample_size"] = (
            None if best_mean_inner_ess is None else float(best_mean_inner_ess)
        )
        full_best["mean_inner_effective_sample_size_threshold"] = (
            None if best_mean_inner_ess_threshold is None else float(best_mean_inner_ess_threshold)
        )

        logger.info(
            "OUTER %d best (inner mean r=%.4f): model=%s weighting=%s",
            outer_idx + 1,
            study.best_value,
            {k: v for k, v in best_decoded.items() if k != "weighting"},
            best_weight_spec,
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
        trial_history_per_fold.append(
            _study_trial_history(
                study=study,
                fold=outer_idx + 1,
                island=isl,
                island_name=isl_name,
            )
        )

        tp_final = _train_params_from_best(best)

        cols: Any = slice(None)
        feature_cols = None
        n_pcs_best: Optional[int] = None
        if pca_active(pca_state) and "n_pcs" in best:
            n_pcs_best = int(best["n_pcs"])
            k = min(n_pcs_best, X.shape[1])
            feature_cols = np.arange(k, dtype=np.int64)
            cols = feature_cols
        elif bool(best.get("use_snp_selection", False)):
            k = int(best.get("num_snps", X.shape[1]))
            feature_cols = _select_top_snps_by_abs_corr(
                X[idx_outer_train],
                y[idx_outer_train],
                min(k, X.shape[1]),
            )
            cols = feature_cols

        if weighting_mode == "avggrm":
            final_weight_result = _avggrm_weight_result(
                grm_mat=grm_mat,
                train_idx=idx_outer_train,
                target_idx=idx_outer_test,
                weight_cfg=best_weight_spec,
            )
        else:
            final_weight_result = _importance_weight_result(
                X=X,
                train_idx=idx_outer_train,
                target_idx=idx_outer_test,
                weight_cfg=best_weight_spec,
                feature_cols=feature_cols,
                precomputed_pcs=pca_active(pca_state),
            )
        final_train_weights = (
            None if best_weight_spec.get("name", "uniform") == "uniform" else final_weight_result["weights"]
        )

        X_tr, X_te = X[idx_outer_train][:, cols], X[idx_outer_test][:, cols]
        x_tr = torch.from_numpy(X_tr).to(device)
        y_tr_t = torch.from_numpy(y[idx_outer_train]).to(device).float()
        x_te = torch.from_numpy(X_te).to(device)

        model = make_model(in_dim=X_tr.shape[1], tp=tp_final).to(device)
        opt = _optimizer(tp_final.optimizer, model.parameters(), tp_final.lr, tp_final.weight_decay)
        _train_epochs_weighted(
            model=model,
            x=x_tr,
            y=y_tr_t,
            epochs=tp_final.epochs,
            opt=opt,
            loss_name=tp_final.loss_name,
            sample_weight=final_train_weights,
        )

        model.eval()
        with torch.no_grad():
            yhat_te = model(x_te).detach().cpu().numpy().ravel()
        r_test = _pearson_corr(y_eval[idx_outer_test], yhat_te)

        logger.info("OUTER %d TEST r = %.4f", outer_idx + 1, r_test)
        outer_results.append(float(r_test))
        per_fold_metrics.append(
            {
                "fold": int(outer_idx + 1),
                "test_corr": float(r_test),
                "test_size": int(len(idx_outer_test)),
                "test_island": None if isl is None else int(isl),
                "test_island_name": str(isl_name),
                "model_type": "pc_mlp" if pca_active(pca_state) else "mlp",
                "weighting_mode": weighting_mode,
                "n_pcs": n_pcs_best,
                "hidden_dims": _jsonable(tp_final.hidden_dims),
                "dropout": float(tp_final.dropout),
                "batch_norm": bool(tp_final.batch_norm),
                "lr": float(tp_final.lr),
                "weight_decay": float(tp_final.weight_decay),
                "epochs": int(tp_final.epochs),
                "loss": str(tp_final.loss_name),
                "optimizer": str(tp_final.optimizer),
                "use_snp_selection": bool(best.get("use_snp_selection", False)) and n_pcs_best is None,
                "num_snps": None if feature_cols is None else int(len(feature_cols)),
                "weighting": best_weight_spec,
                "effective_sample_size": float(final_weight_result["effective_sample_size"]),
                "pre_shrink_effective_sample_size": None
                if final_weight_result.get("pre_shrink_effective_sample_size") is None
                else float(final_weight_result["pre_shrink_effective_sample_size"]),
                "n_components_used": int(final_weight_result["n_components_used"]),
                "inner_validation_top_k_related_islands_requested": requested_inner_top_k,
                "inner_validation_top_k_related_islands_used": effective_inner_top_k,
                "inner_validation_islands": inner_validation_rankings[:effective_inner_top_k]
                if effective_inner_top_k is not None
                else None,
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Incremental persistence: write the JSON after each completed outer fold.
        out_path = _result_output_path(base["paths"], selected_set=selected_set)
        partial_summary = _build_summary(
            outer_results=outer_results,
            selected_set=selected_set,
            unique_islands=unique_islands,
            best_params_per_fold=best_params_per_fold,
            per_fold_metrics=per_fold_metrics,
            weighting_method_choices=weighting_method_choices,
            trial_history_per_fold=trial_history_per_fold,
            inner_top_k_related_islands=inner_top_k_related_islands,
        )
        partial_summary["cv_strategy"] = strategy
        _write_summary(partial_summary, out_path)
        logger.info("OUTER %d incremental summary saved to %s", outer_idx + 1, out_path)

    out_path = _result_output_path(base["paths"], selected_set=selected_set)
    summary = _build_summary(
        outer_results=outer_results,
        selected_set=selected_set,
        unique_islands=unique_islands,
        best_params_per_fold=best_params_per_fold,
        per_fold_metrics=per_fold_metrics,
        weighting_method_choices=weighting_method_choices,
        trial_history_per_fold=trial_history_per_fold,
        inner_top_k_related_islands=inner_top_k_related_islands,
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
        description="Nested CV (MLP) with PCA + logistic importance-weight hyperparameter tuning"
    )
    parser.add_argument("--mode", choices=["worker", "merge"], default="worker")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--selected_splits",
        type=str,
        default=None,
        help="Optional: JSON list or comma-separated 1-based outer split indices to run (e.g., '[10,11]' or '10,11'). Use 'false' to disable.",
    )
    parser.add_argument("--worker_mode", action="store_true", help=argparse.SUPPRESS)

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

    if not args.worker_mode and os.environ.get("NESTED_CV_DISABLE_PARALLEL_OUTER") != "1":
        _run_parallel_outer_splits(cfg, args.config)
        return

    run_nested_cv_importance_weighted_mlp(cfg, config_path=config_path)


if __name__ == "__main__":
    main()
