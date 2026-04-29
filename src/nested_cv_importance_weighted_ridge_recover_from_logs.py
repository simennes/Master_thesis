from __future__ import annotations

import argparse
import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.cv_utils import island_label, make_inner_loio_splits
from src.data import load_data
from src.importance_weighting import compute_pc_logistic_importance_weights
from src.nested_cv_importance_weighted_ridge import (
    _apply_include_islands_filter,
    _build_summary,
    _jsonable,
    _parse_selected_splits,
    _result_output_path,
    _write_summary,
)
from src.training_set_optimization.runner import _evaluate_ridge_subset
from src.utils import _select_top_snps_by_abs_corr, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


BEST_RE = re.compile(r"Best trial:\s*(?P<number>\d+)\.\s*Best value:\s*(?P<value>[-+0-9.eE]+)")
TRIAL_RE = re.compile(
    r"Trial\s+(?P<number>\d+)\s+\|\s+outer=(?P<outer>\d+)\s+\|\s+"
    r"alpha=(?P<alpha>[-+0-9.eE]+)\s+"
    r"use_snp_selection=(?P<use_snp_selection>True|False)\s+"
    r"num_snps=(?P<num_snps>None|\d+)\s+"
    r"weight=(?P<weight>\{.*?\})"
)


def _find_latest_best(log_text: str) -> tuple[int, float]:
    matches = list(BEST_RE.finditer(log_text))
    if not matches:
        raise ValueError("Could not find any 'Best trial' entries in the log.")

    last = matches[-1]
    return int(last.group("number")), float(last.group("value"))


def _parse_trial_lines(log_text: str) -> dict[int, dict[str, Any]]:
    trials: dict[int, dict[str, Any]] = {}
    for match in TRIAL_RE.finditer(log_text):
        number = int(match.group("number"))
        weight_spec = ast.literal_eval(match.group("weight"))
        num_snps_raw = match.group("num_snps")
        trials[number] = {
            "number": number,
            "outer": int(match.group("outer")),
            # The run logs alpha with %.3e, so this is the best recoverable
            # value after the in-memory Optuna study was killed by the queue.
            "alpha": float(match.group("alpha")),
            "use_snp_selection": match.group("use_snp_selection") == "True",
            "num_snps": None if num_snps_raw == "None" else int(num_snps_raw),
            "weighting": weight_spec,
        }
    return trials


def best_trial_from_log(log_path: Path) -> dict[str, Any]:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    best_trial_number, best_value = _find_latest_best(log_text)
    trials = _parse_trial_lines(log_text)
    if best_trial_number not in trials:
        raise ValueError(f"Best trial {best_trial_number} was not found as a Trial line in {log_path}")

    best = dict(trials[best_trial_number])
    best["mean_inner_r"] = best_value
    best["log_path"] = str(log_path)
    best["logged_trials"] = len(trials)
    return best


def _candidate_log_paths(log_pattern: str, split_idx: int) -> list[Path]:
    pattern_path = Path(log_pattern)
    parent = pattern_path.parent if str(pattern_path.parent) != "." else Path(".")
    name_pattern = pattern_path.name
    candidates = sorted(parent.glob(name_pattern))
    suffix = f"_{split_idx}.err"
    return [path for path in candidates if path.name.endswith(suffix)]


def _resolve_log_path(log_pattern: str, split_idx: int) -> Path:
    candidates = _candidate_log_paths(log_pattern, split_idx)
    if not candidates:
        raise FileNotFoundError(f"No timeout log matched split {split_idx} with pattern {log_pattern!r}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _discover_missing_splits(config: dict[str, Any]) -> set[int]:
    base = config["base_train"]
    cv_cfg = config.get("cv", {})

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
    del X, y, ids, grm_df, code_to_label, y_eval

    missing: set[int] = set()
    for split_idx, _ in enumerate(np.unique(locality), start=1):
        out_path = Path(_result_output_path(base["paths"], selected_set={split_idx}))
        if not out_path.exists():
            missing.add(int(split_idx))
    return missing


def recover_from_timeout_logs(
    config: dict[str, Any],
    *,
    selected_splits_override: Optional[str] = None,
    log_pattern: str = "logs/ncv_ridge_iw_*.err",
    overwrite: bool = False,
) -> list[str]:
    base = config["base_train"]
    cv_cfg = config.get("cv", {})
    search_space = config.get("search_space", {})
    weighting_space = search_space.get("importance_weighting", {})
    weighting_method_choices = [str(x).lower() for x in weighting_space.get("method_choices", ["pc_logistic"])]

    seed = int(base.get("seed", config.get("seed", 42)))
    set_seed(seed)

    selected_raw = selected_splits_override
    if selected_raw is None:
        selected_raw = config.get("selected_splits", None)
    if selected_raw is None:
        selected_raw = cv_cfg.get("selected_splits", None)
    selected_set = _parse_selected_splits(selected_raw)
    if selected_set is None:
        selected_set = _discover_missing_splits(config)
        if not selected_set:
            logger.info("No missing split result files were found.")
            return []
        logger.info("No selected splits provided; recovering missing splits: %s", sorted(selected_set))
    else:
        logger.info("Recovering selected splits: %s", sorted(selected_set))

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
    del ids, grm_df

    unique_islands = np.unique(locality)
    written_paths: list[str] = []

    for outer_idx, isl in enumerate(unique_islands, start=1):
        if outer_idx not in selected_set:
            continue

        out_path = _result_output_path(base["paths"], selected_set={outer_idx})
        if Path(out_path).exists() and not overwrite:
            logger.info("Skipping split %d because result already exists: %s", outer_idx, out_path)
            continue

        log_path = _resolve_log_path(log_pattern, outer_idx)
        recovered = best_trial_from_log(log_path)
        if int(recovered["outer"]) != int(outer_idx):
            raise ValueError(f"Log {log_path} says outer={recovered['outer']}, expected {outer_idx}")

        idx_outer_train = np.where(locality != isl)[0]
        idx_outer_test = np.where(locality == isl)[0]
        isl_name = island_label(int(isl), code_to_label)

        use_snp_selection = bool(recovered["use_snp_selection"])
        num_snps = recovered["num_snps"]
        snp_cols = None
        if use_snp_selection:
            if num_snps is None:
                raise ValueError(f"Split {outer_idx} log says use_snp_selection=True but num_snps=None")
            k = min(int(num_snps), X.shape[1])
            snp_cols = _select_top_snps_by_abs_corr(X[idx_outer_train], y[idx_outer_train], k)

        best_weight_spec = dict(recovered["weighting"])
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
            alpha=float(recovered["alpha"]),
            snp_cols=snp_cols,
            sample_weight=final_train_weights,
        )

        inner_plan = make_inner_loio_splits(locality, idx_outer_train)
        inner_ess_thresholds = [float(len(in_tr)) * float(weighting_space.get("min_effective_sample_size_frac", 0.0)) for in_tr, _, _ in inner_plan]
        mean_inner_ess_threshold = float(np.mean(inner_ess_thresholds)) if inner_ess_thresholds else None

        full_best = {
            "model_type": "ridge",
            "alpha": float(recovered["alpha"]),
            "use_snp_selection": use_snp_selection,
            "num_snps": None if num_snps is None else int(num_snps),
            "weighting": _jsonable(best_weight_spec),
            "mean_inner_effective_sample_size": None,
            "mean_inner_effective_sample_size_threshold": mean_inner_ess_threshold,
        }
        best_params_per_fold = [
            {
                "fold": int(outer_idx),
                "best_params": full_best,
                "mean_inner_r": float(recovered["mean_inner_r"]),
            }
        ]
        per_fold_metrics = [
            {
                "fold": int(outer_idx),
                "test_corr": float(eval_result["corr_eval"]),
                "test_size": int(len(idx_outer_test)),
                "test_island": int(isl),
                "test_island_name": str(isl_name),
                "alpha": float(recovered["alpha"]),
                "use_snp_selection": use_snp_selection,
                "num_snps": None if num_snps is None else int(num_snps),
                "weighting": _jsonable(best_weight_spec),
                "effective_sample_size": float(final_weight_result["effective_sample_size"]),
                "pre_shrink_effective_sample_size": None
                if final_weight_result.get("pre_shrink_effective_sample_size") is None
                else float(final_weight_result["pre_shrink_effective_sample_size"]),
                "n_components_used": int(final_weight_result["n_components_used"]),
            }
        ]
        trial_history_per_fold = [
            {
                "fold": int(outer_idx),
                "test_island": int(isl),
                "test_island_name": str(isl_name),
                "n_trials": int(recovered["logged_trials"]),
                "trials": [
                    {
                        "number": int(recovered["number"]),
                        "state": "COMPLETE",
                        "value": float(recovered["mean_inner_r"]),
                        "params": {
                            "alpha": float(recovered["alpha"]),
                            "use_snp_selection": use_snp_selection,
                            "iw_method": str(best_weight_spec.get("name", "uniform")),
                            "iw_clip_max": best_weight_spec.get("clip_max"),
                            "iw_prob_clip": best_weight_spec.get("prob_clip"),
                            "iw_n_components": best_weight_spec.get("n_components"),
                            "iw_logistic_c": best_weight_spec.get("logistic_c"),
                            "iw_pca_fit": best_weight_spec.get("pca_fit"),
                            "iw_solver": best_weight_spec.get("solver"),
                            "iw_rho": best_weight_spec.get("rho"),
                        },
                        "weighting": _jsonable(best_weight_spec),
                        "mean_inner_effective_sample_size": None,
                        "mean_inner_effective_sample_size_threshold": mean_inner_ess_threshold,
                        "effective_sample_size_rejected": False,
                    }
                ],
            }
        ]

        summary = _build_summary(
            outer_results=[float(eval_result["corr_eval"])],
            selected_set={int(outer_idx)},
            unique_islands=unique_islands,
            best_params_per_fold=best_params_per_fold,
            per_fold_metrics=per_fold_metrics,
            weighting_method_choices=weighting_method_choices,
            trial_history_per_fold=trial_history_per_fold,
        )
        summary["cv_strategy"] = str(cv_cfg.get("strategy", "leave_island_out"))
        summary["recovered_from_timeout_log"] = {
            "log_path": str(log_path),
            "best_trial_number": int(recovered["number"]),
            "logged_trial_count": int(recovered["logged_trials"]),
            "note": "alpha was recovered from the SLURM log, where it is printed with %.3e precision.",
        }

        _write_summary(summary, out_path)
        written_paths.append(out_path)
        logger.info(
            "Recovered split %d (%s) from trial %d. OUTER r = %.4f -> %s",
            outer_idx,
            isl_name,
            int(recovered["number"]),
            float(eval_result["corr_eval"]),
            out_path,
        )

    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover timed-out importance-weighted ridge nested-CV shards from their SLURM logs."
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--selected_splits",
        type=str,
        default=None,
        help="Optional JSON list or comma-separated 1-based splits. Defaults to missing shard result files.",
    )
    parser.add_argument(
        "--log_pattern",
        type=str,
        default="logs/ncv_ridge_iw_*.err",
        help="Glob for SLURM stderr logs. The newest file ending in _{split}.err is used for each split.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    recover_from_timeout_logs(
        config,
        selected_splits_override=args.selected_splits,
        log_pattern=args.log_pattern,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
