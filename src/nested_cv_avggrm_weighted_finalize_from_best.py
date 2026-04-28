from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.avggrm_weighting import (
    avg_grm_train_to_target,
    parse_top_k_related_islands,
    ranks_from_desc_scores,
    weights_from_scheme,
)
from src.data import load_data
from src.nested_cv_avggrm_weighted_unified import (
    _apply_include_islands_filter,
    _best_trial_checkpoint_path,
    _bpcrr_one_step_enabled,
    _build_bpcrr_fold_cache,
    _build_summary,
    _evaluate_bpcrr_from_fold_cache,
    _normalize_model_type,
    _parse_bpcrr_prior_settings,
    _parse_selected_splits,
    _prepare_bpcrr_one_step_covariates,
    _result_output_path,
    _write_summary,
)
from src.cv_utils import island_label
from src.training_set_optimization.runner import _evaluate_ridge_subset
from src.utils import _select_top_snps_by_abs_corr, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_best_trial_checkpoint(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def finalize_from_best(
    config: Dict[str, Any],
    config_path: Optional[Path] = None,
    selected_splits_override: Optional[str] = None,
) -> list[str]:
    base = config["base_train"]
    cv_cfg = config.get("cv", {})
    model_type = _normalize_model_type(config.get("model", {}).get("type", "ridge"))
    weighting_space = config.get("search_space", {}).get("weighting", {})
    scheme_choices = [
        str(x).lower()
        for x in weighting_space.get("scheme_choices", ["uniform", "linear", "minmax", "exponential", "top-heavy"])
    ]
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

    grm_mat = None if grm_df is None else grm_df.to_numpy(dtype=np.float64)
    unique_islands = np.unique(locality)

    selected_raw = selected_splits_override
    if selected_raw is None:
        selected_raw = config.get("selected_splits", None)
    if selected_raw is None:
        selected_raw = cv_cfg.get("selected_splits", None)
    selected_set = _parse_selected_splits(selected_raw)
    if selected_set is None and selected_raw is not None:
        raise ValueError(f"Could not parse selected_splits={selected_raw!r}; refusing to finalize every split.")

    if selected_set is None:
        discovered: set[int] = set()
        for outer_idx, _ in enumerate(unique_islands, start=1):
            checkpoint_path = _best_trial_checkpoint_path(base["paths"], model_type, outer_idx)
            if checkpoint_path.exists():
                discovered.add(int(outer_idx))
        if not discovered:
            raise FileNotFoundError(
                f"No best-trial checkpoint files found under "
                f"{_best_trial_checkpoint_path(base['paths'], model_type, 1).parent}"
            )
        selected_set = discovered
        logger.info("No selected_splits provided; finalizing splits discovered from checkpoint files: %s", sorted(selected_set))
    else:
        logger.info("Finalizing only selected splits: %s", sorted(selected_set))

    rr_prior_mode = "default"
    rr_va_apriori = None
    one_step_covars = None
    one_step_enabled = False
    if model_type == "bpcrr":
        rr_prior_mode, rr_va_apriori = _parse_bpcrr_prior_settings(config)
        one_step_enabled = _bpcrr_one_step_enabled(config)
        one_step_covars = _prepare_bpcrr_one_step_covariates(
            config=config,
            config_path=config_path,
            base_paths=base["paths"],
            ids=ids,
            locality_codes=locality,
            code_to_label=code_to_label,
        )
        if one_step_enabled and one_step_covars is None:
            raise RuntimeError("BPCRR one_step is enabled but covariates could not be prepared.")

    written_paths: list[str] = []
    for outer_idx, isl in enumerate(unique_islands, start=1):
        if outer_idx not in selected_set:
            continue

        checkpoint_path = _best_trial_checkpoint_path(base["paths"], model_type, outer_idx)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing best-trial checkpoint for split {outer_idx}: {checkpoint_path}")

        payload = _load_best_trial_checkpoint(checkpoint_path)
        full_best = dict(payload.get("best_params", {}))
        if not full_best:
            raise ValueError(f"Checkpoint file does not contain best_params: {checkpoint_path}")

        idx_outer_train = np.where(locality != isl)[0]
        idx_outer_test = np.where(locality == isl)[0]
        isl_name = island_label(isl, code_to_label)

        requested_inner_top_k = payload.get(
            "inner_validation_top_k_related_islands_requested",
            inner_top_k_related_islands,
        )
        effective_inner_top_k = payload.get("inner_validation_top_k_related_islands_used", None)
        inner_validation_islands = payload.get("inner_validation_islands", None)

        best_weight_spec = dict(full_best.get("weighting", {"name": "uniform"}))
        final_train_weights = None
        if best_weight_spec.get("name", "uniform") != "uniform":
            if grm_mat is None:
                raise RuntimeError("GRM matrix is required for non-uniform AvgGRM weighting")
            avg_grm_outer = avg_grm_train_to_target(grm_mat, idx_outer_train, idx_outer_test)
            ranks_outer = ranks_from_desc_scores(avg_grm_outer)
            final_train_weights = weights_from_scheme(avg_grm_outer, ranks_outer, best_weight_spec)

        if model_type == "bpcrr":
            n_components = int(full_best["n_components"])
            fold_cache = _build_bpcrr_fold_cache(
                train_idx=idx_outer_train,
                target_idx=idx_outer_test,
                X=X,
                one_step_covars=one_step_covars,
                grm_mat=grm_mat if best_weight_spec.get("name", "uniform") != "uniform" else None,
                max_n_components=n_components,
            )
            if fold_cache is None:
                raise RuntimeError(f"Failed to build BPCRR outer-fold cache for split {outer_idx}.")
            if best_weight_spec.get("name", "uniform") != "uniform":
                avg_grm_outer = fold_cache.get("avg_grm")
                ranks_outer = fold_cache.get("ranks")
                if avg_grm_outer is None or ranks_outer is None:
                    raise RuntimeError("Cached AvgGRM rankings are required for non-uniform BPCRR weighting.")
                final_train_weights = weights_from_scheme(avg_grm_outer, ranks_outer, best_weight_spec)

            eval_result = _evaluate_bpcrr_from_fold_cache(
                fold_cache=fold_cache,
                y=y,
                y_eval=y_eval,
                n_components=n_components,
                train_weights=final_train_weights,
                rr_prior_mode=str(full_best.get("prior_mode", rr_prior_mode)),
                rr_va_apriori=full_best.get("va_apriori", rr_va_apriori),
            )
            per_fold_metrics = [
                {
                    "fold": int(outer_idx),
                    "test_corr": float(eval_result["corr_eval"]),
                    "test_size": int(len(idx_outer_test)),
                    "test_island": int(isl),
                    "test_island_name": str(isl_name),
                    "n_components": n_components,
                    "prior_mode": str(full_best.get("prior_mode", rr_prior_mode)),
                    "va_apriori": full_best.get("va_apriori", rr_va_apriori),
                    "one_step_enabled": bool(full_best.get("one_step_enabled", one_step_enabled)),
                    "weighting": best_weight_spec,
                    "inner_validation_top_k_related_islands_requested": requested_inner_top_k,
                    "inner_validation_top_k_related_islands_used": effective_inner_top_k,
                    "inner_validation_islands": inner_validation_islands,
                }
            ]
        else:
            use_snp_selection = bool(full_best.get("use_snp_selection", False))
            num_snps = (
                int(full_best["num_snps"])
                if use_snp_selection and full_best.get("num_snps") is not None
                else None
            )
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
                alpha=float(full_best["alpha"]),
                snp_cols=snp_cols,
                sample_weight=final_train_weights,
            )
            per_fold_metrics = [
                {
                    "fold": int(outer_idx),
                    "test_corr": float(eval_result["corr_eval"]),
                    "test_size": int(len(idx_outer_test)),
                    "test_island": int(isl),
                    "test_island_name": str(isl_name),
                    "alpha": float(full_best["alpha"]),
                    "use_snp_selection": use_snp_selection,
                    "num_snps": num_snps,
                    "weighting": best_weight_spec,
                    "inner_validation_top_k_related_islands_requested": requested_inner_top_k,
                    "inner_validation_top_k_related_islands_used": effective_inner_top_k,
                    "inner_validation_islands": inner_validation_islands,
                }
            ]

        best_params_per_fold = [
            {
                "fold": int(outer_idx),
                "best_params": full_best,
                "mean_inner_r": payload.get("mean_inner_r", None),
                "inner_validation_top_k_related_islands": effective_inner_top_k,
                "inner_validation_islands": inner_validation_islands,
            }
        ]
        summary = _build_summary(
            model_type=model_type,
            strategy=str(cv_cfg.get("strategy", "leave_island_out")).lower(),
            outer_results=[float(per_fold_metrics[0]["test_corr"])],
            selected_set={int(outer_idx)},
            unique_islands=unique_islands,
            best_params_per_fold=best_params_per_fold,
            per_fold_metrics=per_fold_metrics,
            scheme_choices=scheme_choices,
            inner_top_k_related_islands=inner_top_k_related_islands,
        )
        summary["resume_from_best_trial"] = {
            "checkpoint_path": str(checkpoint_path),
            "trial_number": payload.get("trial_number"),
            "saved_at_utc": payload.get("saved_at_utc"),
        }

        out_path = _result_output_path(base["paths"], selected_set={int(outer_idx)}, model_type=model_type)
        _write_summary(summary, out_path)
        written_paths.append(out_path)
        logger.info(
            "Finalized split %d (%s) from best-trial checkpoint. OUTER r = %.4f -> %s",
            outer_idx,
            isl_name,
            float(per_fold_metrics[0]["test_corr"]),
            out_path,
        )

    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize nested-CV outer-fold results from saved best-trial checkpoint files."
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--selected_splits",
        type=str,
        default=None,
        help="Optional: JSON list or comma-separated 1-based outer split indices to finalize.",
    )
    args = parser.parse_args()
    selected_splits = args.selected_splits
    if selected_splits is None:
        selected_splits = os.environ.get("FINALIZE_SPLIT_INDEX")
    if selected_splits is None and os.environ.get("SLURM_ARRAY_JOB_ID"):
        selected_splits = os.environ.get("SLURM_ARRAY_TASK_ID")
    if selected_splits is not None:
        logger.info("CLI selected_splits resolved to: %s", selected_splits)

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    finalize_from_best(cfg, config_path=config_path, selected_splits_override=selected_splits)


if __name__ == "__main__":
    main()
