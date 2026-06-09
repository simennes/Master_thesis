#!/usr/bin/env python
"""Within-population subset curves for avgGRM, avgGRM-diversity, and random."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.within_pop_subset_utils import (
    append_csv_rows,
    as_list,
    avg_train_train_block,
    build_trait_specs,
    compute_step_counts,
    evaluate_pc_ridge_subset,
    fit_dataset_pca,
    load_fold_hyperparams,
    load_trait_data,
    make_cv_splits,
    parse_int_list,
    selected_rows,
    write_summary,
)
from src.avggrm_weighting import greedy_avggrm_diversity_order
from src.utils import set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _format_lambda(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _method_for_lambda(lambda_div: float) -> tuple[str, str]:
    if abs(float(lambda_div)) < 1e-12:
        return "avggrm_pc_ridge", "avggrm"
    return "avggrm_diversity_pc_ridge", "avggrm_diversity"


def _evaluate_and_row(
    *,
    trait_name: str,
    fold_idx: int,
    method: str,
    selection_method: str,
    lambda_div: float | None,
    repeat: int | None,
    repeat_seed: int | None,
    n_train_available: int,
    n_test: int,
    selected_global: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    y_eval: np.ndarray,
    test_idx: np.ndarray,
    n_pcs: int,
    alpha: float,
    avg_grm_target_obj: float,
    avg_grm_train_train_obj: float,
    objective_obj: float,
    selection_time_seconds: float,
) -> dict[str, Any]:
    metrics = evaluate_pc_ridge_subset(
        Z,
        y,
        y_eval,
        selected_global,
        test_idx,
        n_pcs=int(n_pcs),
        alpha=float(alpha),
    )
    return {
        "trait": trait_name,
        "fold": int(fold_idx),
        "method": method,
        "selection_method": selection_method,
        "lambda_label": None if lambda_div is None else f"lambda_{_format_lambda(float(lambda_div))}",
        "lambda_div": np.nan if lambda_div is None else float(lambda_div),
        "repeat": -1 if repeat is None else int(repeat),
        "repeat_seed": -1 if repeat_seed is None else int(repeat_seed),
        "n_individuals": int(len(selected_global)),
        "n_train_available": int(n_train_available),
        "n_test": int(n_test),
        "n_pcs": int(n_pcs),
        "n_pcs_fit": int(metrics["n_pcs_fit"]),
        "alpha": float(alpha),
        "pearson_r": float(metrics["pearson_r"]),
        "mse_adj": float(metrics["mse_adj"]),
        "avg_grm_target_obj": float(avg_grm_target_obj),
        "avg_grm_train_train_obj": float(avg_grm_train_train_obj),
        "objective_obj": float(objective_obj),
        "selection_time_seconds": float(selection_time_seconds),
        "fit_time_seconds": float(metrics["fit_time_seconds"]),
    }


def _run_trait(
    *,
    config: dict[str, Any],
    trait_spec: dict[str, Any],
    output_root: Path,
    file_stem: str,
    selected_folds: set[int] | None,
) -> None:
    seed = int(config.get("seed", 14))
    subset_cfg = config.get("subsets", {})
    cv_cfg = config.get("cv", {})
    trait_name = str(trait_spec["name"])

    logger.info("==== Trait '%s' avgGRM/random within-pop subsets ====", trait_name)
    X, y, ids, grm_df, locality, code_to_label, y_eval = load_trait_data(
        trait_spec,
        config,
        require_grm=True,
    )
    if grm_df is None:
        raise RuntimeError("GRM missing after load_trait_data(require_grm=True)")
    grm = grm_df.to_numpy(dtype=np.float64)

    fold_params = load_fold_hyperparams(config, trait_name)
    outer_plan = make_cv_splits(
        np.arange(len(y), dtype=int),
        locality,
        n_splits=int(cv_cfg.get("n_splits", 10)),
        shuffle=bool(cv_cfg.get("shuffle", True)),
        random_state=int(cv_cfg.get("random_state", seed)),
        stratify_by_locality=bool(cv_cfg.get("stratify_by_locality", True)),
    )
    folds_to_fit = [
        fold_idx for fold_idx in range(1, len(outer_plan) + 1)
        if selected_folds is None or fold_idx in selected_folds
    ]
    max_r = int(max(int(fold_params[f]["n_pcs"]) for f in folds_to_fit))
    Z, pca_seconds, var_ratio = fit_dataset_pca(
        X=X,
        max_r=max_r,
        seed=seed,
        standardize=bool(trait_spec.get("standardize_features", False)),
    )
    logger.info(
        "Trait=%s PCA fit for subset evaluation (n=%d, p=%d, r=%d) in %.2fs (cumvar=%.3f)",
        trait_name,
        int(X.shape[0]),
        int(X.shape[1]),
        int(Z.shape[1]),
        pca_seconds,
        float(np.sum(var_ratio)),
    )

    trait_dir = output_root / trait_name
    trait_dir.mkdir(parents=True, exist_ok=True)
    results_path = trait_dir / f"{file_stem}_results.csv"
    selected_path = trait_dir / f"{file_stem}_selected_individuals.csv"
    summary_path = trait_dir / f"{file_stem}_summary.csv"
    if bool(config.get("output", {}).get("overwrite", True)):
        for path in (results_path, selected_path, summary_path):
            if path.exists():
                path.unlink()

    n_train_sizes = subset_cfg.get("n_train_sizes")
    lambda_divs = [float(v) for v in subset_cfg.get("lambda_divs", [0.0, 1.0])]
    random_repeats = int(subset_cfg.get("random_repeats", 10))
    include_full_baseline = bool(subset_cfg.get("include_full_train_baseline", True))
    include_full_in_curves = bool(subset_cfg.get("include_full_train", False))
    write_selected = bool(subset_cfg.get("write_selected_individuals", False))
    include_diagonal = bool(subset_cfg.get("diversity_include_diagonal", True))

    result_buffer: list[dict[str, Any]] = []
    selected_buffer: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_plan, start=1):
        if selected_folds is not None and fold_idx not in selected_folds:
            continue
        if fold_idx not in fold_params:
            raise KeyError(f"No fixed PC-ridge parameters found for fold {fold_idx}")

        params = fold_params[fold_idx]
        n_pcs = int(params["n_pcs"])
        alpha = float(params["alpha"])
        n_candidates = int(len(train_idx))
        step_counts = compute_step_counts(
            n_train_sizes,
            n_candidates,
            include_full_train=include_full_in_curves,
        )
        logger.info(
            "Trait=%s fold=%d n_train_available=%d n_test=%d n_pcs=%d alpha=%.3e sizes=%s",
            trait_name,
            fold_idx,
            n_candidates,
            int(len(test_idx)),
            n_pcs,
            alpha,
            step_counts.tolist(),
        )

        avg_grm_to_target = np.asarray(grm[np.ix_(train_idx, test_idx)].mean(axis=1), dtype=float)
        needs_train_train = any(abs(v) >= 1e-12 for v in lambda_divs) or include_full_baseline
        train_train_grm = (
            np.asarray(grm[np.ix_(train_idx, train_idx)], dtype=float)
            if needs_train_train else None
        )

        if include_full_baseline:
            full_local = np.arange(n_candidates, dtype=np.int64)
            full_global = train_idx
            avg_tt = (
                avg_train_train_block(train_train_grm, full_local, include_diagonal)
                if train_train_grm is not None else float("nan")
            )
            result_buffer.append(
                _evaluate_and_row(
                    trait_name=trait_name,
                    fold_idx=fold_idx,
                    method="full_train_pc_ridge",
                    selection_method="full_train",
                    lambda_div=None,
                    repeat=None,
                    repeat_seed=None,
                    n_train_available=n_candidates,
                    n_test=int(len(test_idx)),
                    selected_global=full_global,
                    Z=Z,
                    y=y,
                    y_eval=y_eval,
                    test_idx=test_idx,
                    n_pcs=n_pcs,
                    alpha=alpha,
                    avg_grm_target_obj=float(np.mean(avg_grm_to_target)),
                    avg_grm_train_train_obj=avg_tt,
                    objective_obj=float("nan"),
                    selection_time_seconds=0.0,
                )
            )

        for repeat in range(random_repeats):
            repeat_seed = seed + fold_idx * 100_000 + repeat
            rng = np.random.default_rng(repeat_seed)
            order = rng.permutation(n_candidates)
            for k in step_counts:
                n_train = int(k)
                if n_train >= n_candidates and include_full_baseline:
                    continue
                chosen_local = order[:n_train]
                chosen_global = train_idx[chosen_local]
                result_buffer.append(
                    _evaluate_and_row(
                        trait_name=trait_name,
                        fold_idx=fold_idx,
                        method="random_pc_ridge",
                        selection_method="random",
                        lambda_div=None,
                        repeat=repeat,
                        repeat_seed=repeat_seed,
                        n_train_available=n_candidates,
                        n_test=int(len(test_idx)),
                        selected_global=chosen_global,
                        Z=Z,
                        y=y,
                        y_eval=y_eval,
                        test_idx=test_idx,
                        n_pcs=n_pcs,
                        alpha=alpha,
                        avg_grm_target_obj=float(np.mean(avg_grm_to_target[chosen_local])),
                        avg_grm_train_train_obj=float("nan"),
                        objective_obj=float("nan"),
                        selection_time_seconds=0.0,
                    )
                )
                if write_selected:
                    selected_buffer.extend(
                        selected_rows(
                            trait_name=trait_name,
                            fold=fold_idx,
                            method="random_pc_ridge",
                            n_train=n_train,
                            ids=ids,
                            locality=locality,
                            code_to_label=code_to_label,
                            selected_idx=chosen_global,
                            extra={"selection_method": "random", "repeat": repeat, "repeat_seed": repeat_seed},
                        )
                    )

        for lambda_div in lambda_divs:
            method, selection_method = _method_for_lambda(lambda_div)
            t0 = time.perf_counter()
            if abs(lambda_div) < 1e-12:
                order = np.argsort(-avg_grm_to_target, kind="mergesort").astype(np.int64)
                selection_seconds = float(time.perf_counter() - t0)
                cum_avg_target = np.cumsum(avg_grm_to_target[order]) / np.arange(1, len(order) + 1)
                trace = None
            else:
                if train_train_grm is None:
                    train_train_grm = np.asarray(grm[np.ix_(train_idx, train_idx)], dtype=float)
                max_k = int(min(np.max(step_counts), n_candidates))
                trace = greedy_avggrm_diversity_order(
                    avg_grm_to_target=avg_grm_to_target,
                    train_train_grm=train_train_grm,
                    lambda_div=float(lambda_div),
                    max_size=max_k,
                    include_diagonal=include_diagonal,
                )
                order = np.asarray(trace["order"], dtype=np.int64)
                selection_seconds = float(time.perf_counter() - t0)

            for k in step_counts:
                n_train = int(min(int(k), len(order)))
                if n_train >= n_candidates and include_full_baseline:
                    continue
                chosen_local = order[:n_train]
                chosen_global = train_idx[chosen_local]
                if trace is None:
                    avg_target_obj = float(cum_avg_target[n_train - 1])
                    avg_tt_obj = float("nan")
                    objective_obj = avg_target_obj
                else:
                    pos = n_train - 1
                    avg_target_obj = float(trace["avg_grm_train_target_after"][pos])
                    avg_tt_obj = float(trace["avg_grm_train_train_after"][pos])
                    objective_obj = float(trace["objective_after"][pos])

                result_buffer.append(
                    _evaluate_and_row(
                        trait_name=trait_name,
                        fold_idx=fold_idx,
                        method=method,
                        selection_method=selection_method,
                        lambda_div=float(lambda_div),
                        repeat=None,
                        repeat_seed=None,
                        n_train_available=n_candidates,
                        n_test=int(len(test_idx)),
                        selected_global=chosen_global,
                        Z=Z,
                        y=y,
                        y_eval=y_eval,
                        test_idx=test_idx,
                        n_pcs=n_pcs,
                        alpha=alpha,
                        avg_grm_target_obj=avg_target_obj,
                        avg_grm_train_train_obj=avg_tt_obj,
                        objective_obj=objective_obj,
                        selection_time_seconds=selection_seconds,
                    )
                )
                if write_selected:
                    selected_buffer.extend(
                        selected_rows(
                            trait_name=trait_name,
                            fold=fold_idx,
                            method=method,
                            n_train=n_train,
                            ids=ids,
                            locality=locality,
                            code_to_label=code_to_label,
                            selected_idx=chosen_global,
                            extra={
                                "selection_method": selection_method,
                                "lambda_div": float(lambda_div),
                                "repeat": -1,
                                "repeat_seed": -1,
                            },
                        )
                    )

        append_csv_rows(result_buffer, results_path)
        append_csv_rows(selected_buffer, selected_path)
        result_buffer = []
        selected_buffer = []

    write_summary(results_path, summary_path)
    logger.info("Trait '%s' complete. Results: %s", trait_name, results_path)


def merge_outputs(config: dict[str, Any], only_traits: set[str] | None = None) -> None:
    output_cfg = config.get("output", {})
    output_root = Path(output_cfg.get("root_dir", "outputs/final_results/within_pop_subset_avggrm_random"))
    file_stem = str(output_cfg.get("file_stem", "within_pop_subset_avggrm_random"))
    trait_specs = build_trait_specs(config, only_traits, require_grm=True)

    for suffix in ["results", "selected_individuals", "summary"]:
        parts = []
        for spec in trait_specs:
            path = output_root / str(spec["name"]) / f"{file_stem}_{suffix}.csv"
            if path.exists():
                parts.append(pd.read_csv(path))
            elif suffix != "selected_individuals":
                logger.warning("Missing %s for trait %s: %s", suffix, spec["name"], path)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(output_root / f"{file_stem}_{suffix}.csv", index=False)
    logger.info("Merged available avgGRM/random outputs under %s", output_root)


def run(
    config: dict[str, Any],
    *,
    only_traits: set[str] | None,
    selected_folds: set[int] | None,
) -> None:
    seed = int(config.get("seed", 14))
    set_seed(seed)

    output_cfg = config.get("output", {})
    output_root = Path(output_cfg.get("root_dir", "outputs/final_results/within_pop_subset_avggrm_random"))
    file_stem = str(output_cfg.get("file_stem", "within_pop_subset_avggrm_random"))
    if selected_folds is not None:
        suffix = "_".join(str(i) for i in sorted(selected_folds))
        file_stem = f"{file_stem}_folds_{suffix}"
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / f"{file_stem}_config_used.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    for trait_spec in build_trait_specs(config, only_traits, require_grm=True):
        _run_trait(
            config=config,
            trait_spec=trait_spec,
            output_root=output_root,
            file_stem=file_stem,
            selected_folds=selected_folds,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trait", action="append", default=None)
    parser.add_argument("--selected_folds", default=None)
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    only_traits = set(as_list(args.trait)) if args.trait else None
    selected = parse_int_list(args.selected_folds)
    selected_folds = set(selected) if selected else None

    if args.mode == "merge":
        merge_outputs(config, only_traits=only_traits)
    else:
        run(config, only_traits=only_traits, selected_folds=selected_folds)


if __name__ == "__main__":
    main()
