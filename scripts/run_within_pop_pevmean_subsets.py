#!/usr/bin/env python
"""Within-population PEVmean-GA subset curves evaluated with fixed PC-ridge."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.within_pop_subset_utils import (
    append_csv_rows,
    as_list,
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
from src.training_set_optimization.ga_subset import GAConfig, run_ga
from src.training_set_optimization.pevmean import pev_mean, pev_mean_batch
from src.utils import set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _assign_jobs_weighted(jobs: list[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
    if num_shards <= 1:
        return [jobs]
    bins: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    loads = np.zeros(num_shards, dtype=np.float64)
    for job in sorted(jobs, key=lambda item: float(item.get("weight", 1.0)), reverse=True):
        target = int(np.argmin(loads))
        bins[target].append(job)
        loads[target] += float(job.get("weight", 1.0))
    return bins


def _ga_config(config: dict[str, Any]) -> GAConfig:
    cfg = config.get("pevmean_ga", {})
    return GAConfig(
        pop_size=int(cfg.get("pop_size", 100)),
        n_generations=int(cfg.get("n_generations", 300)),
        n_elite=int(cfg.get("n_elite", 2)),
        tournament_k=int(cfg.get("tournament_k", 5)),
        crossover_prob=float(cfg.get("crossover_prob", 0.9)),
        mutation_prob=float(cfg.get("mutation_prob", 0.3)),
        n_swaps_per_mut=int(cfg.get("n_swaps_per_mut", 2)),
        seed=int(config.get("seed", 14)),
        verbose=bool(cfg.get("verbose", False)),
        stagnation_limit=int(cfg.get("stagnation_limit", 20)),
    )


def _make_repeat_seed(seed: int, fold: int, repeat: int, n_train: int) -> int:
    return int(seed + fold * 1_000_000 + repeat * 10_000 + n_train)


def _evaluate_and_row(
    *,
    trait_name: str,
    fold_idx: int,
    selected_global: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    y_eval: np.ndarray,
    test_idx: np.ndarray,
    n_pcs: int,
    selection_n_pcs: int,
    alpha: float,
    repeat: int,
    repeat_seed: int,
    n_train_available: int,
    pevmean_obj: float,
    ga_stats: dict[str, Any],
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
        "method": "pevmean_ga_pc_ridge",
        "selection_method": "pevmean_ga",
        "repeat": int(repeat),
        "repeat_seed": int(repeat_seed),
        "n_individuals": int(len(selected_global)),
        "n_train_available": int(n_train_available),
        "n_test": int(len(test_idx)),
        "n_pcs": int(n_pcs),
        "n_pcs_selection": int(selection_n_pcs),
        "n_pcs_fit": int(metrics["n_pcs_fit"]),
        "alpha": float(alpha),
        "pearson_r": float(metrics["pearson_r"]),
        "mse_adj": float(metrics["mse_adj"]),
        "pevmean_obj": float(pevmean_obj),
        "ga_generations": int(ga_stats.get("generations_run", 0)),
        "ga_cache_size": int(ga_stats.get("cache_size", 0)),
        "ga_elapsed_sec": float(ga_stats.get("elapsed_sec", 0.0)),
        "fit_time_seconds": float(metrics["fit_time_seconds"]),
    }


def _build_jobs(
    *,
    outer_plan: list[tuple[np.ndarray, np.ndarray]],
    n_train_sizes: list[Any] | None,
    n_repeats: int,
    include_full_train_baseline: bool,
    include_full_train: bool,
    selected_folds: set[int] | None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for fold_idx, (train_idx, _test_idx) in enumerate(outer_plan, start=1):
        if selected_folds is not None and fold_idx not in selected_folds:
            continue
        n_available = int(len(train_idx))
        if include_full_train_baseline:
            jobs.append(
                {
                    "kind": "full",
                    "fold": int(fold_idx),
                    "repeat": -1,
                    "n_train": int(n_available),
                    "weight": 1.0,
                }
            )
        step_counts = compute_step_counts(
            n_train_sizes,
            n_available,
            include_full_train=include_full_train,
        )
        for repeat in range(n_repeats):
            for n_train in step_counts:
                n_train = int(n_train)
                if include_full_train_baseline and n_train >= n_available:
                    continue
                jobs.append(
                    {
                        "kind": "ga",
                        "fold": int(fold_idx),
                        "repeat": int(repeat),
                        "n_train": int(n_train),
                        "weight": float(max(1, n_train) ** 2),
                    }
                )
    return jobs


def _run_trait(
    *,
    config: dict[str, Any],
    trait_spec: dict[str, Any],
    output_root: Path,
    file_stem: str,
    selected_folds: set[int] | None,
    shard_index: int,
    num_shards: int,
) -> None:
    seed = int(config.get("seed", 14))
    subset_cfg = config.get("subsets", {})
    pev_cfg = config.get("pevmean_ga", {})
    cv_cfg = config.get("cv", {})
    trait_name = str(trait_spec["name"])

    logger.info(
        "==== Trait '%s' PEVmean within-pop subsets shard %d/%d ====",
        trait_name,
        shard_index,
        num_shards,
    )
    X, y, ids, _grm_df, locality, code_to_label, y_eval = load_trait_data(
        trait_spec,
        config,
        require_grm=False,
    )

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
        "Trait=%s PCA fit for PEVmean/eval (n=%d, p=%d, r=%d) in %.2fs (cumvar=%.3f)",
        trait_name,
        int(X.shape[0]),
        int(X.shape[1]),
        int(Z.shape[1]),
        pca_seconds,
        float(np.sum(var_ratio)),
    )

    n_repeats = int(subset_cfg.get("n_repeats", 1))
    include_full_baseline = bool(subset_cfg.get("include_full_train_baseline", True))
    include_full_in_curves = bool(subset_cfg.get("include_full_train", False))
    jobs = _build_jobs(
        outer_plan=outer_plan,
        n_train_sizes=subset_cfg.get("n_train_sizes"),
        n_repeats=n_repeats,
        include_full_train_baseline=include_full_baseline,
        include_full_train=include_full_in_curves,
        selected_folds=selected_folds,
    )
    shard_jobs = _assign_jobs_weighted(jobs, num_shards)[shard_index] if num_shards > 1 else jobs
    logger.info("Shard %d/%d assigned %d/%d PEVmean jobs", shard_index, num_shards, len(shard_jobs), len(jobs))

    trait_dir = output_root / trait_name
    if num_shards > 1:
        trait_dir = output_root / "shards" / f"shard_{shard_index:03d}" / trait_name
    trait_dir.mkdir(parents=True, exist_ok=True)
    results_path = trait_dir / f"{file_stem}_results.csv"
    selected_path = trait_dir / f"{file_stem}_selected_individuals.csv"
    summary_path = trait_dir / f"{file_stem}_summary.csv"
    if bool(config.get("output", {}).get("overwrite", True)):
        for path in (results_path, selected_path, summary_path):
            if path.exists():
                path.unlink()

    jobs_by_fold: dict[int, list[dict[str, Any]]] = {}
    for job in shard_jobs:
        jobs_by_fold.setdefault(int(job["fold"]), []).append(job)

    base_ga_cfg = _ga_config(config)
    n_jobs = int(pev_cfg.get("n_jobs", 1))
    parallel_min_n_train = int(pev_cfg.get("parallel_min_n_train", 1))
    max_selection_pcs = pev_cfg.get("max_selection_pcs")
    max_selection_pcs = None if max_selection_pcs is None else int(max_selection_pcs)
    write_selected = bool(subset_cfg.get("write_selected_individuals", False))

    for fold_idx, fold_jobs in sorted(jobs_by_fold.items()):
        train_idx, test_idx = outer_plan[fold_idx - 1]
        params = fold_params[fold_idx]
        n_pcs = int(params["n_pcs"])
        alpha = float(params["alpha"])
        selection_n_pcs = int(min(n_pcs, Z.shape[1]))
        if max_selection_pcs is not None:
            selection_n_pcs = int(min(selection_n_pcs, max_selection_pcs))
        P = Z[:, :selection_n_pcs]

        logger.info(
            "Trait=%s fold=%d jobs=%d n_train_available=%d n_test=%d n_pcs=%d selection_pcs=%d alpha=%.3e",
            trait_name,
            fold_idx,
            len(fold_jobs),
            int(len(train_idx)),
            int(len(test_idx)),
            n_pcs,
            selection_n_pcs,
            alpha,
        )

        result_buffer: list[dict[str, Any]] = []
        selected_buffer: list[dict[str, Any]] = []

        for job_i, job in enumerate(sorted(fold_jobs, key=lambda item: (item["kind"], item["repeat"], item["n_train"])), start=1):
            n_train = int(job["n_train"])
            repeat = int(job["repeat"])
            if job["kind"] == "full":
                selected_global = train_idx
                pev_obj = float(pev_mean(P, selected_global, test_idx, lam=alpha))
                result_buffer.append(
                    _evaluate_and_row(
                        trait_name=trait_name,
                        fold_idx=fold_idx,
                        selected_global=selected_global,
                        Z=Z,
                        y=y,
                        y_eval=y_eval,
                        test_idx=test_idx,
                        n_pcs=n_pcs,
                        selection_n_pcs=selection_n_pcs,
                        alpha=alpha,
                        repeat=-1,
                        repeat_seed=-1,
                        n_train_available=int(len(train_idx)),
                        pevmean_obj=pev_obj,
                        ga_stats={"generations_run": 0, "cache_size": 0, "elapsed_sec": 0.0},
                    )
                )
                result_buffer[-1]["method"] = "full_train_pc_ridge"
                result_buffer[-1]["selection_method"] = "full_train"
                continue

            repeat_seed = _make_repeat_seed(seed, fold_idx, repeat, n_train)
            step_ga_cfg = copy.deepcopy(base_ga_cfg)
            step_ga_cfg.seed = repeat_seed

            def fitness_fn(local_subset: np.ndarray) -> float:
                return pev_mean(P, train_idx[local_subset], test_idx, lam=alpha)

            if n_jobs > 1 and n_train >= parallel_min_n_train:
                def batch_fitness_fn(local_subsets):
                    return pev_mean_batch(
                        P,
                        [train_idx[np.asarray(s, dtype=np.int64)] for s in local_subsets],
                        test_idx,
                        alpha,
                        n_jobs=n_jobs,
                    )
            else:
                batch_fitness_fn = None

            logger.info(
                "  fold=%d job=%d/%d repeat=%d n_train=%d",
                fold_idx,
                job_i,
                len(fold_jobs),
                repeat,
                n_train,
            )
            best_local, best_pev, ga_stats = run_ga(
                n_candidates=int(len(train_idx)),
                n_train=n_train,
                fitness_fn=fitness_fn,
                cfg=step_ga_cfg,
                candidate_indices=np.arange(len(train_idx), dtype=np.int64),
                batch_fitness_fn=batch_fitness_fn,
            )
            selected_global = train_idx[np.asarray(best_local, dtype=np.int64)]
            result_buffer.append(
                _evaluate_and_row(
                    trait_name=trait_name,
                    fold_idx=fold_idx,
                    selected_global=selected_global,
                    Z=Z,
                    y=y,
                    y_eval=y_eval,
                    test_idx=test_idx,
                    n_pcs=n_pcs,
                    selection_n_pcs=selection_n_pcs,
                    alpha=alpha,
                    repeat=repeat,
                    repeat_seed=repeat_seed,
                    n_train_available=int(len(train_idx)),
                    pevmean_obj=float(best_pev),
                    ga_stats=ga_stats,
                )
            )
            if write_selected:
                selected_buffer.extend(
                    selected_rows(
                        trait_name=trait_name,
                        fold=fold_idx,
                        method="pevmean_ga_pc_ridge",
                        n_train=n_train,
                        ids=ids,
                        locality=locality,
                        code_to_label=code_to_label,
                        selected_idx=selected_global,
                        extra={
                            "selection_method": "pevmean_ga",
                            "repeat": repeat,
                            "repeat_seed": repeat_seed,
                            "pevmean_obj": float(best_pev),
                            "n_pcs_selection": selection_n_pcs,
                        },
                    )
                )

            append_csv_rows(result_buffer, results_path)
            append_csv_rows(selected_buffer, selected_path)
            result_buffer = []
            selected_buffer = []

        append_csv_rows(result_buffer, results_path)
        append_csv_rows(selected_buffer, selected_path)

    write_summary(results_path, summary_path)
    logger.info("Trait '%s' shard complete. Results: %s", trait_name, results_path)


def merge_outputs(config: dict[str, Any], only_traits: set[str] | None = None) -> None:
    output_cfg = config.get("output", {})
    output_root = Path(output_cfg.get("root_dir", "outputs/final_results/within_pop_subset_pevmean"))
    file_stem = str(output_cfg.get("file_stem", "within_pop_subset_pevmean"))
    trait_specs = build_trait_specs(config, only_traits, require_grm=False)
    shards_root = output_root / "shards"

    for spec in trait_specs:
        trait_name = str(spec["name"])
        trait_dir = output_root / trait_name
        trait_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ["results", "selected_individuals"]:
            parts = []
            if shards_root.exists():
                pattern = f"shard_*/{trait_name}/{file_stem}_{suffix}.csv"
                for path in sorted(shards_root.glob(pattern)):
                    parts.append(pd.read_csv(path))
            else:
                path = trait_dir / f"{file_stem}_{suffix}.csv"
                if path.exists():
                    parts.append(pd.read_csv(path))
            if parts:
                pd.concat(parts, ignore_index=True).to_csv(trait_dir / f"{file_stem}_{suffix}.csv", index=False)
        write_summary(
            trait_dir / f"{file_stem}_results.csv",
            trait_dir / f"{file_stem}_summary.csv",
        )

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
    logger.info("Merged available PEVmean outputs under %s", output_root)


def run(
    config: dict[str, Any],
    *,
    only_traits: set[str] | None,
    selected_folds: set[int] | None,
    shard_index: int,
    num_shards: int,
) -> None:
    seed = int(config.get("seed", 14))
    set_seed(seed)

    output_cfg = config.get("output", {})
    output_root = Path(output_cfg.get("root_dir", "outputs/final_results/within_pop_subset_pevmean"))
    file_stem = str(output_cfg.get("file_stem", "within_pop_subset_pevmean"))
    if selected_folds is not None:
        suffix = "_".join(str(i) for i in sorted(selected_folds))
        file_stem = f"{file_stem}_folds_{suffix}"
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / f"{file_stem}_config_used.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}]")

    for trait_spec in build_trait_specs(config, only_traits, require_grm=False):
        _run_trait(
            config=config,
            trait_spec=trait_spec,
            output_root=output_root,
            file_stem=file_stem,
            selected_folds=selected_folds,
            shard_index=shard_index,
            num_shards=num_shards,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trait", action="append", default=None)
    parser.add_argument("--selected_folds", default=None)
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    only_traits = set(as_list(args.trait)) if args.trait else None
    selected = parse_int_list(args.selected_folds)
    selected_folds = set(selected) if selected else None

    if args.mode == "merge":
        merge_outputs(config, only_traits=only_traits)
        return

    shard_index = args.shard_index
    if shard_index is None:
        shard_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    num_shards = args.num_shards
    if num_shards is None:
        num_shards = int(os.environ.get("SWEEP_NUM_SHARDS", "1"))

    run(
        config,
        only_traits=only_traits,
        selected_folds=selected_folds,
        shard_index=int(shard_index),
        num_shards=int(num_shards),
    )


if __name__ == "__main__":
    main()
