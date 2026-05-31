"""Variance benchmark: run N GA seeds at fixed (trait, target, n_train)
and report the spread of best PEVmean and downstream Pearson r.

Uses marker-form PEVmean (see src/training_set_optimization/pevmean.py).

Usage:
    python -m scripts._bench_pevmean_ga_seeds \\
        --config config/e3/final_e3_pevmean_ga_body_mass_config.json \\
        --n_train 500 --target_island 0 --n_seeds 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from scripts.run_pevmean_ga import (
    _build_trait_specs,
    _compute_dataset_pcs,
    resolve_island_code,
)
from src.data import load_data
from src.training_set_optimization.ga_subset import GAConfig, run_ga
from src.training_set_optimization.pevmean import (
    paper_style_lambda,
    pev_mean,
    pev_mean_batch,
    sum_pc_variances,
)
from src.training_set_optimization.runner import _evaluate_ridge_subset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--target_island", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=5)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    trait_specs = _build_trait_specs(cfg)
    trait = trait_specs[0]
    print(f"Trait: {trait['name']}  downstream_ridge: {trait['downstream_ridge']}")

    X, y, ids, _grm, locality, code_to_label, y_eval = load_data(
        paths=trait["paths"],
        target_column=trait["target_column"],
        standardize_features=trait["standardize_features"],
        return_locality=True,
        min_count=trait["min_count"],
        return_eval=True,
        eval_target_column=trait["eval_target_column"],
    )

    pevmean_cfg = cfg.get("pevmean", {})
    pevmean_n_pcs = int(pevmean_cfg.get("n_pcs", 100))
    downstream_n_pcs = int(trait["downstream_ridge"]["n_pcs"])
    downstream_alpha = float(trait["downstream_ridge"]["alpha"])
    max_r = max(pevmean_n_pcs, downstream_n_pcs)

    Z, var_ratio, dt = _compute_dataset_pcs(
        X=X, n_pcs=max_r,
        seed=int(pevmean_cfg.get("pca_seed", cfg.get("seed", 14))),
        standardize=bool(pevmean_cfg.get("standardize_for_pca", False)),
    )
    cumvar_pev = float(np.sum(var_ratio[:pevmean_n_pcs]))
    print(f"PCA fit (r={max_r}): {dt:.1f}s  cumvar at r={pevmean_n_pcs}: {cumvar_pev:.3f}")

    # PEVmean lambda (paper mode)
    va = float(pevmean_cfg.get("va_apriori"))
    sigma_e2 = pevmean_cfg.get("sigma_e2_apriori")
    if sigma_e2 is None:
        sigma_e2 = float(np.var(np.asarray(y, dtype=np.float64), ddof=1))
    else:
        sigma_e2 = float(sigma_e2)
    Z_pev = np.ascontiguousarray(Z[:, :pevmean_n_pcs], dtype=np.float64)
    sum_var_pc = sum_pc_variances(Z_pev)
    pevmean_lambda = paper_style_lambda(sigma_e2, va, sum_var_pc)
    print(f"PEVmean lambda (paper mode): sigma_e^2={sigma_e2:.4f} * sum_var_PC={sum_var_pc:.2f} "
          f"/ va={va:.3f} = {pevmean_lambda:.3e}")

    target_code = resolve_island_code(args.target_island, code_to_label, set(int(c) for c in np.unique(locality)))
    target_mask = (locality == target_code)
    source_mask = ~target_mask
    source_idx_global = np.flatnonzero(source_mask).astype(np.int64)
    target_idx_global = np.flatnonzero(target_mask).astype(np.int64)
    N_src = source_idx_global.size
    N_tgt = target_idx_global.size
    print(f"target={target_code}: n_source={N_src}, n_target={N_tgt}")

    Z_src_eval = Z[source_mask][:, :downstream_n_pcs]
    Z_tgt_eval = Z[target_mask][:, :downstream_n_pcs]

    ga_raw = cfg.get("pevmean_ga", {})
    base_ga_cfg = GAConfig(
        pop_size=int(ga_raw.get("pop_size", 100)),
        n_generations=int(ga_raw.get("n_generations", 200)),
        n_elite=int(ga_raw.get("n_elite", 2)),
        tournament_k=int(ga_raw.get("tournament_k", 5)),
        crossover_prob=float(ga_raw.get("crossover_prob", 0.9)),
        mutation_prob=float(ga_raw.get("mutation_prob", 0.3)),
        n_swaps_per_mut=int(ga_raw.get("n_swaps_per_mut", 2)),
        seed=0,
        verbose=False,
        stagnation_limit=int(ga_raw.get("stagnation_limit", 20)),
    )
    n_jobs = int(ga_raw.get("n_jobs", 1))
    parallel_min_n_train = int(ga_raw.get("parallel_min_n_train", 1))
    cand_idx = np.arange(N_src, dtype=np.int64)

    rows = []
    for s in range(args.n_seeds):
        ga_cfg = GAConfig(**{**base_ga_cfg.__dict__, "seed": 1000 + s})
        if n_jobs > 1 and args.n_train >= parallel_min_n_train:
            def batch_fn(subsets):
                return pev_mean_batch(
                    Z_pev,
                    [source_idx_global[ss] for ss in subsets],
                    target_idx_global,
                    pevmean_lambda,
                    n_jobs=n_jobs,
                )
        else:
            batch_fn = None

        def fit_fn(subset):
            return pev_mean(Z_pev, source_idx_global[subset], target_idx_global, pevmean_lambda)

        t0 = time.perf_counter()
        best_subset, best_pev, stats = run_ga(
            n_candidates=N_src,
            n_train=args.n_train,
            fitness_fn=fit_fn,
            cfg=ga_cfg,
            candidate_indices=cand_idx,
            batch_fitness_fn=batch_fn,
        )
        dt_ga = time.perf_counter() - t0

        res = _evaluate_ridge_subset(
            train_idx=best_subset,
            X_source=Z_src_eval,
            y_source=y[source_mask],
            X_test=Z_tgt_eval,
            y_test=y[target_mask],
            y_eval_test=y_eval[target_mask],
            alpha=downstream_alpha,
        )
        rows.append({"seed": ga_cfg.seed, "best_pev": best_pev, "corr": res["corr_eval"],
                     "ga_time_s": dt_ga, "gens": stats.get("generations_run", None)})
        print(f"seed={ga_cfg.seed}  pev={best_pev:.4f}  corr={res['corr_eval']:.4f}  "
              f"time={dt_ga:.1f}s  gens={stats.get('generations_run', '?')}")

    arr_pev = np.array([r["best_pev"] for r in rows], dtype=float)
    arr_corr = np.array([r["corr"] for r in rows], dtype=float)
    print("\n=== Distribution across seeds ===")
    print(f"PEVmean: mean={arr_pev.mean():.4f}  std={arr_pev.std(ddof=1):.4f}  "
          f"range=[{arr_pev.min():.4f}, {arr_pev.max():.4f}]  CV={arr_pev.std(ddof=1)/arr_pev.mean():.4f}")
    print(f"Corr:    mean={arr_corr.mean():.4f}  std={arr_corr.std(ddof=1):.4f}  "
          f"range=[{arr_corr.min():.4f}, {arr_corr.max():.4f}]")


if __name__ == "__main__":
    main()
