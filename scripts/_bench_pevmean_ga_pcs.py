"""Quick benchmark for E3 PEVmean-GA with PC feature space.

Runs one full GA cycle (one target island, one repeat, one subset size) using
the same code path the real runner uses, and prints wall-time totals so we
can extrapolate to the full job. Not part of the experiment pipeline -- intended
to be invoked manually before launching SLURM jobs.

Usage:
    python -m scripts._bench_pevmean_ga_pcs --config config/final_e3_pevmean_ga_body_mass_config.json \
        --n_train 2500 --target_island 0
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
from src.training_set_optimization.pevmean import build_kernel, pev_mean, pev_mean_batch
from src.training_set_optimization.runner import _evaluate_ridge_subset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_train", type=int, default=2500)
    ap.add_argument("--target_island", type=int, default=0)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    trait_specs = _build_trait_specs(cfg)
    trait = trait_specs[0]
    print(f"Trait: {trait['name']}  downstream_ridge: {trait['downstream_ridge']}")

    t0 = time.perf_counter()
    X, y, ids, _grm, locality, code_to_label, y_eval = load_data(
        paths=trait["paths"],
        target_column=trait["target_column"],
        standardize_features=trait["standardize_features"],
        return_locality=True,
        min_count=trait["min_count"],
        return_eval=True,
        eval_target_column=trait["eval_target_column"],
    )
    print(f"load_data: {time.perf_counter()-t0:.1f}s  X.shape={X.shape}")

    feature_space = cfg.get("feature_space", {})
    n_pcs_for_kernel = int(feature_space.get("n_pcs_for_kernel", 5000))
    downstream_n_pcs = int(trait["downstream_ridge"]["n_pcs"])
    max_r = max(n_pcs_for_kernel, downstream_n_pcs)

    Z, var_ratio, dt = _compute_dataset_pcs(
        X=X, n_pcs=max_r,
        seed=int(feature_space.get("pca_seed", cfg.get("seed", 14))),
        standardize=bool(feature_space.get("standardize_for_pca", False)),
    )
    print(f"PCA fit (r={max_r}): {dt:.1f}s  cumvar={float(np.sum(var_ratio)):.3f}")

    target_code = resolve_island_code(args.target_island, code_to_label, set(int(c) for c in np.unique(locality)))
    target_mask = (locality == target_code)
    source_mask = ~target_mask

    Z_src = Z[source_mask][:, :n_pcs_for_kernel]
    Z_tgt = Z[target_mask][:, :n_pcs_for_kernel]
    N_src, N_tgt = Z_src.shape[0], Z_tgt.shape[0]
    print(f"target={target_code}: n_source={N_src}, n_target={N_tgt}")

    t0 = time.perf_counter()
    Z_all = np.vstack([Z_src, Z_tgt])
    K, dK = build_kernel(Z_all, dtype=np.float32)
    print(f"build_kernel: {time.perf_counter()-t0:.1f}s  K.shape={K.shape}  K.dtype={K.dtype}")

    cand_idx = np.arange(N_src, dtype=np.int64)
    target_idx = np.arange(N_src, N_src + N_tgt, dtype=np.int64)
    lam = float(trait["downstream_ridge"]["alpha"])
    print(f"PEVmean lambda = downstream alpha = {lam:.3e}")

    # One fitness eval on a random subset to get a unit cost.
    rng = np.random.default_rng(0)
    rand_subset = np.sort(rng.choice(N_src, size=args.n_train, replace=False))
    t0 = time.perf_counter()
    val = pev_mean(K, dK, rand_subset, target_idx, lam)
    dt_one = time.perf_counter() - t0
    print(f"single pev_mean (n_train={args.n_train}): {dt_one*1000:.0f} ms  -> {val:.4f}")

    # One GA run with the config's GA settings.
    ga_raw = cfg.get("pevmean_ga", {})
    ga_cfg = GAConfig(
        pop_size=int(ga_raw.get("pop_size", 100)),
        n_generations=int(ga_raw.get("n_generations", 200)),
        n_elite=int(ga_raw.get("n_elite", 2)),
        tournament_k=int(ga_raw.get("tournament_k", 5)),
        crossover_prob=float(ga_raw.get("crossover_prob", 0.9)),
        mutation_prob=float(ga_raw.get("mutation_prob", 0.3)),
        n_swaps_per_mut=int(ga_raw.get("n_swaps_per_mut", 2)),
        seed=42,
        verbose=False,
        stagnation_limit=int(ga_raw.get("stagnation_limit", 20)),
    )
    print(f"GAConfig: pop={ga_cfg.pop_size} gens={ga_cfg.n_generations} stag={ga_cfg.stagnation_limit}")

    n_jobs = int(ga_raw.get("n_jobs", 1))
    parallel_min_n_train = int(ga_raw.get("parallel_min_n_train", 1000))
    if n_jobs > 1 and args.n_train >= parallel_min_n_train:
        def batch_fn(subsets):
            return pev_mean_batch(K, dK, subsets, target_idx, lam, n_jobs=n_jobs)
    else:
        batch_fn = None

    def fit_fn(subset):
        return pev_mean(K, dK, subset, target_idx, lam)

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
    print(
        f"run_ga (n_train={args.n_train}): {dt_ga:.1f}s  best_pev={best_pev:.4f}  "
        f"stats={stats.get('generations_run', '?')} gens"
    )

    # Downstream PC-ridge eval.
    Z_src_eval = Z[source_mask][:, :downstream_n_pcs]
    Z_tgt_eval = Z[target_mask][:, :downstream_n_pcs]
    t0 = time.perf_counter()
    res = _evaluate_ridge_subset(
        train_idx=best_subset,
        X_source=Z_src_eval,
        y_source=y[source_mask],
        X_test=Z_tgt_eval,
        y_test=y[target_mask],
        y_eval_test=y_eval[target_mask],
        alpha=float(trait["downstream_ridge"]["alpha"]),
    )
    print(
        f"ridge_eval (r={downstream_n_pcs}, alpha={trait['downstream_ridge']['alpha']:.3e}): "
        f"{time.perf_counter()-t0:.1f}s  corr={res['corr_eval']:.4f}"
    )

    # Rough extrapolation: 15 targets x 5 repeats x 9 sizes = 675 GA runs per trait.
    n_runs = 15 * int(ga_raw.get("n_repeats", 5)) * len(ga_raw.get("n_train_sizes", []))
    if n_runs > 0:
        est = dt_ga * n_runs
        print(
            f"\nIf every GA run cost {dt_ga:.1f}s (this n_train={args.n_train}): "
            f"~{est/3600:.1f} h total for one trait."
        )
        print(
            "Note: smaller subsets are much cheaper; real total will be lower. "
            "Run --n_train 4500 to see the worst case."
        )


if __name__ == "__main__":
    main()
