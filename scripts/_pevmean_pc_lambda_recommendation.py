"""Compute PC cumulative-variance curves and trait-specific lambda recommendations
for the E3 PEVmean-GA setup. Not part of the experiment pipeline.

Usage:
    python -m scripts._pevmean_pc_lambda_recommendation
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.decomposition import PCA

from src.data import load_data


# va_apriori = additive genetic variance from BPCRR (Aspheim 2024 / final BPCRR configs)
TRAITS = [
    ("body_mass", "Data/npz/snp_body_mass_ALL.npz", 1.35),
    ("thr_tarsus", "Data/npz/snp_thr_tarsus_ALL.npz", 0.27),
    ("thr_wing", "Data/npz/snp_thr_wing_ALL.npz", 1.61),
]

# PCs at which to report cumulative variance.
PC_GRID = [20, 40, 60, 80, 100, 150, 200, 300, 500]


def main() -> None:
    print(f"{'trait':<12} {'n':>5} {'p':>7} {'var(y_adj)':>11} {'sigma2_G':>10} "
          f"{'sigma2_E':>10} {'h2':>6} {'lambda':>10}")
    print("-" * 90)

    trait_results = []
    for name, npz, sigma2_G in TRAITS:
        X, y_adj, ids, _grm, locality, code_to_label, y_mean = load_data(
            paths={"npz": npz},
            target_column="y_adjusted",
            standardize_features=False,
            return_locality=True,
            min_count=20,
            return_eval=True,
            eval_target_column="y_mean",
        )

        # Use unadjusted phenotype variance for sigma2_P (the y_adjusted has
        # most fixed effects already removed and is closer to the genetic-only scale).
        sigma2_P = float(np.var(y_mean, ddof=1))
        sigma2_E = max(sigma2_P - sigma2_G, 1e-9)
        h2 = sigma2_G / sigma2_P
        lam = sigma2_E / sigma2_G

        print(f"{name:<12} {X.shape[0]:>5} {X.shape[1]:>7} {sigma2_P:>11.3f} "
              f"{sigma2_G:>10.3f} {sigma2_E:>10.3f} {h2:>6.3f} {lam:>10.3e}")

        trait_results.append({
            "name": name,
            "X": X,
            "sigma2_P": sigma2_P,
            "sigma2_G": sigma2_G,
            "sigma2_E": sigma2_E,
            "h2": h2,
            "lam": lam,
        })

    print()
    print("Cumulative variance explained by first k PCs (randomized PCA, seed=14):")
    print(f"{'k':>5} | " + " | ".join(f"{t['name']:>10}" for t in trait_results))
    print("-" * 60)

    # Compute cumulative variance once per trait at the largest k, then slice.
    max_k = max(PC_GRID)
    cumvar_by_trait = {}
    for t in trait_results:
        Xc = t["X"].astype(np.float32, copy=False)
        feasible = min(max_k, Xc.shape[0], Xc.shape[1])
        pca = PCA(n_components=feasible, svd_solver="randomized", random_state=14)
        pca.fit(Xc)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        cumvar_by_trait[t["name"]] = cumvar

    for k in PC_GRID:
        row = f"{k:>5} | "
        cells = []
        for t in trait_results:
            cv = cumvar_by_trait[t["name"]]
            v = float(cv[k - 1]) if k <= len(cv) else float("nan")
            cells.append(f"{v:>10.3f}")
        print(row + " | ".join(cells))

    # Suggest two natural choices for k.
    print()
    print("Smallest k reaching variance thresholds:")
    print(f"{'threshold':>9} | " + " | ".join(f"{t['name']:>10}" for t in trait_results))
    print("-" * 60)
    for thr in [0.50, 0.75, 0.90, 0.95, 0.99]:
        row = f"{thr:>9.2f} | "
        cells = []
        for t in trait_results:
            cv = cumvar_by_trait[t["name"]]
            idx = int(np.searchsorted(cv, thr) + 1)
            cells.append(f"{idx:>10d}")
        print(row + " | ".join(cells))


if __name__ == "__main__":
    main()
