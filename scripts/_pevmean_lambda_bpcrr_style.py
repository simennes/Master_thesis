"""Compute lambda for PEVmean using BPCRR-style parameterization:
   sigma_u*^2 = va_apriori / sum_j Var(PC_j)
   lambda     = sigma_e^2 / sigma_u*^2  = sigma_e^2 * sum_var_PC / va_apriori

This matches what `scripts/run_bpcrr_inla_rank_select.py` does in "paper" mode.

Usage:
    python -m scripts._pevmean_lambda_bpcrr_style
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.decomposition import PCA

from src.data import load_data

TRAITS = [
    ("body_mass", "Data/npz/snp_body_mass_ALL.npz", 1.35),
    ("thr_tarsus", "Data/npz/snp_thr_tarsus_ALL.npz", 0.27),
    ("thr_wing", "Data/npz/snp_thr_wing_ALL.npz", 1.61),
]
PC_GRID = [40, 60, 80, 100, 150, 200, 300, 500, 1000]


def main() -> None:
    print("BPCRR-style lambda for PEVmean:  lambda = sigma_e^2 * sum_var_PC / va_apriori")
    print()
    for name, npz, va in TRAITS:
        X, y_adj, ids, _grm, locality, code_to_label, y_mean = load_data(
            paths={"npz": npz},
            target_column="y_adjusted",
            standardize_features=False,
            return_locality=True,
            min_count=20,
            return_eval=True,
            eval_target_column="y_mean",
        )
        sigma_e2 = float(np.var(y_adj, ddof=1))
        sigma_e2_raw = float(np.var(y_mean, ddof=1))
        print(f"=== {name} ===")
        print(f"  n={X.shape[0]}  p={X.shape[1]}  va_apriori={va:.3f}")
        print(f"  sigma_e^2 (var y_adjusted) = {sigma_e2:.4f}  "
              f"(var y_mean = {sigma_e2_raw:.4f}, "
              f"naive sigma_e^2 = {max(sigma_e2_raw - va, 0):.4f})")

        # PCA up to the largest k.
        max_k = max(PC_GRID)
        Xc = X.astype(np.float32, copy=False)
        pca = PCA(n_components=min(max_k, Xc.shape[0], Xc.shape[1]),
                  svd_solver="randomized", random_state=14)
        Z = pca.fit_transform(Xc)
        evar = pca.explained_variance_  # = Var(PC_j) (unbiased, ddof=1 in sklearn>=1.0)
        cumvar_frac = np.cumsum(pca.explained_variance_ratio_)

        print(f"  {'k':>5} {'cumvar':>8} {'sum_var_PC':>12} {'lambda_y_adj':>14} {'lambda_y_mean':>14}")
        for k in PC_GRID:
            if k > len(evar):
                continue
            sum_var = float(np.sum(evar[:k]))
            lam_adj = sigma_e2 * sum_var / va
            lam_raw = max(sigma_e2_raw - va, 1e-9) * sum_var / va
            print(f"  {k:>5d} {float(cumvar_frac[k-1]):>8.3f} {sum_var:>12.2f} "
                  f"{lam_adj:>14.3e} {lam_raw:>14.3e}")
        print()


if __name__ == "__main__":
    main()
