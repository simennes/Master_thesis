"""
Evaluate top-k TMC individual Shapley selection with ridge regression.

For each target island in tmc_shapley_individuals outputs:
  - Load per-repeat Shapley CSVs and SNP columns
  - Reconstruct the same cal/test split (matching split_seed)
  - For each k, select top-k individuals by Shapley phi
  - Retrain ridge (same alpha as TMC run) on selected individuals and SNP subset
  - Evaluate on held-out test set (corr_eval vs y_mean)
  - Compute random_individual baseline and full_source baseline for the same SNP subset
  - Save results as CSV compatible with visualize_bpcrr_inla.ipynb

Usage
-----
    python -m scripts.run_tmc_individual_topk_eval \
        --config config/tmc_shapley_individuals.json \
        --k_values 50 100 250 500 1000 1500 2000 2500 3000 3500 4000 4500 \
        --n_random_orders 5 \
        --output outputs/tmc_shapley_individuals/tmc_individual_topk_results.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cv_utils import island_label
from src.data import load_data
from src.removal_curve import split_target_island
from src.utils import _pearson_corr, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_K_VALUES = [50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]


def _eval_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test_adj: np.ndarray,
    y_eval_test: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    if len(X_train) < 2:
        return {"corr_eval": 0.0, "mse_adj": float("inf")}
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    corr = float(_pearson_corr(pred, y_eval_test))
    if not np.isfinite(corr):
        corr = 0.0
    mse = float(np.mean((pred - y_test_adj) ** 2))
    return {"corr_eval": corr, "mse_adj": mse}


def _find_repeat_dirs(island_dir: Path, target_code: int) -> List[Tuple[int, Path]]:
    result = []
    for p in sorted(island_dir.glob("repeat_*")):
        try:
            rep_idx = int(p.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        phi_path = p / f"shapley_individual_values_target_{target_code}.csv"
        snp_path = p / f"snp_cols_target_{target_code}.csv"
        if phi_path.exists() and snp_path.exists():
            result.append((rep_idx, p))
    return sorted(result, key=lambda x: x[0])


def evaluate_target_island(
    target_code: int,
    target_name: str,
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    locality: np.ndarray,
    y_eval: np.ndarray,
    island_dir: Path,
    cfg: Dict[str, Any],
    k_values: List[int],
    n_random_orders: int,
    trait: str = "body_mass",
) -> List[Dict[str, Any]]:
    ridge_alpha = float(cfg["model"]["alpha"])
    cal_fraction = float(cfg.get("cal_fraction") or 0.1)
    max_cal_fraction = cfg.get("max_cal_fraction", None)
    n_cal_fixed = cfg.get("n_cal_fixed", None)
    seed = int(cfg.get("seed", 14))

    repeat_dirs = _find_repeat_dirs(island_dir, target_code)
    if not repeat_dirs:
        logger.warning("No complete repeat dirs found for island %d in %s", target_code, island_dir)
        return []

    # SNP cols are the same across repeats; load from the first available repeat.
    first_rep_dir = repeat_dirs[0][1]
    snp_cols = pd.read_csv(first_rep_dir / f"snp_cols_target_{target_code}.csv")["snp_col"].values.astype(np.int64)
    logger.info("Island %d: loaded %d SNP cols", target_code, len(snp_cols))

    # Source pool is deterministic (target split only randomises the target island rows).
    source_mask = locality != target_code
    X_source_all = X[source_mask]
    y_source_all = y[source_mask]

    rows: List[Dict[str, Any]] = []

    for rep_idx, rep_dir in repeat_dirs:
        split_seed = seed + 1_000 * rep_idx
        phi_df = pd.read_csv(rep_dir / f"shapley_individual_values_target_{target_code}.csv")

        split = split_target_island(
            X, y, ids, locality, target_code,
            y_eval=y_eval,
            cal_fraction=cal_fraction,
            seed=split_seed,
            max_cal_fraction=max_cal_fraction,
            n_cal_fixed=n_cal_fixed,
        )
        X_test = split["X_test"][:, snp_cols]
        y_test_adj = split["y_test"]
        y_eval_test = split["y_eval_test"]

        # Sort phi_df by rank (ascending, so rank 1 = best individual is first).
        phi_sorted = phi_df.sort_values("rank_phi").reset_index(drop=True)
        pool_indices = phi_sorted["source_pool_index"].values.astype(np.int64)
        n_pool = len(pool_indices)

        def _base_row():
            return {
                "target_island": int(target_code),
                "target_island_name": str(target_name),
                "repeat": int(rep_idx),
                "repeat_seed": int(split_seed),
                "trait": trait,
                "model": "ridge_tmc_individual",
                "weight_scheme": "none",
                "order_seed": -1,
                "weighted_fit_used": False,
            }

        # Full-source baseline: all source individuals (with SNP selection).
        full_indices = pool_indices
        X_full = X_source_all[full_indices][:, snp_cols]
        y_full = y_source_all[full_indices]
        full_res = _eval_ridge(X_full, y_full, X_test, y_test_adj, y_eval_test, ridge_alpha)
        row_full = _base_row()
        row_full.update({
            "analysis": "full_baseline",
            "method": "full_source_unweighted",
            "n_individuals": int(n_pool),
            "order_seed": -2,
            **full_res,
        })
        rows.append(row_full)

        # Top-k Shapley selection.
        for k in k_values:
            if k > n_pool:
                continue
            chosen = np.sort(pool_indices[:k])
            X_train = X_source_all[chosen][:, snp_cols]
            y_train = y_source_all[chosen]
            res = _eval_ridge(X_train, y_train, X_test, y_test_adj, y_eval_test, ridge_alpha)
            row = _base_row()
            row.update({
                "analysis": "ranked_subset",
                "method": "tmc_shapley_topk",
                "n_individuals": int(k),
                **res,
            })
            rows.append(row)

        # Random baselines: multiple random orderings of the source pool.
        rng = np.random.default_rng(split_seed + 999_999)
        for r_ord in range(n_random_orders):
            rand_order = rng.permutation(n_pool)
            rand_pool = pool_indices[rand_order]
            for k in k_values:
                if k > n_pool:
                    continue
                chosen = np.sort(rand_pool[:k])
                X_train = X_source_all[chosen][:, snp_cols]
                y_train = y_source_all[chosen]
                res = _eval_ridge(X_train, y_train, X_test, y_test_adj, y_eval_test, ridge_alpha)
                row = _base_row()
                row.update({
                    "analysis": "ranked_subset",
                    "method": "random_individual",
                    "n_individuals": int(k),
                    "order_seed": int(r_ord),
                    **res,
                })
                rows.append(row)

        logger.info(
            "Island %d (%s) repeat %d done — %d k-values, %d random orders",
            target_code, target_name, rep_idx, len(k_values), n_random_orders,
        )

    return rows


def compute_shap_per_island_stats(
    tmc_output_root: Path,
    target_codes: List[int],
) -> pd.DataFrame:
    """Average Shapley phi per source island, for each target island."""
    parts = []
    for target_code in target_codes:
        island_dir = tmc_output_root / f"island_{target_code}"
        for rep_dir in sorted(island_dir.glob("repeat_*")):
            try:
                rep_idx = int(rep_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            phi_path = rep_dir / f"shapley_individual_values_target_{target_code}.csv"
            if not phi_path.exists():
                continue
            df = pd.read_csv(phi_path)
            df["target_island"] = target_code
            parts.append(df)

    if not parts:
        return pd.DataFrame()

    all_phi = pd.concat(parts, ignore_index=True)

    stats = (
        all_phi.groupby(
            ["target_island", "target_island_name", "source_island", "source_island_name"],
            as_index=False,
        )
        .agg(
            phi_mean=("phi", "mean"),
            phi_std=("phi", "std"),
            phi_median=("phi", "median"),
            phi_p25=("phi", lambda x: np.quantile(x, 0.25)),
            phi_p75=("phi", lambda x: np.quantile(x, 0.75)),
            n_individuals=("phi", "count"),
            n_repeats=("repeat", "nunique"),
        )
        .sort_values(["target_island", "phi_mean"], ascending=[True, False])
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate top-k TMC Shapley ridge selection")
    parser.add_argument("--config", default="config/tmc_shapley_individuals.json")
    parser.add_argument(
        "--k_values", nargs="+", type=int,
        default=DEFAULT_K_VALUES,
    )
    parser.add_argument("--n_random_orders", type=int, default=5)
    parser.add_argument(
        "--output",
        default="outputs/tmc_shapley_individuals/tmc_individual_topk_results.csv",
    )
    parser.add_argument(
        "--stats_output",
        default="outputs/tmc_shapley_individuals/tmc_individual_shap_per_source_island.csv",
    )
    parser.add_argument("--trait", default="body_mass")
    parser.add_argument(
        "--target_islands", nargs="+", type=int, default=None,
        help="Override target_islands from config (subset to evaluate).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = json.load(f)

    seed = int(cfg.get("seed", 14))
    set_seed(seed)

    paths = cfg["paths"]
    logger.info("Loading data...")
    X, y, ids, _GRM_df, locality, code_to_label, y_eval = load_data(
        paths=paths,
        target_column=cfg.get("target_column", "y_adjusted"),
        standardize_features=bool(cfg.get("standardize_features", True)),
        return_locality=True,
        min_count=int(cfg.get("min_count", 20)),
        return_eval=True,
        eval_target_column=cfg.get("eval_target_column", "y_mean"),
    )
    logger.info("Data: %d samples, %d SNPs", X.shape[0], X.shape[1])

    tmc_output_root = Path(cfg["paths"]["output_dir"])
    if args.target_islands is not None:
        target_codes = [int(t) for t in args.target_islands]
    else:
        target_codes = [int(t) for t in cfg.get("target_islands", [])]
    k_values = sorted(set(int(k) for k in args.k_values))
    logger.info("Target islands: %s", target_codes)
    logger.info("k values: %s", k_values)

    all_rows: List[Dict[str, Any]] = []
    for target_code in target_codes:
        island_dir = tmc_output_root / f"island_{target_code}"
        if not island_dir.exists():
            logger.warning("No output dir for island %d; skipping", target_code)
            continue
        target_name = island_label(target_code, code_to_label)
        logger.info("=" * 60)
        logger.info("Evaluating island %d (%s)", target_code, target_name)
        rows = evaluate_target_island(
            target_code=target_code,
            target_name=target_name,
            X=X,
            y=y,
            ids=ids,
            locality=locality,
            y_eval=y_eval,
            island_dir=island_dir,
            cfg=cfg,
            k_values=k_values,
            n_random_orders=args.n_random_orders,
            trait=args.trait,
        )
        all_rows.extend(rows)
        logger.info("Island %d: %d result rows", target_code, len(rows))

    if not all_rows:
        logger.error("No results produced; check tmc_shapley_individuals outputs exist.")
        return

    results_df = pd.DataFrame(all_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(results_df), out_path)

    # Print a quick summary table.
    summary = (
        results_df[results_df["method"] == "tmc_shapley_topk"]
        .groupby(["target_island", "target_island_name", "n_individuals"], as_index=False)["corr_eval"]
        .mean()
        .rename(columns={"corr_eval": "mean_corr_eval"})
    )
    print("\nTop-k Shapley selection summary (mean over repeats):")
    print(summary.to_string(index=False))

    # Shapley stats per source island.
    logger.info("Computing Shapley statistics per source island...")
    stats_df = compute_shap_per_island_stats(tmc_output_root, target_codes)
    if not stats_df.empty:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(stats_path, index=False)
        logger.info("Saved Shapley source-island stats to %s", stats_path)
        print("\nShapley stats per source island (top rows):")
        print(stats_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
