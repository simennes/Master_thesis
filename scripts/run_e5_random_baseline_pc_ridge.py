#!/usr/bin/env python
"""Native random baseline for E5 island-level Shapley add-curves.

This evaluates random individual subsets on the same target calibration/test
splits and at the same cumulative sample sizes as the E5 Shapley add-curves.
The output is a per-trait CSV loaded by ``plot_e5_shapley_thesis_figures.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from scripts.run_e5_shapley_islands_pc_ridge import (
    _compute_pcs,
    _fixed_cal_test_split,
    _trait_specs,
    resolve_island_code,
)
from src.cv_utils import island_label
from src.data import load_data
from src.tmc_shapley_islands import _pearson_corr
from src.utils import set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _existing_keys(path: Path) -> set[tuple[int, int, int, int]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["target_island", "repeat", "n_islands", "order_seed"])
    return {
        (int(row.target_island), int(row.repeat), int(row.n_islands), int(row.order_seed))
        for row in df.itertuples(index=False)
    }


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", index=False, header=not path.exists())


def _fit_eval_ridge(
    X_pcs: np.ndarray,
    y: np.ndarray,
    y_eval: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    if len(train_idx) < 2 or len(test_idx) < 2:
        return {"corr_eval": 0.0, "mse_adj": float("inf")}
    model = Ridge(alpha=max(float(alpha), 1e-12))
    model.fit(X_pcs[train_idx], y[train_idx])
    pred = model.predict(X_pcs[test_idx])
    corr_eval = float(_pearson_corr(pred, y_eval[test_idx]))
    if not np.isfinite(corr_eval):
        corr_eval = 0.0
    mse_adj = float(np.mean((pred - y[test_idx]) ** 2))
    return {"corr_eval": corr_eval, "mse_adj": mse_adj}


def run_trait_random_baseline(
    spec: dict[str, Any],
    cfg: dict[str, Any],
    output_root: Path,
    n_random_orders: int,
    overwrite: bool = False,
) -> Path:
    global_seed = int(cfg.get("seed", 14))
    n_cal_fixed = int(cfg.get("cal_n_fixed", 25))
    trait_root = output_root / spec["name"]
    add_path = trait_root / "add_curve_repeats_all_targets.csv"
    if not add_path.exists():
        raise FileNotFoundError(add_path)

    random_path = trait_root / "random_baseline_repeats_all_targets.csv"
    if overwrite and random_path.exists():
        random_path.unlink()
    done = _existing_keys(random_path)

    logger.info("Loading trait '%s'", spec["name"])
    X, y, ids, _grm, locality, code_to_label, y_eval = load_data(
        paths=spec["paths"],
        target_column=spec["target_column"],
        standardize_features=spec["standardize_features"],
        return_locality=True,
        min_count=spec["min_count"],
        return_eval=True,
        eval_target_column=spec["eval_target_column"],
    )
    if y_eval is None:
        y_eval = y.copy()
    locality = locality.astype(int)
    present_codes = set(int(c) for c in np.unique(locality))

    included_raw = cfg.get("included_islands", None)
    if included_raw is not None:
        included_codes = [resolve_island_code(v, code_to_label, present_codes) for v in included_raw]
    else:
        included_codes = sorted(present_codes)

    logger.info("Computing PCs for trait '%s'", spec["name"])
    X_pcs, cumvar, pca_time = _compute_pcs(
        X=X,
        n_pcs=spec["n_pcs"],
        seed=global_seed,
        standardize=spec["standardize_features"],
    )
    logger.info(
        "Trait '%s': PCA r=%d in %.1fs (cumvar=%.4f)",
        spec["name"],
        X_pcs.shape[1],
        pca_time,
        cumvar,
    )

    shap_add = pd.read_csv(add_path)
    shap_add = shap_add[shap_add["method"].eq("shapley_mean")].copy()
    shap_add = shap_add.sort_values(["target_island", "repeat", "n_islands"])
    total_jobs = shap_add.groupby(["target_island", "repeat"]).ngroups * int(n_random_orders)
    logger.info("Trait '%s': evaluating %d random orders", spec["name"], total_jobs)

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    n_written = 0
    for (target_code, repeat_idx), curve in shap_add.groupby(["target_island", "repeat"], sort=True):
        target_code = int(target_code)
        repeat_idx = int(repeat_idx)
        target_mask_positions = np.flatnonzero(locality == target_code)
        if len(target_mask_positions) <= n_cal_fixed:
            logger.warning("Skipping target %s repeat %d because target is too small", target_code, repeat_idx)
            continue

        split_seed = int(curve["split_seed"].iloc[0])
        _cal_local, test_local = _fixed_cal_test_split(len(target_mask_positions), n_cal_fixed, split_seed)
        test_idx = target_mask_positions[test_local]

        source_idx = np.flatnonzero(np.isin(locality, included_codes) & (locality != target_code))
        target_name = str(curve["target_island_name"].iloc[0]) if "target_island_name" in curve.columns else island_label(target_code, code_to_label)

        full_row = curve.loc[curve["n_islands"].idxmax()]
        for order_seed in range(int(n_random_orders)):
            rng = np.random.default_rng(global_seed + 500_000 + target_code * 10_000 + repeat_idx * 100 + order_seed)
            shuffled = rng.permutation(source_idx)
            for step in curve.itertuples(index=False):
                n_islands = int(step.n_islands)
                key = (target_code, repeat_idx, n_islands, order_seed)
                if key in done:
                    continue
                n_include = min(int(step.n_individuals), len(source_idx))
                if n_islands == int(full_row.n_islands):
                    corr_eval = float(full_row.corr_eval)
                    mse_adj = float(full_row.mse_adj)
                else:
                    train_idx = np.sort(shuffled[:n_include]).astype(np.int64)
                    metrics = _fit_eval_ridge(
                        X_pcs=X_pcs,
                        y=y,
                        y_eval=y_eval,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        alpha=float(spec["alpha"]),
                    )
                    corr_eval = metrics["corr_eval"]
                    mse_adj = metrics["mse_adj"]
                rows.append(
                    {
                        "trait": spec["name"],
                        "target_island": target_code,
                        "target_island_name": target_name,
                        "repeat": repeat_idx,
                        "split_seed": split_seed,
                        "n_islands": n_islands,
                        "n_individuals": n_include,
                        "corr_eval": corr_eval,
                        "mse_adj": mse_adj,
                        "method": "random_individual",
                        "order_seed": order_seed,
                    }
                )
                if len(rows) >= 200:
                    _append_rows(random_path, rows)
                    n_written += len(rows)
                    rows = []
            done.add((target_code, repeat_idx, int(full_row.n_islands), order_seed))

    _append_rows(random_path, rows)
    n_written += len(rows)
    logger.info(
        "Trait '%s': wrote %d new rows to %s in %.1fs",
        spec["name"],
        n_written,
        random_path,
        time.perf_counter() - t0,
    )
    return random_path


def run(config_path: Path, only_traits: set[str] | None, n_random_orders: int, overwrite: bool) -> list[Path]:
    cfg = _load_json(config_path)
    set_seed(int(cfg.get("seed", 14)))
    output_root = Path(cfg["output"].get("root_dir", "outputs/final_results/e5_shapley_islands_pc_ridge"))
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    paths = []
    for spec in _trait_specs(cfg, only_traits):
        paths.append(
            run_trait_random_baseline(
                spec=spec,
                cfg=cfg,
                output_root=output_root,
                n_random_orders=n_random_orders,
                overwrite=overwrite,
            )
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native random baseline for E5 Shapley add-curves.")
    parser.add_argument("--config", default="config/e5/final_e5_shapley_islands_pc_ridge_config.json")
    parser.add_argument("--trait", action="append", default=None, help="Trait to run. Can be passed multiple times.")
    parser.add_argument("--n-random-orders", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traits = set(args.trait) if args.trait else None
    paths = run(
        config_path=Path(args.config),
        only_traits=traits,
        n_random_orders=int(args.n_random_orders),
        overwrite=bool(args.overwrite),
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
