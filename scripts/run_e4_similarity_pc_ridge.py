#!/usr/bin/env python
"""E4: Similarity-based subset selection evaluated with tuned PC ridge.

For each trait and target island we build three deterministic orderings of
the source pool and evaluate PC ridge on top-k subsets:

    * avggrm_topk          — rank source individuals by avg-GRM to target
                              (descending; closer relatives first).
    * pca_target_topk      — rank by Euclidean distance to target centroid
                              in the first ``pca_distance_pcs`` global PCs
                              (ascending; closest in PC space first).
    * avggrm_diversity     — greedy maximisation of
                              avgGRM(train, target) - lambda_div * avgGRM(train, train)
                              over the source pool. One method per lambda
                              in ``lambda_divs`` (suffix ``_lamX``).

PCs and ridge alpha come from the per-trait tuned settings (E1 fixed).

Outputs
-------
``outputs/final_results/e4_similarity_pc_ridge/<trait>/e4_similarity_pc_ridge_results.csv``
matches the schema produced by ``run_pc_ridge_subsets.py`` so the notebook
can ingest both via the same loader.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from src.avggrm_weighting import (
    avg_grm_train_to_target,
    greedy_avggrm_diversity_order,
)
from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.utils import _pearson_corr, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_island_code(target_island, code_to_label: dict, present_codes: set) -> int:
    try:
        val = int(target_island)
    except (ValueError, TypeError):
        val = None
    if val is not None:
        if val in present_codes:
            return val
        for code, label in code_to_label.items():
            if int(label) == val:
                return code
        raise ValueError(f"Island {target_island} not found in dataset")
    if isinstance(target_island, str):
        target_lower = target_island.lower()
        for orig_label, name in ISLAND_ID_TO_NAME.items():
            if name.lower() == target_lower:
                for code, lbl in code_to_label.items():
                    if int(lbl) == orig_label:
                        return code
        for code, label in code_to_label.items():
            if str(label).lower() == target_lower:
                return code
    raise ValueError(f"Could not resolve target island: {target_island!r}")


def _trait_specs(cfg: dict[str, Any], only_traits: set[str] | None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    base_paths = dict(cfg.get("paths", {}))
    for raw in cfg.get("traits", []):
        name = str(raw["name"])
        if only_traits is not None and name not in only_traits:
            continue
        paths = dict(base_paths)
        paths["npz"] = str(raw["npz"])
        if "grm_rds" in raw:
            paths["grm_rds"] = str(raw["grm_rds"])
        specs.append({
            "name": name,
            "paths": paths,
            "n_pcs": int(raw["n_pcs"]),
            "alpha": float(raw["alpha"]),
            "target_column": raw.get("target_column", cfg.get("target_column", "y_adjusted")),
            "eval_target_column": raw.get("eval_target_column", cfg.get("eval_target_column", "y_mean")),
            "standardize_features": bool(raw.get("standardize_features", cfg.get("standardize_features", False))),
            "min_count": int(raw.get("min_count", cfg.get("min_count", 20))),
        })
    if not specs:
        raise ValueError("No traits selected for E4 run")
    return specs


def _compute_pcs(X: np.ndarray, n_pcs: int, seed: int, standardize: bool) -> tuple[np.ndarray, float]:
    n_components = int(min(n_pcs, X.shape[0], X.shape[1]))
    if n_components < 1:
        raise ValueError("n_pcs must be >= 1")
    if standardize:
        mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = X.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std == 0] = 1.0
        X_in = (X - mean) / std
    else:
        X_in = X
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed))
    t0 = time.perf_counter()
    scores = pca.fit_transform(X_in).astype(np.float32, copy=False)
    elapsed = float(time.perf_counter() - t0)
    return scores, float(np.sum(pca.explained_variance_ratio_)), elapsed


def _fit_eval(
    X_pcs: np.ndarray,
    y: np.ndarray,
    y_eval: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    if len(train_idx) == 0:
        raise ValueError("Empty training subset")
    if len(test_idx) == 0:
        raise ValueError("Empty target set")
    model = Ridge(alpha=max(float(alpha), 1e-12))
    t0 = time.perf_counter()
    model.fit(X_pcs[train_idx], y[train_idx])
    pred = model.predict(X_pcs[test_idx])
    fit_time = float(time.perf_counter() - t0)
    r = float(_pearson_corr(y_eval[test_idx], pred))
    if not np.isfinite(r):
        r = 0.0
    mse = float(np.mean((y_eval[test_idx] - pred) ** 2))
    return {"pearson_r": r, "mse_adj": mse, "fit_time_seconds": fit_time, "n_test": int(len(test_idx))}


def _format_lambda_suffix(lam: float) -> str:
    return f"lam{lam:g}".replace(".", "p").replace("-", "neg")


def _run_for_trait(
    spec: dict[str, Any],
    cfg: dict[str, Any],
    output_root: Path,
    seed: int,
) -> None:
    file_stem = str(cfg["output"].get("file_stem", "e4_similarity_pc_ridge"))
    trait_dir = output_root / spec["name"]
    trait_dir.mkdir(parents=True, exist_ok=True)
    results_path = trait_dir / f"{file_stem}_results.csv"
    summary_path = trait_dir / f"{file_stem}_summary.csv"
    selected_path = trait_dir / f"{file_stem}_selected_individuals.csv"
    for p in (results_path, summary_path, selected_path):
        if p.exists():
            p.unlink()

    sim_cfg = cfg.get("similarity", {})
    n_train_sizes = sorted(int(x) for x in sim_cfg.get("n_train_sizes", [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]))
    pca_distance_pcs = int(sim_cfg.get("pca_distance_pcs", 20))
    lambda_divs = [float(x) for x in sim_cfg.get("lambda_divs", [0.5])]
    include_diagonal = bool(sim_cfg.get("diversity_include_diagonal", True))
    write_selected = bool(sim_cfg.get("write_selected_individuals", False))

    logger.info(
        "Trait '%s': n_pcs=%d alpha=%.4g n_train_sizes=%s lambda_divs=%s",
        spec["name"], spec["n_pcs"], spec["alpha"], n_train_sizes, lambda_divs,
    )

    X, y, ids, GRM_df, locality, code_to_label, y_eval = load_data(
        paths=spec["paths"],
        target_column=spec["target_column"],
        standardize_features=spec["standardize_features"],
        return_locality=True,
        min_count=spec["min_count"],
        return_eval=True,
        eval_target_column=spec["eval_target_column"],
    )
    if GRM_df is None:
        raise ValueError(
            f"Trait '{spec['name']}': GRM is required. Provide paths.grm_rds in config."
        )
    if y_eval is None:
        y_eval = y.copy()
    ids = ids.astype(str)
    locality = locality.astype(int)

    present_codes = set(int(c) for c in np.unique(locality))
    included_raw = cfg.get("included_islands", None)
    if included_raw is not None:
        included_codes = [_resolve_island_code(v, code_to_label, present_codes) for v in included_raw]
    else:
        included_codes = sorted(present_codes)
    included_mask = np.isin(locality, included_codes)
    if not np.all(included_mask):
        X = X[included_mask]
        y = y[included_mask]
        y_eval = y_eval[included_mask]
        kept_ids = ids[included_mask]
        locality = locality[included_mask]
        GRM_df = GRM_df.loc[kept_ids, kept_ids]
        ids = kept_ids

    raw_targets = cfg.get("target_islands", sorted(included_codes))
    target_codes = [_resolve_island_code(t, code_to_label, present_codes) for t in raw_targets]

    X_pcs, var_explained, pca_time = _compute_pcs(
        X=X, n_pcs=spec["n_pcs"], seed=seed, standardize=spec["standardize_features"],
    )
    logger.info(
        "Trait '%s': PCA fit r=%d in %.2fs (cumvar=%.4f); using first %d PCs for distance",
        spec["name"], X_pcs.shape[1], pca_time, var_explained, min(pca_distance_pcs, X_pcs.shape[1]),
    )
    pca_distance_pcs = int(min(pca_distance_pcs, X_pcs.shape[1]))

    grm_mat = GRM_df.to_numpy(dtype=np.float64)
    id_to_idx = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids)

    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for target_code in target_codes:
        target_name = island_label(int(target_code), code_to_label)
        target_mask_full = (locality == int(target_code))
        source_mask_full = ~target_mask_full
        source_idx_global = np.flatnonzero(source_mask_full).astype(np.int64)
        target_idx_global = np.flatnonzero(target_mask_full).astype(np.int64)
        n_source = int(source_idx_global.size)
        n_target = int(target_idx_global.size)
        if n_source < 2 or n_target == 0:
            logger.info("Target %s: skipping (n_source=%d, n_target=%d)", target_code, n_source, n_target)
            continue

        logger.info("Target %s (%s): n_source=%d n_target=%d", target_code, target_name, n_source, n_target)

        # --- avg-GRM scores ----------------------------------------------------
        avg_grm = avg_grm_train_to_target(grm_mat, source_idx_global, target_idx_global)
        avg_grm = np.asarray(avg_grm, dtype=float)

        # --- avg-GRM top-k ordering -------------------------------------------
        order_avggrm = np.argsort(-avg_grm, kind="mergesort").astype(np.int64)

        # --- PC-distance ordering (closest to target centroid first) ----------
        target_centroid = X_pcs[target_idx_global, :pca_distance_pcs].mean(axis=0)
        source_pc = X_pcs[source_idx_global, :pca_distance_pcs]
        pc_distance = np.linalg.norm(
            source_pc.astype(np.float64) - target_centroid.astype(np.float64)[None, :],
            axis=1,
        )
        order_pca_distance = np.argsort(pc_distance, kind="mergesort").astype(np.int64)

        # --- avg-GRM diversity orderings (one per lambda) ---------------------
        max_k = int(min(max(n_train_sizes), n_source))
        grm_src_src = grm_mat[np.ix_(source_idx_global, source_idx_global)]
        diversity_orders: dict[float, np.ndarray] = {}
        for lam in lambda_divs:
            res = greedy_avggrm_diversity_order(
                avg_grm_to_target=avg_grm,
                train_train_grm=grm_src_src,
                lambda_div=float(lam),
                max_size=max_k,
                include_diagonal=include_diagonal,
            )
            diversity_orders[float(lam)] = np.asarray(res["order"], dtype=np.int64)

        # --- Evaluate every ordering at every k --------------------------------
        methods_and_orders: list[tuple[str, np.ndarray]] = [
            ("avggrm_topk", order_avggrm),
            ("pca_target_topk", order_pca_distance),
        ]
        for lam, order_div in diversity_orders.items():
            suffix = _format_lambda_suffix(lam)
            methods_and_orders.append((f"avggrm_diversity_{suffix}", order_div))

        for method_name, ordering in methods_and_orders:
            for k in n_train_sizes:
                k_eff = int(min(int(k), int(ordering.size), n_source))
                if k_eff < 1:
                    continue
                chosen_local = ordering[:k_eff]
                train_idx_global = source_idx_global[chosen_local]
                metrics = _fit_eval(
                    X_pcs=X_pcs,
                    y=y,
                    y_eval=y_eval,
                    train_idx=train_idx_global,
                    test_idx=target_idx_global,
                    alpha=float(spec["alpha"]),
                )
                rows.append({
                    "trait": spec["name"],
                    "method": method_name,
                    "target_island": int(target_code),
                    "target_island_name": str(target_name),
                    "repeat": 0,
                    "repeat_seed": int(seed),
                    "n_train_size": int(k),
                    "n_train": int(k_eff),
                    "n_pcs": int(X_pcs.shape[1]),
                    "n_pcs_requested": int(spec["n_pcs"]),
                    "alpha": float(spec["alpha"]),
                    **metrics,
                })

                if write_selected:
                    for local_i, src_local in enumerate(chosen_local):
                        global_i = int(source_idx_global[int(src_local)])
                        selected_rows.append({
                            "trait": spec["name"],
                            "method": method_name,
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": 0,
                            "repeat_seed": int(seed),
                            "n_train_size": int(k),
                            "rank": int(local_i + 1),
                            "ringnumber": str(ids[global_i]),
                            "source_island": int(locality[global_i]),
                            "source_island_name": island_label(int(locality[global_i]), code_to_label),
                        })

    if not rows:
        logger.warning("Trait '%s': no rows produced; nothing to write", spec["name"])
        return

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    summary = (
        df.groupby(["trait", "method", "n_train_size"], as_index=False)
        .agg(
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_std=("pearson_r", "std"),
            mse_adj_mean=("mse_adj", "mean"),
            mse_adj_std=("mse_adj", "std"),
            n_evaluations=("pearson_r", "size"),
            n_pcs=("n_pcs", "first"),
            alpha=("alpha", "first"),
            fit_time_total_seconds=("fit_time_seconds", "sum"),
        )
        .sort_values(["trait", "method", "n_train_size"])
    )
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), results_path)
    logger.info("Wrote summary to %s", summary_path)

    if write_selected and selected_rows:
        pd.DataFrame(selected_rows).to_csv(selected_path, index=False)
        logger.info("Wrote selected individuals to %s", selected_path)


def run(cfg: dict[str, Any], only_traits: set[str] | None = None) -> None:
    seed = int(cfg.get("seed", 14))
    set_seed(seed)
    output_root = Path(cfg["output"]["root_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    config_used_path = output_root / f"{cfg['output'].get('file_stem', 'e4_similarity_pc_ridge')}_config_used.json"
    with config_used_path.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    for spec in _trait_specs(cfg, only_traits):
        _run_for_trait(spec, cfg, output_root, seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--trait", action="append", default=None, help="Trait to run; can be repeated.")
    args = ap.parse_args()

    with args.config.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    only = set(args.trait) if args.trait else None
    run(cfg, only_traits=only)


if __name__ == "__main__":
    main()
