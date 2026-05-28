#!/usr/bin/env python
"""E5: Island-level Data Shapley with the tuned PC ridge surrogate.

For each trait and target island we compute a TMC-Shapley value per source
island, using PC-ridge accuracy on a fixed-size target calibration set as the
utility. PCA is fit once per trait on the full genotype matrix (genotype-only,
no phenotype leakage) and the PC scores are passed as the feature matrix, so
the library's ``model_type='ridge'`` branch becomes PC ridge with the tuned
(r, alpha) from E1.

Repeats draw a fresh random calibration set (fixed size) from the target
island; everything else (permutations, source pool) is shared.

Reuses ``src.tmc_shapley_islands`` for the heavy lifting.

Usage
-----
    python -m scripts.run_e5_shapley_islands_pc_ridge \
        --config config/final_e5_shapley_islands_pc_ridge_config.json [--trait body_mass]
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.tmc_shapley_islands import (
    ShapleyConfig,
    mirror_add_to_remove_curve,
    run_shapley_experiment,
)
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_island_code(target_island, code_to_label: dict, present_codes: set) -> int:
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
        raise ValueError(f"Island {val} not found. Present: {sorted(present_codes)}")
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
        raise ValueError("No traits selected")
    return specs


def _compute_pcs(X: np.ndarray, n_pcs: int, seed: int, standardize: bool) -> tuple[np.ndarray, float, float]:
    n_components = int(min(n_pcs, X.shape[0], X.shape[1]))
    if standardize:
        mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = X.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std == 0] = 1.0
        X_in = (X - mean) / std
    else:
        X_in = X
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed))
    t0 = time.perf_counter()
    Z = pca.fit_transform(X_in).astype(np.float32, copy=False)
    return Z, float(np.sum(pca.explained_variance_ratio_)), float(time.perf_counter() - t0)


def _fixed_cal_test_split(n_target: int, n_cal: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_target)
    k = max(1, min(int(n_cal), n_target - 1))
    return perm[:k].astype(np.int64), perm[k:].astype(np.int64)


def _run_trait(spec: dict[str, Any], cfg: dict[str, Any], output_root: Path, device: torch.device) -> None:
    global_seed = int(cfg.get("seed", 14))
    tmc_raw = cfg.get("tmc", {})
    n_permutations = int(tmc_raw.get("n_permutations", 30))
    n_cal_fixed = int(cfg.get("cal_n_fixed", 25))
    n_repeats = int(cfg.get("uncertainty", {}).get("n_target_split_repeats", 10))
    n_random_orders = int(cfg.get("baselines", {}).get("n_random_orders", 0))

    trait_root = output_root / spec["name"]
    trait_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Trait '%s': n_pcs=%d alpha=%.4g permutations=%d cal_n=%d repeats=%d random_orders=%d",
        spec["name"], spec["n_pcs"], spec["alpha"], n_permutations, n_cal_fixed, n_repeats, n_random_orders,
    )

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

    Z, cumvar, pca_time = _compute_pcs(
        X=X, n_pcs=spec["n_pcs"], seed=global_seed, standardize=spec["standardize_features"],
    )
    logger.info("Trait '%s': PCA r=%d in %.1fs (cumvar=%.4f)", spec["name"], Z.shape[1], pca_time, cumvar)

    raw_targets = cfg.get("target_islands", sorted(included_codes))
    target_codes = [resolve_island_code(t, code_to_label, present_codes) for t in raw_targets]

    base_cfg = ShapleyConfig(
        n_permutations=n_permutations,
        use_truncation=bool(tmc_raw.get("use_truncation", False)),
        eps_trunc=float(tmc_raw.get("eps_trunc", 0.005)),
        min_prefix_islands=int(tmc_raw.get("min_prefix_islands", 3)),
        cap_per_island=tmc_raw.get("cap_per_island", None),
        n_random_orders=n_random_orders,
        model_type="ridge",
        ridge_alpha=float(spec["alpha"]),
        seed=global_seed,
    )

    for target_code in target_codes:
        target_name = island_label(int(target_code), code_to_label)
        source_codes = [c for c in included_codes if c != target_code]
        if not source_codes:
            continue
        source_names = {c: island_label(int(c), code_to_label) for c in source_codes}

        n_target = int(np.sum(locality == int(target_code)))
        if n_target <= n_cal_fixed:
            logger.warning(
                "Target %s has n=%d <= cal_n=%d; skipping", target_code, n_target, n_cal_fixed
            )
            continue

        island_dir = trait_root / f"island_{target_code}"
        phi_rows: list[dict[str, Any]] = []
        add_rows: list[pd.DataFrame] = []
        completed: list[tuple[int, int]] = []

        for repeat_idx in range(n_repeats):
            repeat_seed = global_seed + 1_000 * repeat_idx
            cal_idx, test_idx = _fixed_cal_test_split(n_target, n_cal_fixed, repeat_seed)
            rep_cfg = copy.deepcopy(base_cfg)
            rep_cfg.seed = repeat_seed
            rep_out = str(island_dir / f"repeat_{repeat_idx}")

            logger.info(
                "Trait=%s target=%s (%s) repeat=%d/%d seed=%d cal=%d test=%d",
                spec["name"], target_code, target_name, repeat_idx + 1, n_repeats,
                repeat_seed, len(cal_idx), len(test_idx),
            )

            try:
                result = run_shapley_experiment(
                    X=Z, y=y, ids=ids, locality=locality, y_eval=y_eval,
                    target_island_code=int(target_code), target_island_name=target_name,
                    source_codes=source_codes, source_names=source_names,
                    train_params=None, cfg=rep_cfg, output_dir=rep_out, device=device,
                    cal_idx=cal_idx, test_idx=test_idx,
                    compute_random_individual=(n_random_orders > 0),
                )
            except Exception as exc:
                logger.exception("repeat %d failed for target %s: %s", repeat_idx, target_code, exc)
                continue

            completed.append((repeat_idx, repeat_seed))
            phi = result["phi"]
            n_ind = result["n_individuals"]
            for i, code in enumerate(source_codes):
                n_i = int(n_ind[i])
                phi_rows.append({
                    "trait": spec["name"],
                    "target_island": int(target_code),
                    "target_island_name": target_name,
                    "repeat": repeat_idx,
                    "split_seed": repeat_seed,
                    "source_island": int(code),
                    "source_island_name": source_names.get(code, str(code)),
                    "n_island": n_i,
                    "phi": float(phi[i]),
                    "phi_per_individual": float(phi[i] / max(n_i, 1)),
                    "v_full": float(result["v_full"]),
                })
            add_df = result["add_curve_df"].copy()
            add_df["trait"] = spec["name"]
            add_df["target_island"] = int(target_code)
            add_df["repeat"] = repeat_idx
            add_df["split_seed"] = repeat_seed
            add_rows.append(add_df)

        if not completed:
            logger.warning("All repeats failed for target %s", target_code)
            continue

        island_dir.mkdir(parents=True, exist_ok=True)
        phi_repeats = pd.DataFrame(phi_rows)
        phi_repeats.to_csv(island_dir / f"shapley_repeats_target_{target_code}.csv", index=False)

        phi_summary = (
            phi_repeats
            .groupby(["trait", "target_island", "target_island_name", "source_island",
                      "source_island_name", "n_island"], as_index=False)
            .agg(
                phi_mean=("phi", "mean"),
                phi_std=("phi", "std"),
                phi_per_ind_mean=("phi_per_individual", "mean"),
                phi_per_ind_std=("phi_per_individual", "std"),
                n_repeats=("phi", "size"),
            )
        )
        phi_summary["rank_mean_phi_per_ind"] = (
            phi_summary["phi_per_ind_mean"].rank(ascending=False, method="min").astype(int)
        )
        phi_summary.to_csv(island_dir / f"shapley_uncertainty_target_{target_code}.csv", index=False)

        add_repeats = pd.concat(add_rows, ignore_index=True)
        remove_repeats = mirror_add_to_remove_curve(add_repeats, n_source_islands=len(source_codes))
        add_repeats.to_csv(island_dir / f"add_curve_repeats_target_{target_code}.csv", index=False)
        remove_repeats.to_csv(island_dir / f"remove_curve_repeats_target_{target_code}.csv", index=False)

        logger.info("Target %s (%s) done: %d/%d repeats", target_code, target_name, len(completed), n_repeats)

    # Trait-level concat of per-island Shapley summaries for easy notebook loading.
    summary_frames = [
        pd.read_csv(p) for p in sorted(trait_root.glob("island_*/shapley_uncertainty_target_*.csv"))
    ]
    if summary_frames:
        pd.concat(summary_frames, ignore_index=True).to_csv(
            trait_root / "shapley_island_summary_all_targets.csv", index=False
        )


def run(cfg: dict[str, Any], only_traits: set[str] | None = None) -> None:
    set_seed(int(cfg.get("seed", 14)))
    device_str = cfg.get("device", "cpu")
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else
                          (device_str if device_str != "auto" else "cpu"))

    output_root = Path(cfg["output"]["root_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / f"{cfg['output'].get('file_stem', 'e5_shapley_islands_pc_ridge')}_config_used.json").open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    for spec in _trait_specs(cfg, only_traits):
        _run_trait(spec, cfg, output_root, device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--trait", action="append", default=None, help="Trait to run; repeatable.")
    args = ap.parse_args()
    with args.config.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    only = set(args.trait) if args.trait else None
    run(cfg, only_traits=only)


if __name__ == "__main__":
    main()
