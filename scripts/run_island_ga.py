#!/usr/bin/env python
"""
Island-level training-set optimisation via Genetic Algorithm.

Given a target island, select the best subset of *source islands* (of size
``n_train_islands``) that maximises

    fitness  =  corr_cal  /  n_individuals

where
  * ``corr_cal`` is the Pearson correlation on a calibration subset of the
    target island after fitting a ridge regression on the selected islands,
  * ``n_individuals`` is the total number of training individuals across the
    selected islands.

This mirrors the Shapley island experiment (same cal/test split, same
evaluation metrics) but replaces TMC-Shapley with a GA search.

Outputs
-------
``island_ga_results.csv`` – one row per (target, cal_setting, repeat,
n_train_islands, method) with columns:
    n_train_islands, n_individuals, corr_eval_cal, corr_eval_test,
    mse_adj_test, fitness, method, selected_islands, target_island,
    target_island_name, n_cal_samples, repeat, repeat_seed, trait

Usage
-----
    python -m scripts.run_island_ga --config config/island_ga_config.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.removal_curve import split_target_island
from src.training_set_optimization.ga_subset import GAConfig, run_ga
from src.utils import _pearson_corr, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_island_code(target_island, code_to_label: dict, present_codes: set) -> int:
    """Resolve a target island specification to its internal integer code."""
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
        raise ValueError(
            f"Island {val} not found as code or label. "
            f"Present codes: {sorted(present_codes)}"
        )

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


def _as_int_list(v: Any, default: List[int]) -> List[int]:
    if v is None:
        return default
    if isinstance(v, list):
        return [int(x) for x in v]
    return [int(v)]


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    """Append a DataFrame to a CSV, creating it (with header) if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def _make_repeat_seed(
    global_seed: int,
    target_code: int,
    cal_val: float,
    repeat_idx: int,
) -> int:
    """Deterministic seed for a (target, cal_val, repeat) combination.

    Uses the same scheme as ``run_tmc_shapley_sweep.py`` so that cal/test
    splits are identical when the global seed and cal setting match.
    """
    token = (
        f"{int(global_seed)}|{int(target_code)}"
        f"|{float(cal_val):.12g}|{int(repeat_idx)}"
    )
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % 2_147_483_647


def _build_trait_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build per-trait specs from config (matches Shapley sweep convention)."""
    traits_cfg = cfg.get("traits", None)
    if traits_cfg is None:
        paths = dict(cfg["paths"])
        return [{
            "name": "default",
            "paths": paths,
            "target_column": cfg.get("target_column", "y_adjusted"),
            "eval_target_column": cfg.get("eval_target_column", "y_mean"),
            "standardize_features": cfg.get("standardize_features", True),
            "min_count": cfg.get("min_count", 20),
        }]

    specs: List[Dict[str, Any]] = []
    for t in traits_cfg:
        if not isinstance(t, dict):
            raise ValueError("traits must be a list of objects")
        if "name" not in t or "npz" not in t:
            raise ValueError("each trait must define 'name' and 'npz'")
        paths = dict(cfg["paths"])
        paths["npz"] = t["npz"]
        specs.append({
            "name": str(t["name"]),
            "paths": paths,
            "target_column": t.get(
                "target_column", cfg.get("target_column", "y_adjusted")
            ),
            "eval_target_column": t.get(
                "eval_target_column",
                cfg.get("eval_target_column", "y_mean"),
            ),
            "standardize_features": t.get(
                "standardize_features",
                cfg.get("standardize_features", True),
            ),
            "min_count": int(
                t.get("min_count", cfg.get("min_count", 20))
            ),
        })
    return specs


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _evaluate_island_subset_on_cal(
    island_indices: np.ndarray,
    source_codes: List[int],
    X_source: np.ndarray,
    y_source: np.ndarray,
    locality_source: np.ndarray,
    X_cal: np.ndarray,
    y_eval_cal: np.ndarray,
    ridge_alpha: float,
    snp_cols: Optional[np.ndarray] = None,
) -> Tuple[float, int]:
    """
    Fit ridge on the selected source islands and predict on the cal set.

    Returns
    -------
    corr_cal : float
        Pearson correlation on the calibration set (using y_eval).
    n_individuals : int
        Total number of training individuals in the selected islands.
    """
    # Gather training data from selected islands
    x_parts, y_parts = [], []
    for idx in island_indices:
        code = source_codes[idx]
        mask = locality_source == code
        x_parts.append(X_source[mask])
        y_parts.append(y_source[mask])

    if not x_parts:
        return 0.0, 0

    X_train = np.concatenate(x_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    n_individuals = len(X_train)

    if n_individuals < 2:
        return 0.0, n_individuals

    # SNP selection
    if snp_cols is not None:
        X_train_sel = X_train[:, snp_cols]
        X_cal_sel = X_cal[:, snp_cols]
    else:
        X_train_sel = X_train
        X_cal_sel = X_cal

    # Fit ridge
    model = Ridge(alpha=max(float(ridge_alpha), 1e-12))
    model.fit(X_train_sel, y_train)
    pred = model.predict(X_cal_sel)

    corr_cal = float(_pearson_corr(pred, y_eval_cal))
    if not np.isfinite(corr_cal):
        corr_cal = 0.0

    return corr_cal, n_individuals


def _evaluate_island_subset_on_test(
    island_indices: np.ndarray,
    source_codes: List[int],
    X_source: np.ndarray,
    y_source: np.ndarray,
    locality_source: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    ridge_alpha: float,
    snp_cols: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Fit ridge on selected source islands, evaluate on the test set.

    Returns dict with corr_eval_test, mse_adj_test, n_individuals.
    """
    x_parts, y_parts = [], []
    for idx in island_indices:
        code = source_codes[idx]
        mask = locality_source == code
        x_parts.append(X_source[mask])
        y_parts.append(y_source[mask])

    if not x_parts:
        return {"corr_eval_test": 0.0, "mse_adj_test": float("inf"), "n_individuals": 0}

    X_train = np.concatenate(x_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    n_individuals = len(X_train)

    if n_individuals < 2:
        return {"corr_eval_test": 0.0, "mse_adj_test": float("inf"), "n_individuals": n_individuals}

    if snp_cols is not None:
        X_train_sel = X_train[:, snp_cols]
        X_test_sel = X_test[:, snp_cols]
    else:
        X_train_sel = X_train
        X_test_sel = X_test

    model = Ridge(alpha=max(float(ridge_alpha), 1e-12))
    model.fit(X_train_sel, y_train)
    pred = model.predict(X_test_sel)

    corr_eval = float(_pearson_corr(pred, y_eval_test))
    if not np.isfinite(corr_eval):
        corr_eval = 0.0
    mse_adj = float(np.mean((pred - y_test) ** 2))

    return {
        "corr_eval_test": corr_eval,
        "mse_adj_test": mse_adj,
        "n_individuals": n_individuals,
    }


# ---------------------------------------------------------------------------
# GA runner (island level)
# ---------------------------------------------------------------------------

def run_island_ga(
    n_train_islands: int,
    source_codes: List[int],
    X_source: np.ndarray,
    y_source: np.ndarray,
    locality_source: np.ndarray,
    X_cal: np.ndarray,
    y_eval_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    ridge_alpha: float,
    ga_cfg: GAConfig,
    snp_cols: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run the GA to find the best subset of ``n_train_islands`` source islands
    maximising ``corr_cal / n_individuals``.

    Returns a dict with:
      - best_islands: selected island codes
      - best_island_indices: indices into source_codes
      - corr_eval_cal, corr_eval_test, mse_adj_test
      - n_individuals, fitness, ga_stats
    """
    n_source = len(source_codes)

    # Precompute island sizes for the fitness function
    island_sizes = {}
    for idx, code in enumerate(source_codes):
        island_sizes[idx] = int(np.sum(locality_source == code))

    def fitness_fn(island_idx_subset: np.ndarray) -> float:
        """Minimise negative (corr_cal / n_individuals)."""
        corr_cal, n_ind = _evaluate_island_subset_on_cal(
            island_indices=island_idx_subset,
            source_codes=source_codes,
            X_source=X_source,
            y_source=y_source,
            locality_source=locality_source,
            X_cal=X_cal,
            y_eval_cal=y_eval_cal,
            ridge_alpha=ridge_alpha,
            snp_cols=snp_cols,
        )
        if n_ind == 0:
            return float("inf")
        # GA minimises, so negate what we want to maximise
        return -(corr_cal / n_ind)

    best_subset, best_fitness, ga_stats = run_ga(
        n_candidates=n_source,
        n_train=n_train_islands,
        fitness_fn=fitness_fn,
        cfg=ga_cfg,
    )

    # Evaluate the best subset on the test set
    test_result = _evaluate_island_subset_on_test(
        island_indices=best_subset,
        source_codes=source_codes,
        X_source=X_source,
        y_source=y_source,
        locality_source=locality_source,
        X_test=X_test,
        y_test=y_test,
        y_eval_test=y_eval_test,
        ridge_alpha=ridge_alpha,
        snp_cols=snp_cols,
    )

    # Also get cal correlation for reporting
    corr_cal, n_ind = _evaluate_island_subset_on_cal(
        island_indices=best_subset,
        source_codes=source_codes,
        X_source=X_source,
        y_source=y_source,
        locality_source=locality_source,
        X_cal=X_cal,
        y_eval_cal=y_eval_cal,
        ridge_alpha=ridge_alpha,
        snp_cols=snp_cols,
    )

    best_island_codes = [source_codes[i] for i in best_subset]

    return {
        "best_islands": best_island_codes,
        "best_island_indices": best_subset.tolist(),
        "corr_eval_cal": corr_cal,
        "corr_eval_test": test_result["corr_eval_test"],
        "mse_adj_test": test_result["mse_adj_test"],
        "n_individuals": test_result["n_individuals"],
        "fitness": -best_fitness,  # flip sign back: positive = better
        "ga_stats": ga_stats,
    }


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def _random_island_baseline(
    n_train_islands: int,
    source_codes: List[int],
    X_source: np.ndarray,
    y_source: np.ndarray,
    locality_source: np.ndarray,
    X_cal: np.ndarray,
    y_eval_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    ridge_alpha: float,
    n_random: int,
    rng: np.random.Generator,
    snp_cols: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate ``n_random`` random subsets of ``n_train_islands`` source islands.
    """
    n_source = len(source_codes)
    results = []

    for i in range(n_random):
        subset = np.sort(rng.choice(n_source, size=n_train_islands, replace=False))

        corr_cal, n_ind_cal = _evaluate_island_subset_on_cal(
            island_indices=subset,
            source_codes=source_codes,
            X_source=X_source,
            y_source=y_source,
            locality_source=locality_source,
            X_cal=X_cal,
            y_eval_cal=y_eval_cal,
            ridge_alpha=ridge_alpha,
            snp_cols=snp_cols,
        )

        test_result = _evaluate_island_subset_on_test(
            island_indices=subset,
            source_codes=source_codes,
            X_source=X_source,
            y_source=y_source,
            locality_source=locality_source,
            X_test=X_test,
            y_test=y_test,
            y_eval_test=y_eval_test,
            ridge_alpha=ridge_alpha,
            snp_cols=snp_cols,
        )

        n_ind = test_result["n_individuals"]
        fitness = (corr_cal / n_ind) if n_ind > 0 else 0.0

        selected_codes = [source_codes[j] for j in subset]

        results.append({
            "selected_islands": selected_codes,
            "corr_eval_cal": corr_cal,
            "corr_eval_test": test_result["corr_eval_test"],
            "mse_adj_test": test_result["mse_adj_test"],
            "n_individuals": n_ind,
            "fitness": fitness,
            "random_seed": int(rng.integers(0, 2**31)),
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Island-level training-set optimisation via GA"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to island_ga_config.json",
    )
    parser.add_argument(
        "--target_islands",
        nargs="+",
        default=None,
        help="Override target_islands from config",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    # ---- Parse config --------------------------------------------------------
    ga_section = cfg.get("island_ga", {})

    output_dir = Path(
        ga_section.get("output_dir", cfg["paths"].get("output_dir", "outputs/island_ga"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    global_seed = int(cfg.get("seed", 42))
    set_seed(global_seed)

    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "ridge")).lower()
    if model_type != "ridge":
        raise ValueError("island_ga currently only supports model.type='ridge'")
    ridge_alpha = float(model_cfg.get("alpha", 1e5))

    # n_train_islands to sweep over
    n_train_islands_list = _as_int_list(
        ga_section.get("n_train_islands", None), default=[]
    )

    # Calibration settings
    n_cal_samples_list = _as_int_list(
        ga_section.get("n_cal_samples", cfg.get("n_cal_samples", 40)),
        default=[40],
    )
    n_repeats = int(ga_section.get("n_repeats", 5))

    # GA parameters
    ga_pop_size = int(ga_section.get("pop_size", 30))
    ga_n_generations = int(ga_section.get("n_generations", 50))
    ga_n_elite = int(ga_section.get("n_elite", 2))
    ga_tournament_k = int(ga_section.get("tournament_k", 3))
    ga_crossover_prob = float(ga_section.get("crossover_prob", 0.9))
    ga_mutation_prob = float(ga_section.get("mutation_prob", 0.3))
    ga_n_swaps = int(ga_section.get("n_swaps_per_mut", 1))
    ga_stagnation = int(ga_section.get("stagnation_limit", 15))
    ga_verbose = bool(ga_section.get("verbose", True))

    # Baselines
    n_random_orders = int(
        cfg.get("baselines", {}).get("n_random_orders", 5)
    )

    # SNP selection
    use_snp_selection = bool(cfg.get("use_snp_selection", False))
    num_snps = cfg.get("num_snps", None)
    snp_mode = str(cfg.get("snp_selection_mode", "random")).lower()

    trait_specs = _build_trait_specs(cfg)

    # ---- Per-trait loop ------------------------------------------------------
    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        trait_output = output_dir / trait_name if len(trait_specs) > 1 else output_dir
        trait_output.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Loading data for trait '%s' from %s",
            trait_name,
            trait_spec["paths"].get("npz", "<unknown>"),
        )
        X, y, ids, GRM_df, locality, code_to_label, y_eval = load_data(
            paths=trait_spec["paths"],
            target_column=trait_spec["target_column"],
            standardize_features=trait_spec["standardize_features"],
            return_locality=True,
            min_count=trait_spec["min_count"],
            return_eval=True,
            eval_target_column=trait_spec["eval_target_column"],
        )
        logger.info(
            "Trait '%s' loaded: %d samples, %d SNPs, %d islands",
            trait_name, X.shape[0], X.shape[1], len(np.unique(locality)),
        )

        present_codes = set(int(c) for c in np.unique(locality))
        included_raw = cfg.get("included_islands", None)
        if included_raw is not None:
            included_island_codes = [
                resolve_island_code(v, code_to_label, present_codes)
                for v in included_raw
            ]
        else:
            included_island_codes = sorted(present_codes)

        raw_targets = (
            args.target_islands
            if args.target_islands
            else cfg.get("target_islands", [0])
        )
        target_codes = [
            resolve_island_code(t, code_to_label, present_codes)
            for t in raw_targets
        ]

        # Fixed SNP subset for this trait
        fixed_snp_cols: Optional[np.ndarray] = None
        if (
            use_snp_selection
            and num_snps is not None
            and int(num_snps) < X.shape[1]
        ):
            rng_snp = np.random.default_rng(global_seed)
            fixed_snp_cols = np.sort(
                rng_snp.choice(X.shape[1], size=int(num_snps), replace=False)
                .astype(np.int64)
            )
            snp_path = trait_output / "fixed_snp_columns.npy"
            np.save(snp_path, fixed_snp_cols)
            logger.info(
                "Fixed SNP subset for trait '%s': n=%d", trait_name, len(fixed_snp_cols)
            )

        results_path = trait_output / "island_ga_results.csv"
        if results_path.exists():
            results_path.unlink()

        # ---- Per-target loop -------------------------------------------------
        for target_code in target_codes:
            target_name = island_label(target_code, code_to_label)
            source_codes = [c for c in included_island_codes if c != target_code]
            n_source = len(source_codes)
            source_names = {c: island_label(c, code_to_label) for c in source_codes}

            if n_source == 0:
                logger.warning("No source islands for target %s, skipping", target_code)
                continue

            # Determine n_train_islands sweep
            if not n_train_islands_list:
                # Default: 1 to n_source
                sweep_sizes = list(range(1, n_source + 1))
            else:
                sweep_sizes = [k for k in n_train_islands_list if k <= n_source]

            logger.info(
                "Target %s (%s): %d source islands %s, "
                "n_train_islands sweep: %s, n_cal_samples: %s, repeats: %d",
                target_code,
                target_name,
                n_source,
                [f"{c}({source_names[c]})" for c in source_codes],
                sweep_sizes,
                n_cal_samples_list,
                n_repeats,
            )

            # Log island sizes
            for code in source_codes:
                n_k = int(np.sum(locality == code))
                logger.info(
                    "  Source island %d (%s): %d individuals",
                    code, source_names[code], n_k,
                )
            n_target_total = int(np.sum(locality == target_code))
            logger.info(
                "  Target island %d (%s): %d individuals",
                target_code, target_name, n_target_total,
            )

            # ---- Per cal_setting × repeat loop --------------------------------
            for n_cal in n_cal_samples_list:
                if n_cal >= n_target_total:
                    logger.warning(
                        "n_cal_samples=%d >= n_target=%d for target %d, skipping",
                        n_cal, n_target_total, target_code,
                    )
                    continue

                cal_frac = float(n_cal / n_target_total)

                for repeat_idx in range(n_repeats):
                    repeat_seed = _make_repeat_seed(
                        global_seed=global_seed,
                        target_code=int(target_code),
                        cal_val=float(n_cal),
                        repeat_idx=int(repeat_idx),
                    )

                    # Split target into cal + test (same as Shapley sweep)
                    rng_split = np.random.default_rng(repeat_seed)
                    target_mask = locality == target_code
                    n_target = int(target_mask.sum())
                    perm = rng_split.permutation(n_target)
                    cal_idx = perm[:n_cal]
                    test_idx = perm[n_cal:]

                    split = split_target_island(
                        X, y, ids, locality, target_code,
                        y_eval=y_eval,
                        cal_fraction=cal_frac,
                        seed=int(repeat_seed),
                        cal_idx=cal_idx,
                        test_idx=test_idx,
                    )

                    X_source_split = split["X_source"]
                    y_source_split = split["y_source"]
                    locality_source = split["locality_source"]

                    # Filter to included source codes
                    in_source = np.isin(locality_source, source_codes)
                    X_source_filt = X_source_split[in_source]
                    y_source_filt = y_source_split[in_source]
                    locality_source_filt = locality_source[in_source]

                    X_cal = split["X_cal"]
                    y_eval_cal = split["y_eval_cal"]

                    X_test = split["X_test"]
                    y_test = split["y_test"]
                    y_eval_test = split["y_eval_test"]

                    logger.info(
                        "Repeat %d/%d (seed=%d): n_source=%d, n_cal=%d, n_test=%d",
                        repeat_idx + 1, n_repeats, repeat_seed,
                        len(X_source_filt), len(X_cal), len(X_test),
                    )

                    # ---- Per n_train_islands ---------------------------------
                    for n_train_isl in sweep_sizes:
                        logger.info(
                            "  n_train_islands=%d/%d ...",
                            n_train_isl, n_source,
                        )

                        # --- GA optimisation ---
                        ga_seed = int(
                            (repeat_seed + n_train_isl * 7919) % 2_147_483_647
                        )
                        ga_cfg = GAConfig(
                            pop_size=ga_pop_size,
                            n_generations=ga_n_generations,
                            n_elite=ga_n_elite,
                            tournament_k=min(ga_tournament_k, ga_pop_size),
                            crossover_prob=ga_crossover_prob,
                            mutation_prob=ga_mutation_prob,
                            n_swaps_per_mut=min(ga_n_swaps, max(1, n_train_isl - 1)),
                            seed=ga_seed,
                            verbose=ga_verbose,
                            stagnation_limit=ga_stagnation,
                        )

                        t0 = time.perf_counter()
                        ga_result = run_island_ga(
                            n_train_islands=n_train_isl,
                            source_codes=source_codes,
                            X_source=X_source_filt,
                            y_source=y_source_filt,
                            locality_source=locality_source_filt,
                            X_cal=X_cal,
                            y_eval_cal=y_eval_cal,
                            X_test=X_test,
                            y_test=y_test,
                            y_eval_test=y_eval_test,
                            ridge_alpha=ridge_alpha,
                            ga_cfg=ga_cfg,
                            snp_cols=fixed_snp_cols,
                        )
                        elapsed = time.perf_counter() - t0

                        selected_names = [
                            source_names.get(c, str(c))
                            for c in ga_result["best_islands"]
                        ]
                        logger.info(
                            "  GA done (%.1fs): islands=%s (%s), "
                            "n_ind=%d, corr_cal=%.4f, corr_test=%.4f, "
                            "fitness=%.6f",
                            elapsed,
                            ga_result["best_islands"],
                            selected_names,
                            ga_result["n_individuals"],
                            ga_result["corr_eval_cal"],
                            ga_result["corr_eval_test"],
                            ga_result["fitness"],
                        )

                        ga_row = pd.DataFrame([{
                            "n_train_islands": n_train_isl,
                            "n_individuals": ga_result["n_individuals"],
                            "corr_eval_cal": ga_result["corr_eval_cal"],
                            "corr_eval_test": ga_result["corr_eval_test"],
                            "mse_adj_test": ga_result["mse_adj_test"],
                            "fitness": ga_result["fitness"],
                            "method": "island_ga",
                            "selected_islands": str(ga_result["best_islands"]),
                            "selected_island_names": str(selected_names),
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "n_cal_samples": int(n_cal),
                            "repeat": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "trait": trait_name,
                            "ga_generations": ga_result["ga_stats"]["generations_run"],
                            "ga_cache_size": ga_result["ga_stats"]["cache_size"],
                            "ga_elapsed_sec": ga_result["ga_stats"]["elapsed_sec"],
                        }])
                        _append_csv(ga_row, results_path)

                        # --- Random baselines ---
                        rng_rand = np.random.default_rng(ga_seed + 1)
                        rand_results = _random_island_baseline(
                            n_train_islands=n_train_isl,
                            source_codes=source_codes,
                            X_source=X_source_filt,
                            y_source=y_source_filt,
                            locality_source=locality_source_filt,
                            X_cal=X_cal,
                            y_eval_cal=y_eval_cal,
                            X_test=X_test,
                            y_test=y_test,
                            y_eval_test=y_eval_test,
                            ridge_alpha=ridge_alpha,
                            n_random=n_random_orders,
                            rng=rng_rand,
                            snp_cols=fixed_snp_cols,
                        )

                        for ri, rr in enumerate(rand_results):
                            rand_names = [
                                source_names.get(c, str(c))
                                for c in rr["selected_islands"]
                            ]
                            rand_row = pd.DataFrame([{
                                "n_train_islands": n_train_isl,
                                "n_individuals": rr["n_individuals"],
                                "corr_eval_cal": rr["corr_eval_cal"],
                                "corr_eval_test": rr["corr_eval_test"],
                                "mse_adj_test": rr["mse_adj_test"],
                                "fitness": rr["fitness"],
                                "method": "random",
                                "selected_islands": str(rr["selected_islands"]),
                                "selected_island_names": str(rand_names),
                                "target_island": int(target_code),
                                "target_island_name": str(target_name),
                                "n_cal_samples": int(n_cal),
                                "repeat": int(repeat_idx),
                                "repeat_seed": int(repeat_seed),
                                "trait": trait_name,
                                "ga_generations": 0,
                                "ga_cache_size": 0,
                                "ga_elapsed_sec": 0.0,
                            }])
                            _append_csv(rand_row, results_path)

                        logger.info(
                            "  Random baselines (%d): corr_test mean=%.4f",
                            n_random_orders,
                            np.mean([r["corr_eval_test"] for r in rand_results]),
                        )

        logger.info(
            "Trait '%s' complete. Results: %s",
            trait_name,
            results_path,
        )

    logger.info("All traits done.")


if __name__ == "__main__":
    main()
