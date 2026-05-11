#!/usr/bin/env python
"""
Run individual-level TMC-Shapley for source individuals.

The target island is split into calibration and test sets. TMC-Shapley values
are computed with utility measured on the target calibration set. Optional
add/remove curves are then evaluated on the held-out target test set.

This runner is intentionally conservative for the first individual-wise runs:
use ridge, fixed alpha, fixed reduced SNP set, source-individual caps, and
truncation.

Usage
-----
    python -m scripts.run_tmc_shapley_individuals --config config/tmc_shapley_individuals.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Limit accidental oversubscription on CPU Ridge jobs unless the environment
# already provides a stronger setting.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.removal_curve import split_target_island
from src.tmc_shapley_islands import ShapleyConfig, _evaluate_individual_subset
from src.utils import _select_top_snps_by_abs_corr, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_island_code(target_island: Any, code_to_label: dict, present_codes: set) -> int:
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
        raise ValueError(f"Island {val} not found as code or label. Present codes: {sorted(present_codes)}")

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


class IndividualUtilityCache:
    """Small optional in-memory cache for repeated individual subsets."""

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._cache: Dict[Tuple[int, ...], Dict[str, float]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, indices: np.ndarray) -> Optional[Dict[str, float]]:
        if not self.enabled:
            self.misses += 1
            return None
        key = tuple(int(i) for i in np.sort(indices))
        value = self._cache.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def set(self, indices: np.ndarray, value: Dict[str, float]) -> None:
        if not self.enabled:
            return
        key = tuple(int(i) for i in np.sort(indices))
        self._cache[key] = value

    @property
    def total_evals(self) -> int:
        return self.misses

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def _device_from_config(cfg: Dict[str, Any]) -> torch.device:
    device_str = cfg.get("device", "cpu")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _build_shapley_config(cfg: Dict[str, Any], seed: int) -> ShapleyConfig:
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "ridge")).lower()
    if model_type != "ridge":
        raise ValueError("Individual-wise Shapley runner is configured for model.type='ridge'.")

    tmc_cfg = cfg.get("tmc", {})
    min_prefix = tmc_cfg.get("min_prefix_individuals", tmc_cfg.get("min_prefix_islands", 10))

    return ShapleyConfig(
        n_permutations=int(tmc_cfg.get("n_permutations", 10)),
        eps_trunc=float(tmc_cfg.get("eps_trunc", 0.01)),
        min_prefix_islands=int(min_prefix),
        use_truncation=bool(tmc_cfg.get("use_truncation", True)),
        island_sampling_mode="full",
        cap_per_island=None,
        n_train_seeds_per_eval=1,
        cache_dir=tmc_cfg.get("cache_dir", None),
        n_random_orders=int(cfg.get("evaluation_curve", {}).get("n_random_orders", 1)),
        use_early_stopping=False,
        early_stopping_patience=0,
        retrain_use_early_stopping=False,
        retrain_early_stopping_patience=0,
        snp_selection_mode=cfg.get("snp_selection_mode", "random"),
        use_snp_selection=bool(cfg.get("use_snp_selection", True)),
        num_snps=cfg.get("num_snps", None),
        model_type="ridge",
        ridge_alpha=float(model_cfg.get("alpha", 1e5)),
        seed=int(seed),
    )


def _sample_source_individuals(
    locality_source: np.ndarray,
    source_codes: List[int],
    sampling_cfg: Dict[str, Any],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    max_per_island = sampling_cfg.get("max_per_island", None)
    max_total = sampling_cfg.get("max_source_individuals", None)

    parts: List[np.ndarray] = []
    for code in source_codes:
        idx = np.flatnonzero(locality_source == code)
        if max_per_island is not None and len(idx) > int(max_per_island):
            idx = rng.choice(idx, size=int(max_per_island), replace=False)
        parts.append(np.asarray(idx, dtype=np.int64))

    if not parts:
        return np.empty(0, dtype=np.int64)

    selected = np.concatenate(parts)
    if max_total is not None and len(selected) > int(max_total):
        selected = rng.choice(selected, size=int(max_total), replace=False)

    selected = np.asarray(np.sort(selected), dtype=np.int64)
    if len(selected) < 2:
        raise ValueError("Need at least two source individuals after individual_sampling filters.")
    return selected


def _select_snp_cols(
    X_source: np.ndarray,
    y_source: np.ndarray,
    cfg: Dict[str, Any],
    seed: int,
) -> Optional[np.ndarray]:
    use_snp_selection = bool(cfg.get("use_snp_selection", True))
    num_snps = cfg.get("num_snps", None)
    if not use_snp_selection or num_snps is None:
        logger.warning("SNP selection is disabled; individual-wise ridge may be very slow.")
        return None

    k = int(num_snps)
    p = int(X_source.shape[1])
    if k <= 0:
        raise ValueError("num_snps must be positive when use_snp_selection=true")
    if k >= p:
        logger.info("Requested num_snps=%d >= available SNPs=%d; using all SNPs", k, p)
        return np.arange(p, dtype=np.int64)

    mode = str(cfg.get("snp_selection_mode", "random")).lower()
    if mode == "random":
        rng = np.random.default_rng(seed)
        snp_cols = np.sort(rng.choice(p, size=k, replace=False).astype(np.int64))
        logger.info("SNP selection: %d random SNPs from %d (seed=%d)", k, p, seed)
        return snp_cols
    if mode in {"top_corr", "corr", "correlation"}:
        snp_cols = _select_top_snps_by_abs_corr(X_source, y_source, k).astype(np.int64)
        logger.info("SNP selection: top %d SNPs by absolute source correlation", k)
        return snp_cols
    raise ValueError("snp_selection_mode must be 'random' or 'top_corr'")


def _state_path(output_dir: Path, target_code: int, seed: int, cfg: Dict[str, Any]) -> Optional[Path]:
    tmc_cfg = cfg.get("tmc", {})
    if not bool(tmc_cfg.get("resume", True)):
        return None
    explicit = tmc_cfg.get("permutation_state_path", None)
    if explicit:
        return Path(explicit)
    return output_dir / f"tmc_individual_state_target_{target_code}_seed_{seed}.npz"


def _hash_values(values: Any) -> str:
    arr = np.asarray(values)
    h = hashlib.blake2b(digest_size=16)
    if arr.dtype.kind in {"O", "U", "S"}:
        h.update("\n".join(arr.astype(str).tolist()).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _make_state_token(
    ids_source: np.ndarray,
    locality_source: np.ndarray,
    snp_cols: Optional[np.ndarray],
    cfg: ShapleyConfig,
) -> str:
    payload = {
        "source_ids_hash": _hash_values(ids_source.astype(str)),
        "source_islands_hash": _hash_values(locality_source.astype(np.int64)),
        "snp_cols_hash": _hash_values(snp_cols.astype(np.int64)) if snp_cols is not None else "all",
        "model_type": cfg.model_type,
        "ridge_alpha": float(cfg.ridge_alpha),
        "use_truncation": bool(cfg.use_truncation),
        "eps_trunc": float(cfg.eps_trunc),
        "min_prefix_individuals": int(cfg.min_prefix_islands),
    }
    return hashlib.blake2b(
        json.dumps(payload, sort_keys=True).encode("utf-8"),
        digest_size=16,
    ).hexdigest()


def _load_state(
    state_path: Optional[Path],
    n_permutations: int,
    n_players: int,
    seed: int,
    state_token: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi_by_perm = np.full((n_permutations, n_players), np.nan, dtype=np.float64)
    completed = np.zeros(n_permutations, dtype=bool)
    truncated = np.zeros(n_permutations, dtype=bool)

    if state_path is None or not state_path.exists():
        return phi_by_perm, completed, truncated

    try:
        data = np.load(state_path, allow_pickle=False)
        prev_phi = data["phi_by_perm"]
        prev_completed = data["completed"].astype(bool)
        prev_truncated = data["truncated"].astype(bool)
        prev_seed = int(np.asarray(data["seed"]).ravel()[0])
        prev_players = int(np.asarray(data["n_players"]).ravel()[0])
        prev_token = str(np.asarray(data["state_token"]).ravel()[0]) if "state_token" in data else None
        if prev_seed != int(seed) or prev_players != int(n_players):
            logger.warning("Ignoring state with seed/n_players mismatch: %s", state_path)
            return phi_by_perm, completed, truncated
        if prev_token != state_token:
            logger.warning("Ignoring state with config/source-data mismatch: %s", state_path)
            return phi_by_perm, completed, truncated

        n_copy = min(n_permutations, prev_phi.shape[0])
        phi_by_perm[:n_copy] = prev_phi[:n_copy]
        completed[:n_copy] = prev_completed[:n_copy]
        truncated[:n_copy] = prev_truncated[:n_copy]
        logger.info("Loaded %d completed permutations from %s", int(completed.sum()), state_path)
    except Exception as exc:
        logger.warning("Could not load permutation state from %s: %s", state_path, exc)

    return phi_by_perm, completed, truncated


def _save_state(
    state_path: Optional[Path],
    phi_by_perm: np.ndarray,
    completed: np.ndarray,
    truncated: np.ndarray,
    seed: int,
    state_token: str,
) -> None:
    if state_path is None:
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            phi_by_perm=phi_by_perm,
            completed=completed,
            truncated=truncated,
            seed=np.array([int(seed)], dtype=np.int64),
            state_token=np.array([state_token]),
            n_players=np.array([phi_by_perm.shape[1]], dtype=np.int64),
            n_permutations=np.array([phi_by_perm.shape[0]], dtype=np.int64),
        )
    Path(tmp_path).replace(state_path)


def _evaluate_indices(
    individual_indices: np.ndarray,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_eval: np.ndarray,
    y_eval_adjusted: np.ndarray,
    y_eval_original: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    cfg: ShapleyConfig,
    device: torch.device,
    seed: int,
    batch_size: int,
    snp_cols: Optional[np.ndarray],
    cache: Optional[IndividualUtilityCache],
) -> Dict[str, float]:
    individual_indices = np.asarray(np.sort(individual_indices), dtype=np.int64)
    if cache is not None:
        cached = cache.get(individual_indices)
        if cached is not None:
            return cached

    result = _evaluate_individual_subset(
        individual_indices=individual_indices,
        X_source=X_source,
        y_source=y_source,
        X_test=X_eval,
        y_test=y_eval_adjusted,
        y_eval_test=y_eval_original,
        X_val=X_val,
        y_val=y_val,
        train_params=None,
        device=device,
        loss_fn=None,
        cfg=cfg,
        seed=seed,
        batch_size=batch_size,
        snp_cols=snp_cols,
    )
    if cache is not None:
        cache.set(individual_indices, result)
    return result


def run_tmc_shapley_individuals(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    y_eval_cal: np.ndarray,
    cfg: ShapleyConfig,
    device: torch.device,
    batch_size: int,
    snp_cols: Optional[np.ndarray],
    output_dir: Path,
    target_code: int,
    raw_cfg: Dict[str, Any],
    state_token: str,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    n_players = int(X_source.shape[0])
    all_indices = np.arange(n_players, dtype=np.int64)
    cache = IndividualUtilityCache(enabled=bool(raw_cfg.get("tmc", {}).get("cache_utilities", False)))

    logger.info("Computing V_full over all %d source individuals...", n_players)
    v_full_result = _evaluate_indices(
        individual_indices=all_indices,
        X_source=X_source,
        y_source=y_source,
        X_eval=X_cal,
        y_eval_adjusted=y_cal,
        y_eval_original=y_eval_cal,
        X_val=None,
        y_val=None,
        cfg=cfg,
        device=device,
        seed=cfg.seed,
        batch_size=batch_size,
        snp_cols=snp_cols,
        cache=cache,
    )
    v_full = float(v_full_result["corr_eval"])
    logger.info("V_full(cal) = %.4f", v_full)

    rng = np.random.default_rng(cfg.seed)
    permutations = [rng.permutation(n_players).astype(np.int64) for _ in range(cfg.n_permutations)]

    state_path = _state_path(output_dir, target_code, cfg.seed, raw_cfg)
    phi_by_perm, completed, truncated = _load_state(
        state_path, cfg.n_permutations, n_players, cfg.seed, state_token
    )

    save_every = int(raw_cfg.get("tmc", {}).get("save_every", 1))
    save_every = max(1, save_every)
    progress_every = max(1, cfg.n_permutations // 5)
    n_new = 0

    for t, perm in enumerate(permutations):
        if completed[t]:
            continue

        prefix: List[int] = []
        local_phi = np.zeros(n_players, dtype=np.float64)
        old_v = 0.0
        was_truncated = False

        for step, player_idx in enumerate(perm, start=1):
            prefix.append(int(player_idx))
            result = _evaluate_indices(
                individual_indices=np.asarray(prefix, dtype=np.int64),
                X_source=X_source,
                y_source=y_source,
                X_eval=X_cal,
                y_eval_adjusted=y_cal,
                y_eval_original=y_eval_cal,
                X_val=None,
                y_val=None,
                cfg=cfg,
                device=device,
                seed=cfg.seed + t * 10_000 + step,
                batch_size=batch_size,
                snp_cols=snp_cols,
                cache=cache,
            )
            new_v = float(result["corr_eval"])
            local_phi[int(player_idx)] = new_v - old_v

            if (
                cfg.use_truncation
                and step >= cfg.min_prefix_islands
                and abs(new_v - v_full) < cfg.eps_trunc
            ):
                was_truncated = True
                break
            old_v = new_v

        phi_by_perm[t] = local_phi
        completed[t] = True
        truncated[t] = was_truncated
        n_new += 1

        done = int(completed.sum())
        if done % progress_every == 0 or done == cfg.n_permutations:
            logger.info(
                "TMC progress: %d/%d permutations, truncated=%d, utility evals=%d, cache hit rate=%.1f%%",
                done,
                cfg.n_permutations,
                int(truncated.sum()),
                int(cache.total_evals),
                100.0 * cache.hit_rate,
            )
        if n_new % save_every == 0:
            _save_state(state_path, phi_by_perm, completed, truncated, cfg.seed, state_token)

    _save_state(state_path, phi_by_perm, completed, truncated, cfg.seed, state_token)

    if not completed.any():
        raise RuntimeError("No TMC permutations completed.")

    phi = np.nan_to_num(phi_by_perm[completed], nan=0.0).mean(axis=0)
    stats = {
        "n_players": n_players,
        "n_permutations": int(completed.sum()),
        "n_truncated": int(truncated[completed].sum()),
        "n_utility_evals": int(cache.total_evals),
        "cache_hit_rate": float(cache.hit_rate),
        "v_full": v_full,
        "eps_trunc": float(cfg.eps_trunc),
        "use_truncation": bool(cfg.use_truncation),
        "min_prefix_individuals": int(cfg.min_prefix_islands),
        "state_path": str(state_path) if state_path is not None else None,
    }
    return phi, v_full, stats


def _curve_steps(n_players: int, curve_cfg: Dict[str, Any]) -> List[int]:
    explicit = curve_cfg.get("step_sizes", None)
    if explicit is not None:
        steps = [int(v) for v in explicit]
    else:
        n_points = max(2, int(curve_cfg.get("n_points", 6)))
        steps = np.rint(np.linspace(2, n_players, n_points)).astype(int).tolist()
    steps = sorted({max(2, min(n_players, int(v))) for v in steps if int(v) > 0})
    if n_players not in steps:
        steps.append(n_players)
    return steps


def compute_individual_add_curve(
    phi: np.ndarray,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    cfg: ShapleyConfig,
    raw_cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    snp_cols: Optional[np.ndarray],
) -> pd.DataFrame:
    curve_cfg = raw_cfg.get("evaluation_curve", {})
    if not bool(curve_cfg.get("enabled", True)):
        return pd.DataFrame()

    n_players = int(len(phi))
    steps = _curve_steps(n_players, curve_cfg)
    order = np.argsort(-phi)
    rows: List[Dict[str, Any]] = []

    for n_keep in steps:
        chosen = np.sort(order[:n_keep]).astype(np.int64)
        result = _evaluate_indices(
            individual_indices=chosen,
            X_source=X_source,
            y_source=y_source,
            X_eval=X_test,
            y_eval_adjusted=y_test,
            y_eval_original=y_eval_test,
            X_val=X_cal,
            y_val=y_cal,
            cfg=cfg,
            device=device,
            seed=cfg.seed + 900_000 + n_keep,
            batch_size=batch_size,
            snp_cols=snp_cols,
            cache=None,
        )
        rows.append({
            "n_individuals": int(n_keep),
            "corr_eval": float(result["corr_eval"]),
            "mse_adj": float(result["mse_adj"]),
            "method": "shapley_individual",
            "order_seed": -1,
        })
        logger.info("Add-curve Shapley: n=%d -> corr=%.4f", n_keep, result["corr_eval"])

    if bool(curve_cfg.get("include_random", True)):
        n_random_orders = int(curve_cfg.get("n_random_orders", 1))
        for r in range(n_random_orders):
            rng = np.random.default_rng(cfg.seed + 950_000 + r)
            random_order = rng.permutation(n_players)
            for n_keep in steps:
                chosen = np.sort(random_order[:n_keep]).astype(np.int64)
                result = _evaluate_indices(
                    individual_indices=chosen,
                    X_source=X_source,
                    y_source=y_source,
                    X_eval=X_test,
                    y_eval_adjusted=y_test,
                    y_eval_original=y_eval_test,
                    X_val=X_cal,
                    y_val=y_cal,
                    cfg=cfg,
                    device=device,
                    seed=cfg.seed + 960_000 + r * 10_000 + n_keep,
                    batch_size=batch_size,
                    snp_cols=snp_cols,
                    cache=None,
                )
                rows.append({
                    "n_individuals": int(n_keep),
                    "corr_eval": float(result["corr_eval"]),
                    "mse_adj": float(result["mse_adj"]),
                    "method": "random_individual",
                    "order_seed": int(r),
                })

    return pd.DataFrame(rows)


def _save_outputs(
    output_dir: Path,
    target_code: int,
    target_name: str,
    repeat_idx: int,
    split_seed: int,
    ids_source: np.ndarray,
    selected_source_idx: np.ndarray,
    locality_source: np.ndarray,
    code_to_label: Dict[int, str],
    phi: np.ndarray,
    v_full: float,
    stats: Dict[str, Any],
    shapley_cfg: ShapleyConfig,
    snp_cols: Optional[np.ndarray],
    add_curve_df: pd.DataFrame,
    raw_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    order = np.argsort(-phi)
    ranks = np.empty(len(phi), dtype=np.int64)
    ranks[order] = np.arange(1, len(phi) + 1)
    source_names = [island_label(int(c), code_to_label) for c in locality_source]

    phi_df = pd.DataFrame({
        "target_island": int(target_code),
        "target_island_name": str(target_name),
        "repeat": int(repeat_idx),
        "split_seed": int(split_seed),
        "player_index": np.arange(len(phi), dtype=np.int64),
        "source_pool_index": selected_source_idx.astype(np.int64),
        "source_id": ids_source.astype(str),
        "source_island": locality_source.astype(int),
        "source_island_name": source_names,
        "phi": phi.astype(float),
        "rank_phi": ranks.astype(int),
        "v_full": float(v_full),
        "n_permutations": int(stats["n_permutations"]),
    }).sort_values("rank_phi")

    phi_path = output_dir / f"shapley_individual_values_target_{target_code}.csv"
    phi_df.to_csv(phi_path, index=False)
    logger.info("Saved individual Shapley values to %s", phi_path)

    if snp_cols is not None:
        snp_path = output_dir / f"snp_cols_target_{target_code}.csv"
        pd.DataFrame({"snp_col": snp_cols.astype(np.int64)}).to_csv(snp_path, index=False)

    if not add_curve_df.empty:
        add_df = add_curve_df.copy()
        add_df["target_island"] = int(target_code)
        add_df["target_island_name"] = str(target_name)
        add_df["repeat"] = int(repeat_idx)
        add_df["split_seed"] = int(split_seed)
        add_path = output_dir / f"add_curve_individual_target_{target_code}.csv"
        add_df.to_csv(add_path, index=False)

        remove_df = add_df.copy()
        remove_df["n_removed"] = int(len(phi)) - remove_df["n_individuals"].astype(int)
        remove_df["n_individuals_remaining"] = remove_df["n_individuals"].astype(int)
        remove_path = output_dir / f"remove_curve_individual_target_{target_code}.csv"
        remove_df.to_csv(remove_path, index=False)
    else:
        add_df = pd.DataFrame()
        remove_df = pd.DataFrame()

    summary = {
        "target_island": int(target_code),
        "target_island_name": str(target_name),
        "repeat": int(repeat_idx),
        "split_seed": int(split_seed),
        "stats": stats,
        "model": {
            "type": shapley_cfg.model_type,
            "ridge_alpha": float(shapley_cfg.ridge_alpha),
        },
        "snp_selection": {
            "enabled": snp_cols is not None,
            "n_snps": int(len(snp_cols)) if snp_cols is not None else None,
            "mode": raw_cfg.get("snp_selection_mode", "random"),
        },
        "individual_sampling": raw_cfg.get("individual_sampling", {}),
        "shapley_config": asdict(shapley_cfg),
    }
    summary_path = output_dir / f"tmc_individual_summary_target_{target_code}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return phi_df, add_df, remove_df


def run_for_target(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    locality: np.ndarray,
    y_eval: np.ndarray,
    code_to_label: Dict[int, str],
    target_code: int,
    included_island_codes: List[int],
    raw_cfg: Dict[str, Any],
    device: torch.device,
) -> None:
    seed = int(raw_cfg.get("seed", 42))
    target_name = island_label(target_code, code_to_label)
    output_dir = Path(raw_cfg["paths"].get("output_dir", "outputs/tmc_shapley_individuals")) / f"island_{target_code}"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_codes = [int(c) for c in included_island_codes if int(c) != int(target_code)]
    if not source_codes:
        raise ValueError(f"No source islands available for target {target_code}")

    uncertainty_cfg = raw_cfg.get("uncertainty", {})
    n_repeats = int(uncertainty_cfg.get("n_target_split_repeats", 1))
    if n_repeats < 1:
        raise ValueError("uncertainty.n_target_split_repeats must be >= 1")

    phi_repeat_dfs: List[pd.DataFrame] = []
    add_repeat_dfs: List[pd.DataFrame] = []
    remove_repeat_dfs: List[pd.DataFrame] = []
    completed_repeats: List[Dict[str, int]] = []

    for repeat_idx in range(n_repeats):
        split_seed = seed + 1_000 * repeat_idx
        repeat_output_dir = output_dir / f"repeat_{repeat_idx}"
        repeat_output_dir.mkdir(parents=True, exist_ok=True)

        split = split_target_island(
            X,
            y,
            ids,
            locality,
            target_code,
            y_eval=y_eval,
            cal_fraction=float(raw_cfg.get("cal_fraction", 0.2)),
            seed=split_seed,
            max_cal_fraction=raw_cfg.get("max_cal_fraction", None),
        )

        in_source = np.isin(split["locality_source"], source_codes)
        X_source_all = split["X_source"][in_source]
        y_source_all = split["y_source"][in_source]
        ids_source_all = split["ids_source"][in_source]
        locality_source_all = split["locality_source"][in_source]

        selected_source_idx = _sample_source_individuals(
            locality_source=locality_source_all,
            source_codes=source_codes,
            sampling_cfg=raw_cfg.get("individual_sampling", {}),
            seed=seed,
        )

        X_source = X_source_all[selected_source_idx]
        y_source = y_source_all[selected_source_idx]
        ids_source = ids_source_all[selected_source_idx]
        locality_source = locality_source_all[selected_source_idx]

        logger.info(
            "Target %d (%s), repeat %d/%d: source pool %d -> sampled players %d, cal=%d, test=%d, split_seed=%d",
            target_code,
            target_name,
            repeat_idx + 1,
            n_repeats,
            len(X_source_all),
            len(X_source),
            len(split["X_cal"]),
            len(split["X_test"]),
            split_seed,
        )

        shapley_cfg = _build_shapley_config(raw_cfg, split_seed)
        batch_size = int(raw_cfg.get("batch_size", 64))
        snp_cols = _select_snp_cols(X_source, y_source, raw_cfg, seed)
        state_token = _make_state_token(ids_source, locality_source, snp_cols, shapley_cfg)

        phi, v_full, stats = run_tmc_shapley_individuals(
            X_source=X_source,
            y_source=y_source,
            X_cal=split["X_cal"],
            y_cal=split["y_cal"],
            y_eval_cal=split["y_eval_cal"],
            cfg=shapley_cfg,
            device=device,
            batch_size=batch_size,
            snp_cols=snp_cols,
            output_dir=repeat_output_dir,
            target_code=target_code,
            raw_cfg=raw_cfg,
            state_token=state_token,
        )

        add_curve_df = compute_individual_add_curve(
            phi=phi,
            X_source=X_source,
            y_source=y_source,
            X_cal=split["X_cal"],
            y_cal=split["y_cal"],
            X_test=split["X_test"],
            y_test=split["y_test"],
            y_eval_test=split["y_eval_test"],
            cfg=shapley_cfg,
            raw_cfg=raw_cfg,
            device=device,
            batch_size=batch_size,
            snp_cols=snp_cols,
        )

        phi_df, add_df, remove_df = _save_outputs(
            output_dir=repeat_output_dir,
            target_code=target_code,
            target_name=target_name,
            repeat_idx=repeat_idx,
            split_seed=split_seed,
            ids_source=ids_source,
            selected_source_idx=selected_source_idx,
            locality_source=locality_source,
            code_to_label=code_to_label,
            phi=phi,
            v_full=v_full,
            stats=stats,
            shapley_cfg=shapley_cfg,
            snp_cols=snp_cols,
            add_curve_df=add_curve_df,
            raw_cfg=raw_cfg,
        )
        phi_repeat_dfs.append(phi_df)
        if not add_df.empty:
            add_repeat_dfs.append(add_df)
        if not remove_df.empty:
            remove_repeat_dfs.append(remove_df)
        completed_repeats.append({"repeat": int(repeat_idx), "split_seed": int(split_seed)})

    if not phi_repeat_dfs:
        raise RuntimeError(f"No individual-wise Shapley repeats completed for target {target_code}")

    phi_repeats_df = pd.concat(phi_repeat_dfs, ignore_index=True)
    repeats_path = output_dir / f"shapley_individual_repeats_target_{target_code}.csv"
    phi_repeats_df.to_csv(repeats_path, index=False)

    phi_summary_df = (
        phi_repeats_df
        .groupby([
            "target_island",
            "target_island_name",
            "source_id",
            "source_island",
            "source_island_name",
        ], as_index=False)
        .agg(
            phi_mean=("phi", "mean"),
            phi_std=("phi", "std"),
            phi_p05=("phi", lambda x: np.quantile(x, 0.05)),
            phi_p95=("phi", lambda x: np.quantile(x, 0.95)),
            rank_phi_mean=("rank_phi", "mean"),
            n_repeats=("phi", "size"),
        )
    )
    phi_summary_df["rank_phi_mean_order"] = (
        phi_summary_df["phi_mean"].rank(ascending=False, method="min").astype(int)
    )
    phi_summary_df = phi_summary_df.sort_values("rank_phi_mean_order")
    summary_path = output_dir / f"shapley_individual_uncertainty_target_{target_code}.csv"
    phi_summary_df.to_csv(summary_path, index=False)

    if add_repeat_dfs:
        add_repeats_df = pd.concat(add_repeat_dfs, ignore_index=True)
        add_repeats_path = output_dir / f"add_curve_individual_repeats_target_{target_code}.csv"
        add_repeats_df.to_csv(add_repeats_path, index=False)
        add_summary_df = (
            add_repeats_df
            .groupby(["method", "n_individuals"], as_index=False)
            .agg(
                corr_mean=("corr_eval", "mean"),
                corr_std=("corr_eval", "std"),
                mse_mean=("mse_adj", "mean"),
                mse_std=("mse_adj", "std"),
                n_rows=("corr_eval", "size"),
            )
        )
        add_summary_path = output_dir / f"add_curve_individual_uncertainty_target_{target_code}.csv"
        add_summary_df.to_csv(add_summary_path, index=False)

    if remove_repeat_dfs:
        remove_repeats_df = pd.concat(remove_repeat_dfs, ignore_index=True)
        remove_repeats_path = output_dir / f"remove_curve_individual_repeats_target_{target_code}.csv"
        remove_repeats_df.to_csv(remove_repeats_path, index=False)
        remove_summary_df = (
            remove_repeats_df
            .groupby(["method", "n_removed"], as_index=False)
            .agg(
                corr_mean=("corr_eval", "mean"),
                corr_std=("corr_eval", "std"),
                mse_mean=("mse_adj", "mean"),
                mse_std=("mse_adj", "std"),
                n_rows=("corr_eval", "size"),
            )
        )
        remove_summary_path = output_dir / f"remove_curve_individual_uncertainty_target_{target_code}.csv"
        remove_summary_df.to_csv(remove_summary_path, index=False)

    uncertainty_summary = {
        "target_island": int(target_code),
        "target_island_name": str(target_name),
        "n_target_split_repeats_requested": int(n_repeats),
        "n_target_split_repeats_completed": int(len(completed_repeats)),
        "completed_repeats": completed_repeats,
    }
    uncertainty_summary_path = output_dir / f"uncertainty_summary_target_{target_code}.json"
    with open(uncertainty_summary_path, "w") as f:
        json.dump(uncertainty_summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run individual-level TMC-Shapley")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument(
        "--target_islands",
        nargs="+",
        default=None,
        help="Override target_islands from config (encoded codes, labels, or names)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = _device_from_config(cfg)
    logger.info("Using device: %s", device)

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
    logger.info("Data loaded: %d samples, %d SNPs, %d islands", X.shape[0], X.shape[1], len(np.unique(locality)))

    present_codes = set(int(c) for c in np.unique(locality))
    included_raw = cfg.get("included_islands", None)
    if included_raw is None:
        included_island_codes = sorted(present_codes)
    else:
        included_island_codes = [
            resolve_island_code(v, code_to_label, present_codes)
            for v in included_raw
        ]
    logger.info("Included islands: %s", included_island_codes)

    raw_targets = args.target_islands if args.target_islands else cfg.get("target_islands", [0])
    target_codes = [
        resolve_island_code(t, code_to_label, present_codes)
        for t in raw_targets
    ]
    logger.info("Target islands: %s", target_codes)

    for target_code in target_codes:
        logger.info("%s", "=" * 70)
        run_for_target(
            X=X,
            y=y,
            ids=ids,
            locality=locality,
            y_eval=y_eval,
            code_to_label=code_to_label,
            target_code=int(target_code),
            included_island_codes=included_island_codes,
            raw_cfg=cfg,
            device=device,
        )

    logger.info("All target islands processed.")


if __name__ == "__main__":
    main()
