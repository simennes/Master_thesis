"""
TMC-Shapley at island level for genomic prediction.

Implements group-level Shapley values where each source island is a "player"
and the utility V(S) is the Pearson correlation on the target island test set
after training an MLP on individuals from the island subset S.

Reference: Ghorbani & Zou, 2019 (Data Shapley)
"""
from __future__ import annotations

import copy
import gc
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from .models import TrainParams, make_model
from .removal_curve import split_target_island
from .training import evaluate_model, train_simple
from .utils import _optimizer, _pearson_corr, make_loss, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility cache
# ---------------------------------------------------------------------------

class UtilityCache:
    """
    Thread-safe in-memory + on-disk cache mapping island-subset bitmask -> utility dict.

    Keys are stored as strings in JSON because JSON requires string keys.
    """

    def __init__(self, cache_path: Optional[str] = None):
        self._cache: Dict[int, Dict[str, float]] = {}
        self._lock = threading.Lock()
        self.cache_path = cache_path
        self.hits = 0
        self.misses = 0

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                raw = json.load(f)
            self._cache = {int(k): v for k, v in raw.items()}
            logger.info(f"Loaded {len(self._cache)} cached utilities from {cache_path}")

    def get(self, mask: int) -> Optional[Dict[str, float]]:
        with self._lock:
            v = self._cache.get(mask)
            if v is not None:
                self.hits += 1
            else:
                self.misses += 1
            return v

    def set(self, mask: int, value: Dict[str, float]) -> None:
        with self._lock:
            self._cache[mask] = value

    def save(self) -> None:
        with self._lock:
            if not self.cache_path:
                return
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump({str(k): v for k, v in self._cache.items()}, f)

    @property
    def hit_rate(self) -> float:
        with self._lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0

    @property
    def total_evals(self) -> int:
        with self._lock:
            return self.misses


def _detect_devices(primary_device: torch.device) -> List[torch.device]:
    """
    Detect available compute devices for parallelism.

    Returns a list of devices.  If multiple CUDA GPUs are available, returns
    all of them so work can be distributed.  Otherwise returns [primary_device].
    """
    if primary_device.type == "cuda":
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            logger.info(f"Multi-GPU detected: {n_gpus} devices available for parallelism")
            return [torch.device(f"cuda:{i}") for i in range(n_gpus)]
    return [primary_device]


# ---------------------------------------------------------------------------
# Subset utility evaluation
# ---------------------------------------------------------------------------

@dataclass
class ShapleyConfig:
    """Configuration for TMC-Shapley run."""
    n_permutations: int = 50
    eps_trunc: float = 0.005
    min_prefix_islands: int = 3
    use_truncation: bool = False
    island_sampling_mode: str = "mean"       # "mean" | "full"
    cap_per_island: Optional[int] = 150      # max individuals per island in V(S)
    n_train_seeds_per_eval: int = 1
    cache_dir: Optional[str] = None
    n_random_orders: int = 20
    use_early_stopping: bool = False
    early_stopping_patience: int = 0
    retrain_use_early_stopping: bool = True
    retrain_early_stopping_patience: int = 20
    snp_selection_mode: str = "random"
    use_snp_selection: bool = False
    num_snps: Optional[int] = None
    model_type: str = "mlp"
    ridge_alpha: float = 1e5
    permutation_state_path: Optional[str] = None
    seed: int = 42


def _island_mask(island_indices: List[int]) -> int:
    """Convert list of 0-based island indices to integer bitmask."""
    mask = 0
    for i in island_indices:
        mask |= (1 << i)
    return mask


def _mask_to_indices(mask: int, n_islands: int) -> List[int]:
    """Convert bitmask back to list of 0-based island indices."""
    return [i for i in range(n_islands) if (mask >> i) & 1]


def _build_training_data(
    island_indices: List[int],
    X_source: np.ndarray,
    y_source: np.ndarray,
    locality_source: np.ndarray,
    source_codes: List[int],
    cap_per_island: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X_train, y_train) from the islands specified by island_indices.

    With cap_per_island, randomly subsample (deterministically via rng) each island
    to at most cap_per_island individuals for balanced training.
    """
    x_parts = []
    y_parts = []

    for idx in island_indices:
        code = source_codes[idx]
        mask = (locality_source == code)
        X_k = X_source[mask]
        y_k = y_source[mask]

        n_k = len(X_k)
        if cap_per_island is not None and n_k > cap_per_island:
            chosen = rng.choice(n_k, size=cap_per_island, replace=False)
            X_k = X_k[chosen]
            y_k = y_k[chosen]

        x_parts.append(X_k)
        y_parts.append(y_k)

    if not x_parts:
        return np.empty((0, X_source.shape[1]), dtype=X_source.dtype), np.empty(0, dtype=y_source.dtype)

    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def evaluate_subset(
    island_indices: List[int],
    source_codes: List[int],
    X_source: np.ndarray,
    y_source: np.ndarray,
    locality_source: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    train_params: Optional[TrainParams],
    device: torch.device,
    loss_fn: Optional[nn.Module],
    cfg: ShapleyConfig,
    seed: int,
    batch_size: int = 64,
    snp_cols: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate V(S) = (corr_eval, mse_adj) for a subset S of source islands.

    Returns {'corr_eval': float, 'mse_adj': float}.
    If S is empty, returns {'corr_eval': 0.0, 'mse_adj': float('inf')}.
    """
    if len(island_indices) == 0:
        return {"corr_eval": 0.0, "mse_adj": float("inf")}

    rng = np.random.default_rng(seed)

    X_train, y_train = _build_training_data(
        island_indices,
        X_source,
        y_source,
        locality_source,
        source_codes,
        cfg.cap_per_island,
        rng,
    )

    if len(X_train) < 2:
        return {"corr_eval": 0.0, "mse_adj": float("inf")}

    # Apply SNP feature selection if precomputed
    if snp_cols is not None:
        X_train_sel = X_train[:, snp_cols]
        X_test_sel = X_test[:, snp_cols]
        X_val_sel = X_val[:, snp_cols] if X_val is not None else None
    else:
        X_train_sel = X_train
        X_test_sel = X_test
        X_val_sel = X_val

    in_dim = X_train_sel.shape[1]

    if cfg.model_type.lower() == "ridge":
        alpha = max(float(cfg.ridge_alpha), 1e-12)
        model = Ridge(alpha=alpha)
        model.fit(X_train_sel, y_train)
        pred = model.predict(X_test_sel)
        corr_eval = float(_pearson_corr(pred, y_eval_test))
        if not np.isfinite(corr_eval):
            corr_eval = 0.0
        mse_adj = float(np.mean((pred - y_test) ** 2))
        return {
            "corr_eval": corr_eval,
            "mse_adj": mse_adj,
        }

    if train_params is None or loss_fn is None:
        raise ValueError("train_params and loss_fn are required when model_type='mlp'")

    # Average utility over multiple training seeds to reduce noise
    corr_vals = []
    mse_vals = []

    for seed_offset in range(cfg.n_train_seeds_per_eval):
        eval_seed = seed + seed_offset * 10_000

        set_seed(eval_seed)
        model = make_model(in_dim, train_params)
        opt = _optimizer(
            train_params.optimizer,
            model.parameters(),
            train_params.lr,
            train_params.weight_decay,
        )

        x_t = torch.from_numpy(X_train_sel).float()
        y_t = torch.from_numpy(y_train).float()
        x_test_t = torch.from_numpy(X_test_sel).float()
        y_test_t = torch.from_numpy(y_test).float()
        y_eval_test_t = torch.from_numpy(y_eval_test).float()
        x_val_t = torch.from_numpy(X_val_sel).float() if X_val_sel is not None else None
        y_val_t = torch.from_numpy(y_val).float() if y_val is not None else None

        patience = cfg.early_stopping_patience if cfg.use_early_stopping else 0

        model, _ = train_simple(
            model=model,
            x_train=x_t,
            y_train=y_t,
            epochs=train_params.epochs,
            optimizer=opt,
            loss_fn=loss_fn,
            device=device,
            batch_size=batch_size,
            seed=eval_seed,
            patience=patience,
            x_val=x_val_t,
            y_val=y_val_t,
        )

        corr_eval, mse_adj = evaluate_model(
            model, x_test_t, y_test_t, device, y_eval=y_eval_test_t
        )
        corr_vals.append(corr_eval)
        mse_vals.append(mse_adj)

        del model, x_t, y_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "corr_eval": float(np.mean(corr_vals)),
        "mse_adj": float(np.mean(mse_vals)),
    }


def _cached_evaluate(
    island_indices: List[int],
    source_codes: List[int],
    X_source: np.ndarray,
    y_source: np.ndarray,
    locality_source: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    train_params: Optional[TrainParams],
    device: torch.device,
    loss_fn: Optional[nn.Module],
    cfg: ShapleyConfig,
    seed: int,
    cache: UtilityCache,
    snp_cols: Optional[np.ndarray],
    batch_size: int = 64,
) -> Dict[str, float]:
    """Evaluate with caching; key is bitmask over source_codes indices."""
    mask = _island_mask(island_indices)
    cached = cache.get(mask)
    if cached is not None:
        return cached

    result = evaluate_subset(
        island_indices=island_indices,
        source_codes=source_codes,
        X_source=X_source,
        y_source=y_source,
        locality_source=locality_source,
        X_test=X_test,
        y_test=y_test,
        y_eval_test=y_eval_test,
        X_val=X_val,
        y_val=y_val,
        train_params=train_params,
        device=device,
        loss_fn=loss_fn,
        cfg=cfg,
        seed=seed,
        batch_size=batch_size,
        snp_cols=snp_cols,
    )
    cache.set(mask, result)
    return result


# ---------------------------------------------------------------------------
# TMC main loop
# ---------------------------------------------------------------------------

def run_tmc_shapley(
    source_codes: List[int],
    X_source: np.ndarray,
    y_source: np.ndarray,
    locality_source: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    train_params: Optional[TrainParams],
    device: torch.device,
    cfg: ShapleyConfig,
    cache: UtilityCache,
    batch_size: int = 64,
    snp_cols: Optional[np.ndarray] = None,
    devices: Optional[List[torch.device]] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Run TMC-Shapley and return per-island Shapley values.

    If *devices* contains multiple GPUs, permutations are distributed across
    them using a thread pool.  Otherwise runs sequentially.

    Parameters
    ----------
    source_codes : List[int]
        Encoded island codes for all source islands (players).
        Index i in this list maps to bit i in the bitmask.

    Returns
    -------
    shapley_values : np.ndarray of shape (K,)
        Shapley value phi_k for each source island k (same order as source_codes).
    v_full : float
        Utility when trained on all source islands.
    stats : dict
        Run statistics (n_evals, hit_rate, v_full, etc.)
    """
    if devices is None:
        devices = [device]
    n_workers = len(devices) if len(devices) > 1 else 1

    K = len(source_codes)
    if K == 0:
        raise ValueError("No source islands to compute Shapley values for.")

    all_indices = list(range(K))
    loss_fn = make_loss(train_params.loss_name) if cfg.model_type.lower() == "mlp" else None

    # Common kwargs for _cached_evaluate (device will be overridden per-worker)
    def _make_eval_kwargs(dev: torch.device) -> dict:
        return dict(
            source_codes=source_codes,
            X_source=X_source,
            y_source=y_source,
            locality_source=locality_source,
            X_test=X_test,
            y_test=y_test,
            y_eval_test=y_eval_test,
            X_val=None,
            y_val=None,
            train_params=train_params,
            device=dev,
            loss_fn=loss_fn,
            cfg=cfg,
            seed=cfg.seed,
            cache=cache,
            snp_cols=snp_cols,
            batch_size=batch_size,
        )

    # Compute V_full (utility of all source islands) on primary device
    logger.info(f"Computing V_full over all {K} source islands...")
    v_full_result = _cached_evaluate(island_indices=all_indices, **_make_eval_kwargs(device))
    v_full = v_full_result["corr_eval"]
    logger.info(f"V_full = {v_full:.4f}")

    # Pre-generate all permutations (reproducible regardless of parallelism)
    rng = np.random.default_rng(cfg.seed)
    all_perms = [rng.permutation(K).tolist() for _ in range(cfg.n_permutations)]

    # Optional permutation-level state for explicit "head-start" across
    # increasing n_permutations for the same split/seed.
    state_path = cfg.permutation_state_path
    local_phi_by_perm: List[Optional[np.ndarray]] = [None] * cfg.n_permutations
    trunc_by_perm: List[bool] = [False] * cfg.n_permutations
    start_perm = 0

    def _save_perm_state() -> None:
        if not state_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
            completed = 0
            for i, arr in enumerate(local_phi_by_perm):
                if arr is not None:
                    completed = i + 1
                else:
                    break
            payload = {
                "seed": int(cfg.seed),
                "n_islands": int(K),
                "n_permutations_completed": int(completed),
                "local_phi_by_perm": [
                    arr.tolist() if arr is not None else None
                    for arr in local_phi_by_perm[:completed]
                ],
                "trunc_by_perm": [bool(x) for x in trunc_by_perm[:completed]],
            }
            with open(state_path, "w") as f:
                json.dump(payload, f)
        except Exception as e:
            logger.warning(f"Could not save permutation state to {state_path}: {e}")

    if state_path and os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                st = json.load(f)
            if (
                int(st.get("seed", -1)) == int(cfg.seed)
                and int(st.get("n_islands", -1)) == int(K)
            ):
                prev = int(st.get("n_permutations_completed", 0))
                prev = max(0, min(prev, cfg.n_permutations))
                phi_list = st.get("local_phi_by_perm", [])
                trunc_list = st.get("trunc_by_perm", [])
                for t in range(prev):
                    arr = phi_list[t] if t < len(phi_list) else None
                    if arr is None:
                        break
                    local_phi_by_perm[t] = np.asarray(arr, dtype=np.float64)
                    trunc_by_perm[t] = bool(trunc_list[t]) if t < len(trunc_list) else False
                start_perm = sum(v is not None for v in local_phi_by_perm)
                if start_perm > 0:
                    logger.info(
                        "TMC head-start: loaded %d completed permutations from state %s",
                        start_perm,
                        state_path,
                    )
        except Exception as e:
            logger.warning(f"Could not load permutation state from {state_path}: {e}")

    # ---------- worker function for one permutation ----------
    def _run_permutation(t: int, dev: torch.device) -> Tuple[np.ndarray, bool]:
        """Run permutation *t* on *dev*, return (local_phi, truncated)."""
        perm = all_perms[t]
        local_phi = np.zeros(K, dtype=np.float64)
        ek = _make_eval_kwargs(dev)
        prefix: List[int] = []
        old_v = 0.0

        for step, k in enumerate(perm):
            prefix.append(k)
            result = _cached_evaluate(island_indices=sorted(prefix), **ek)
            new_v = result["corr_eval"]
            local_phi[k] = new_v - old_v

            if (
                cfg.use_truncation
                and len(prefix) >= cfg.min_prefix_islands
                and abs(new_v - v_full) < cfg.eps_trunc
            ):
                return local_phi, True
            old_v = new_v

        return local_phi, False

    # ---------- execute permutations ----------
    phi = np.zeros(K, dtype=np.float64)
    n_permutations_run = 0
    truncated_count = 0

    if n_workers > 1 and start_perm < cfg.n_permutations:
        logger.info(
            f"Running {cfg.n_permutations - start_perm} new TMC permutations across "
            f"{n_workers} GPUs (thread pool)"
        )
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_run_permutation, t, devices[t % n_workers]): t
                for t in range(start_perm, cfg.n_permutations)
            }
            for future in as_completed(futures):
                t = futures[future]
                local_phi, truncated = future.result()
                local_phi_by_perm[t] = local_phi
                trunc_by_perm[t] = truncated

                done = sum(v is not None for v in local_phi_by_perm)
                if done % max(1, cfg.n_permutations // 5) == 0:
                    logger.info(
                        f"TMC progress: {done}/{cfg.n_permutations} permutations done, "
                        f"cache hit rate: {cache.hit_rate:.1%}, "
                        f"n_evals: {cache.total_evals}"
                    )
                if done % 10 == 0:
                    cache.save()
                    _save_perm_state()
    elif start_perm < cfg.n_permutations:
        for t in range(start_perm, cfg.n_permutations):
            local_phi, truncated = _run_permutation(t, device)
            local_phi_by_perm[t] = local_phi
            trunc_by_perm[t] = truncated

            if (t + 1) % max(1, cfg.n_permutations // 5) == 0:
                logger.info(
                    f"TMC progress: {t + 1}/{cfg.n_permutations} permutations done, "
                    f"cache hit rate: {cache.hit_rate:.1%}, "
                    f"n_evals: {cache.total_evals}"
                )
            if (t + 1) % 10 == 0:
                cache.save()
                _save_perm_state()
    else:
        logger.info("No new permutations needed; using fully loaded permutation head-start state.")

    # Accumulate in permutation index order and record intermediate ranking trace
    intermediate_rows: List[Dict[str, Any]] = []
    for t in range(cfg.n_permutations):
        local_phi = local_phi_by_perm[t]
        if local_phi is None:
            continue
        phi += local_phi
        n_permutations_run += 1
        if trunc_by_perm[t]:
            truncated_count += 1

        phi_running = phi / n_permutations_run
        order = np.argsort(-phi_running)
        ranks = np.empty(K, dtype=np.int64)
        ranks[order] = np.arange(1, K + 1)
        for k in range(K):
            intermediate_rows.append({
                "permutation_index": n_permutations_run,
                "source_island": int(source_codes[k]),
                "phi_running": float(phi_running[k]),
                "rank_running": int(ranks[k]),
            })

    # Normalize by number of permutations
    phi /= n_permutations_run

    cache.save()
    _save_perm_state()

    stats = {
        "n_permutations": n_permutations_run,
        "n_truncated": truncated_count,
        "n_utility_evals": cache.total_evals,
        "cache_hit_rate": cache.hit_rate,
        "v_full": v_full,
        "eps_trunc": cfg.eps_trunc,
        "use_truncation": cfg.use_truncation,
        "min_prefix_islands": cfg.min_prefix_islands,
        "intermediate_rankings": intermediate_rows,
    }

    logger.info(
        f"TMC complete: {n_permutations_run} permutations, "
        f"{truncated_count} truncated, {cache.total_evals} utility evals, "
        f"cache hit rate: {cache.hit_rate:.1%}"
    )

    return phi, v_full, stats


# ---------------------------------------------------------------------------
# Individual-level evaluation helper (for random-individual baseline)
# ---------------------------------------------------------------------------

def _evaluate_individual_subset(
    individual_indices: np.ndarray,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    train_params: Optional[TrainParams],
    device: torch.device,
    loss_fn: Optional[nn.Module],
    cfg: ShapleyConfig,
    seed: int,
    batch_size: int = 64,
    snp_cols: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Train on an arbitrary subset of individuals and evaluate on test."""
    if len(individual_indices) < 2:
        return {"corr_eval": 0.0, "mse_adj": float("inf")}

    X_train = X_source[individual_indices]
    y_train = y_source[individual_indices]

    if snp_cols is not None:
        X_train = X_train[:, snp_cols]
        X_test_sel = X_test[:, snp_cols]
        X_val_sel = X_val[:, snp_cols] if X_val is not None else None
    else:
        X_test_sel = X_test
        X_val_sel = X_val

    in_dim = X_train.shape[1]

    if cfg.model_type.lower() == "ridge":
        alpha = max(float(cfg.ridge_alpha), 1e-12)
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        pred = model.predict(X_test_sel)
        corr_eval = float(_pearson_corr(pred, y_eval_test))
        if not np.isfinite(corr_eval):
            corr_eval = 0.0
        mse_adj = float(np.mean((pred - y_test) ** 2))
        return {
            "corr_eval": corr_eval,
            "mse_adj": mse_adj,
        }

    if train_params is None or loss_fn is None:
        raise ValueError("train_params and loss_fn are required when model_type='mlp'")
    corr_vals, mse_vals = [], []

    for seed_offset in range(cfg.n_train_seeds_per_eval):
        eval_seed = seed + seed_offset * 10_000
        set_seed(eval_seed)
        model = make_model(in_dim, train_params)
        opt = _optimizer(
            train_params.optimizer, model.parameters(),
            train_params.lr, train_params.weight_decay,
        )

        x_t = torch.from_numpy(X_train).float()
        y_t = torch.from_numpy(y_train).float()
        x_test_t = torch.from_numpy(X_test_sel).float()
        y_test_t = torch.from_numpy(y_test).float()
        y_eval_t = torch.from_numpy(y_eval_test).float()
        x_val_t = torch.from_numpy(X_val_sel).float() if X_val_sel is not None else None
        y_val_t = torch.from_numpy(y_val).float() if y_val is not None else None

        patience = cfg.early_stopping_patience if cfg.use_early_stopping else 0
        model, _ = train_simple(
            model=model, x_train=x_t, y_train=y_t,
            epochs=train_params.epochs, optimizer=opt, loss_fn=loss_fn,
            device=device, batch_size=batch_size, seed=eval_seed,
            patience=patience,
            x_val=x_val_t,
            y_val=y_val_t,
        )

        corr_eval, mse_adj = evaluate_model(
            model, x_test_t, y_test_t, device, y_eval=y_eval_t
        )
        corr_vals.append(corr_eval)
        mse_vals.append(mse_adj)

        del model, x_t, y_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "corr_eval": float(np.mean(corr_vals)),
        "mse_adj": float(np.mean(mse_vals)),
    }


# ---------------------------------------------------------------------------
# Add / remove curves
# ---------------------------------------------------------------------------

def compute_add_curve(
    phi: np.ndarray,
    source_codes: List[int],
    n_individuals: np.ndarray,
    X_source: np.ndarray,
    y_source: np.ndarray,
    locality_source: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    X_cal: Optional[np.ndarray],
    y_cal: Optional[np.ndarray],
    train_params: Optional[TrainParams],
    device: torch.device,
    cfg: ShapleyConfig,
    snp_cols: Optional[np.ndarray],
    n_random_orders: int = 20,
    seed: int = 42,
    batch_size: int = 64,
    devices: Optional[List[torch.device]] = None,
    include_shapley_mean: bool = True,
    include_random_individual: bool = True,
    random_step_counts: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute add-island curves with two methods:

    - shapley_mean  : add islands in descending phi/n order
    - random_individual : add random individuals (count matches shapley_mean steps)

    If *devices* contains multiple GPUs, the random-individual evaluations
    are distributed across them using a thread pool.

    Returns a single DataFrame with columns:
        n_islands, n_individuals, corr_eval, mse_adj, method, order_seed
    """
    if devices is None:
        devices = [device]
    n_workers = len(devices) if len(devices) > 1 else 1

    K = len(source_codes)
    loss_fn = make_loss(train_params.loss_name) if cfg.model_type.lower() == "mlp" else None
    curve_cache = UtilityCache()  # fresh in-memory cache (test-set evaluations)

    eval_kwargs = dict(
        source_codes=source_codes,
        X_source=X_source,
        y_source=y_source,
        locality_source=locality_source,
        X_test=X_test,
        y_test=y_test,
        y_eval_test=y_eval_test,
        X_val=X_cal,
        y_val=y_cal,
        train_params=train_params,
        device=device,
        loss_fn=loss_fn,
        cfg=cfg,
        seed=seed,
        cache=curve_cache,
        snp_cols=snp_cols,
        batch_size=batch_size,
    )

    rows: List[Dict[str, Any]] = []

    # ---- Shapley mean order (descending phi/n) ----
    phi_per_n = phi / np.maximum(n_individuals.astype(np.float64), 1.0)
    mean_order = np.argsort(-phi_per_n).tolist()
    mean_cum = np.cumsum([int(n_individuals[mean_order[m]]) for m in range(K)])

    shapley_full_result: Optional[Dict[str, float]] = None
    if include_shapley_mean:
        cum_n = 0
        for m in range(1, K + 1):
            cum_n += int(n_individuals[mean_order[m - 1]])
            top_m = sorted(mean_order[:m])
            result = _cached_evaluate(island_indices=top_m, **eval_kwargs)
            rows.append({
                "n_islands": m, "n_individuals": cum_n,
                "corr_eval": result["corr_eval"], "mse_adj": result["mse_adj"],
                "method": "shapley_mean", "order_seed": -1,
            })
            logger.info(
                f"Add-curve (mean): {m} islands, n={cum_n} -> corr={result['corr_eval']:.4f}"
            )
            if m == K:
                shapley_full_result = result
    else:
        # Needed to anchor random endpoint at full data
        shapley_full_result = _cached_evaluate(island_indices=sorted(list(range(K))), **eval_kwargs)

    if random_step_counts is not None:
        if len(random_step_counts) != K:
            raise ValueError(
                f"random_step_counts must have length {K}, got {len(random_step_counts)}"
            )
        mean_cum = np.asarray(random_step_counts, dtype=np.int64)

    # ---- Random individual baseline ----
    n_source_total = len(X_source)
    if include_random_individual and n_random_orders > 0 and n_workers > 1:
        logger.info(
            f"Submitting {n_random_orders * max(K - 1, 0)} random-individual add evaluations "
            f"across {n_workers} GPUs"
        )
        futures_info: List[Tuple] = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            task_idx = 0
            for r in range(n_random_orders):
                ind_rng = np.random.default_rng(seed + 500_000 + r)
                shuffled = ind_rng.permutation(n_source_total)
                for m in range(1, K):
                    n_include = min(int(mean_cum[m - 1]), n_source_total)
                    chosen = shuffled[:n_include].copy()
                    dev = devices[task_idx % n_workers]
                    future = pool.submit(
                        _evaluate_individual_subset,
                        individual_indices=chosen,
                        X_source=X_source, y_source=y_source,
                        X_test=X_test, y_test=y_test, y_eval_test=y_eval_test,
                        X_val=X_cal, y_val=y_cal,
                        train_params=train_params, device=dev, loss_fn=loss_fn,
                        cfg=cfg, seed=seed + r * 1000 + m,
                        batch_size=batch_size, snp_cols=snp_cols,
                    )
                    futures_info.append((future, m, n_include, r))
                    task_idx += 1

            done = 0
            total_tasks = len(futures_info)
            for future, m, n_include, r in futures_info:
                result = future.result()
                rows.append({
                    "n_islands": m, "n_individuals": n_include,
                    "corr_eval": result["corr_eval"], "mse_adj": result["mse_adj"],
                    "method": "random_individual", "order_seed": r,
                })
                done += 1
                if done % max(1, total_tasks // 5) == 0:
                    logger.info(
                        f"Random individual add-curve progress: {done}/{total_tasks}"
                    )

        # Endpoint (all data) uses Shapley result for direct comparability
        for r in range(n_random_orders):
            rows.append({
                "n_islands": K,
                "n_individuals": int(mean_cum[K - 1]),
                "corr_eval": float(shapley_full_result["corr_eval"]),
                "mse_adj": float(shapley_full_result["mse_adj"]),
                "method": "random_individual",
                "order_seed": r,
            })
        logger.info("Random individual add-curves complete (parallel)")
    elif include_random_individual and n_random_orders > 0:
        for r in range(n_random_orders):
            ind_rng = np.random.default_rng(seed + 500_000 + r)
            shuffled = ind_rng.permutation(n_source_total)
            for m in range(1, K):
                n_include = min(int(mean_cum[m - 1]), n_source_total)
                chosen = shuffled[:n_include]
                result = _evaluate_individual_subset(
                    individual_indices=chosen,
                    X_source=X_source, y_source=y_source,
                    X_test=X_test, y_test=y_test, y_eval_test=y_eval_test,
                    X_val=X_cal, y_val=y_cal,
                    train_params=train_params, device=device, loss_fn=loss_fn,
                    cfg=cfg, seed=seed + r * 1000 + m,
                    batch_size=batch_size, snp_cols=snp_cols,
                )
                rows.append({
                    "n_islands": m, "n_individuals": n_include,
                    "corr_eval": result["corr_eval"], "mse_adj": result["mse_adj"],
                    "method": "random_individual", "order_seed": r,
                })

            # Endpoint (all data) uses Shapley result for direct comparability
            rows.append({
                "n_islands": K,
                "n_individuals": int(mean_cum[K - 1]),
                "corr_eval": float(shapley_full_result["corr_eval"]),
                "mse_adj": float(shapley_full_result["mse_adj"]),
                "method": "random_individual",
                "order_seed": r,
            })
            logger.info(f"Random individual add-curve (seed={r}) complete")

    return pd.DataFrame(rows)


def mirror_add_to_remove_curve(add_curve_df: pd.DataFrame, n_source_islands: int) -> pd.DataFrame:
    """
    Create a remove-curve DataFrame by mirroring add-curve steps.

    n_islands = m  <->  n_removed = K - m
    """
    if add_curve_df.empty:
        return pd.DataFrame(
            columns=[
                "n_removed", "n_islands_remaining", "n_individuals_remaining",
                "corr_eval", "mse_adj", "method", "order_seed",
            ]
        )

    remove_df = add_curve_df.copy()
    remove_df["n_removed"] = int(n_source_islands) - remove_df["n_islands"].astype(int)
    remove_df["n_islands_remaining"] = remove_df["n_islands"].astype(int)
    remove_df["n_individuals_remaining"] = remove_df["n_individuals"].astype(int)

    core_cols = [
        "n_removed", "n_islands_remaining", "n_individuals_remaining",
        "corr_eval", "mse_adj", "method", "order_seed",
    ]
    extra_cols = [c for c in remove_df.columns if c not in core_cols]
    remove_df = remove_df[core_cols + extra_cols].reset_index(drop=True)
    return remove_df


def compute_remove_curve(
    phi: np.ndarray,
    source_codes: List[int],
    n_individuals: np.ndarray,
    X_source: np.ndarray,
    y_source: np.ndarray,
    locality_source: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    X_cal: Optional[np.ndarray],
    y_cal: Optional[np.ndarray],
    train_params: TrainParams,
    device: torch.device,
    cfg: ShapleyConfig,
    snp_cols: Optional[np.ndarray],
    n_random_orders: int = 20,
    seed: int = 42,
    batch_size: int = 64,
    devices: Optional[List[torch.device]] = None,
) -> pd.DataFrame:
    """
    Compute remove-island curves with two methods:

    - shapley_mean  : remove islands from lowest to highest phi/n
    - random_individual : remove random individuals (count matches shapley_mean steps)

    If *devices* contains multiple GPUs, the random-individual evaluations
    are distributed across them using a thread pool.

    Returns a single DataFrame with columns:
        n_removed, n_islands_remaining, n_individuals_remaining,
        corr_eval, mse_adj, method, order_seed
    """
    if devices is None:
        devices = [device]
    n_workers = len(devices) if len(devices) > 1 else 1

    K = len(source_codes)
    loss_fn = make_loss(train_params.loss_name)
    curve_cache = UtilityCache()  # fresh in-memory cache (test-set evaluations)
    n_total_individuals = int(np.sum(n_individuals))

    eval_kwargs = dict(
        source_codes=source_codes,
        X_source=X_source,
        y_source=y_source,
        locality_source=locality_source,
        X_test=X_test,
        y_test=y_test,
        y_eval_test=y_eval_test,
        X_val=X_cal,
        y_val=y_cal,
        train_params=train_params,
        device=device,
        loss_fn=loss_fn,
        cfg=cfg,
        seed=seed,
        cache=curve_cache,
        snp_cols=snp_cols,
        batch_size=batch_size,
    )

    rows: List[Dict[str, Any]] = []

    def _eval_island_removal(order: List[int], method: str, order_seed: int) -> None:
        """Evaluate removal curve for a given island removal order."""
        for n_removed in range(K):
            keep = sorted(set(range(K)) - set(order[:n_removed]))
            n_ind_remain = int(sum(n_individuals[k] for k in keep)) if keep else 0
            if not keep:
                rows.append({
                    "n_removed": n_removed, "n_islands_remaining": 0,
                    "n_individuals_remaining": 0,
                    "corr_eval": 0.0, "mse_adj": float("inf"),
                    "method": method, "order_seed": order_seed,
                })
                continue
            result = _cached_evaluate(island_indices=keep, **eval_kwargs)
            rows.append({
                "n_removed": n_removed, "n_islands_remaining": len(keep),
                "n_individuals_remaining": n_ind_remain,
                "corr_eval": result["corr_eval"], "mse_adj": result["mse_adj"],
                "method": method, "order_seed": order_seed,
            })

    # ---- Shapley mean order (remove lowest phi/n first) ----
    phi_per_n = phi / np.maximum(n_individuals.astype(np.float64), 1.0)
    mean_remove_order = np.argsort(phi_per_n).tolist()
    _eval_island_removal(mean_remove_order, "shapley_mean", -1)
    logger.info("Remove-curve (shapley_mean) complete")

    # Cumulative individuals removed at each step for shapley_mean
    mean_cum_removed = np.cumsum(
        [int(n_individuals[mean_remove_order[i]]) for i in range(K)]
    )

    # ---- Random individual removal ----
    n_source_total = len(X_source)
    all_indices = np.arange(n_source_total)
    if n_workers > 1:
        logger.info(
            f"Submitting {n_random_orders * K} random-individual remove evaluations "
            f"across {n_workers} GPUs"
        )
        futures_info: List[Tuple] = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            task_idx = 0
            for r in range(n_random_orders):
                ind_rng = np.random.default_rng(seed + 500_000 + r)
                remove_perm = ind_rng.permutation(n_source_total)

                for n_removed_step in range(K):
                    if n_removed_step == 0:
                        chosen = all_indices.copy()
                    else:
                        n_to_remove = min(
                            int(mean_cum_removed[n_removed_step - 1]),
                            n_source_total,
                        )
                        removed_set = set(remove_perm[:n_to_remove].tolist())
                        chosen = np.array(
                            [i for i in all_indices if i not in removed_set]
                        )

                    n_ind_remain = len(chosen)
                    if n_ind_remain < 2:
                        rows.append({
                            "n_removed": n_removed_step,
                            "n_islands_remaining": K - n_removed_step,
                            "n_individuals_remaining": n_ind_remain,
                            "corr_eval": 0.0, "mse_adj": float("inf"),
                            "method": "random_individual", "order_seed": r,
                        })
                        continue

                    dev = devices[task_idx % n_workers]
                    future = pool.submit(
                        _evaluate_individual_subset,
                        individual_indices=chosen,
                        X_source=X_source, y_source=y_source,
                        X_test=X_test, y_test=y_test, y_eval_test=y_eval_test,
                        X_val=X_cal, y_val=y_cal,
                        train_params=train_params, device=dev, loss_fn=loss_fn,
                        cfg=cfg, seed=seed + r * 1000 + n_removed_step,
                        batch_size=batch_size, snp_cols=snp_cols,
                    )
                    futures_info.append((future, n_removed_step, n_ind_remain, r))
                    task_idx += 1

            done = 0
            total_tasks = len(futures_info)
            for future, n_removed_step, n_ind_remain, r in futures_info:
                result = future.result()
                rows.append({
                    "n_removed": n_removed_step,
                    "n_islands_remaining": K - n_removed_step,
                    "n_individuals_remaining": n_ind_remain,
                    "corr_eval": result["corr_eval"],
                    "mse_adj": result["mse_adj"],
                    "method": "random_individual", "order_seed": r,
                })
                done += 1
                if done % max(1, total_tasks // 5) == 0:
                    logger.info(
                        f"Random individual remove-curve progress: {done}/{total_tasks}"
                    )

        logger.info("Random individual remove-curves complete (parallel)")
    else:
        for r in range(n_random_orders):
            ind_rng = np.random.default_rng(seed + 500_000 + r)
            remove_perm = ind_rng.permutation(n_source_total)

            for n_removed_step in range(K):
                if n_removed_step == 0:
                    chosen = all_indices
                else:
                    n_to_remove = min(
                        int(mean_cum_removed[n_removed_step - 1]), n_source_total
                    )
                    removed_set = set(remove_perm[:n_to_remove].tolist())
                    chosen = np.array([i for i in all_indices if i not in removed_set])

                n_ind_remain = len(chosen)
                if n_ind_remain < 2:
                    rows.append({
                        "n_removed": n_removed_step,
                        "n_islands_remaining": K - n_removed_step,
                        "n_individuals_remaining": n_ind_remain,
                        "corr_eval": 0.0, "mse_adj": float("inf"),
                        "method": "random_individual", "order_seed": r,
                    })
                    continue

                result = _evaluate_individual_subset(
                    individual_indices=chosen,
                    X_source=X_source, y_source=y_source,
                    X_test=X_test, y_test=y_test, y_eval_test=y_eval_test,
                    X_val=X_cal, y_val=y_cal,
                    train_params=train_params, device=device, loss_fn=loss_fn,
                    cfg=cfg, seed=seed + r * 1000 + n_removed_step,
                    batch_size=batch_size, snp_cols=snp_cols,
                )
                rows.append({
                    "n_removed": n_removed_step,
                    "n_islands_remaining": K - n_removed_step,
                    "n_individuals_remaining": n_ind_remain,
                    "corr_eval": result["corr_eval"], "mse_adj": result["mse_adj"],
                    "method": "random_individual", "order_seed": r,
                })
            logger.info(f"Random individual remove-curve (seed={r}) complete")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Saving and plotting
# ---------------------------------------------------------------------------

def save_shapley_results(
    phi: np.ndarray,
    source_codes: List[int],
    source_names: Dict[int, str],
    n_individuals: np.ndarray,
    target_island_code: int,
    target_island_name: str,
    v_full: float,
    tmc_stats: Dict[str, Any],
    add_curve_df: pd.DataFrame,
    remove_curve_df: pd.DataFrame,
    output_dir: str,
    cfg: ShapleyConfig,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tag = f"target_{target_island_code}"
    K = len(source_codes)

    # 1) Shapley values CSV
    phi_per_ind = phi / np.maximum(n_individuals.astype(np.float64), 1.0)
    rank_full = K - np.argsort(np.argsort(phi))  # rank 1 = highest
    rank_mean = K - np.argsort(np.argsort(phi_per_ind))

    phi_df = pd.DataFrame({
        "target_island": target_island_code,
        "source_island": source_codes,
        "source_island_name": [source_names.get(c, str(c)) for c in source_codes],
        "phi": phi,
        "phi_per_individual": phi_per_ind,
        "n_island": n_individuals,
        "rank_full": rank_full,
        "rank_mean": rank_mean,
    }).sort_values("rank_full")

    phi_path = os.path.join(output_dir, f"shapley_island_values_{tag}.csv")
    phi_df.to_csv(phi_path, index=False)
    logger.info(f"Saved Shapley values to {phi_path}")

    # 2) Summary JSON
    summary_stats = {k: v for k, v in tmc_stats.items() if k != "intermediate_rankings"}
    summary = {
        "target_island": target_island_code,
        "target_island_name": target_island_name,
        "v_full": v_full,
        **summary_stats,
        "note": "Shapley values computed on cal set; curves evaluated on held-out test set",
        "cap_per_island": cfg.cap_per_island,
        "n_train_seeds_per_eval": cfg.n_train_seeds_per_eval,
    }
    summary_path = os.path.join(output_dir, f"tmc_summary_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    # 2b) Intermediate ranking trace across permutations
    inter_rows = tmc_stats.get("intermediate_rankings")
    if inter_rows:
        inter_df = pd.DataFrame(inter_rows)
        if "source_island" in inter_df.columns:
            inter_df["source_island_name"] = inter_df["source_island"].map(
                lambda c: source_names.get(int(c), str(c))
            )
        inter_df["target_island"] = target_island_code
        inter_path = os.path.join(output_dir, f"tmc_intermediate_rankings_{tag}.csv")
        inter_df.to_csv(inter_path, index=False)
        logger.info(f"Saved intermediate ranking trace to {inter_path}")

    # 3) Add-curve CSV  (all methods in one file)
    add_curve_df = add_curve_df.copy()
    add_curve_df["target_island"] = target_island_code
    add_curve_path = os.path.join(output_dir, f"add_curve_{tag}.csv")
    add_curve_df.to_csv(add_curve_path, index=False)
    logger.info(f"Saved add-curve to {add_curve_path}")

    # 4) Remove-curve CSV  (all methods in one file)
    remove_curve_df = remove_curve_df.copy()
    remove_curve_df["target_island"] = target_island_code
    remove_curve_path = os.path.join(output_dir, f"remove_curve_{tag}.csv")
    remove_curve_df.to_csv(remove_curve_path, index=False)
    logger.info(f"Saved remove-curve to {remove_curve_path}")

    # 5) Plots
    try:
        _plot_shapley_bar(phi_df, target_island_name, output_dir, tag)
        _plot_add_curve(add_curve_df, target_island_name, output_dir, tag)
        _plot_remove_curve(remove_curve_df, target_island_name, output_dir, tag)
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")


def _plot_shapley_bar(
    phi_df: pd.DataFrame,
    target_island_name: str,
    output_dir: str,
    tag: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, label in [
        (axes[0], "phi", "Shapley value φ_k"),
        (axes[1], "phi_per_individual", "Shapley value per individual φ_k / n_k"),
    ]:
        sorted_df = phi_df.sort_values(col, ascending=True)
        colors = ["#4C78A8" if v >= 0 else "#E45756" for v in sorted_df[col]]
        ax.barh(sorted_df["source_island_name"], sorted_df[col], color=colors)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(label)
        ax.set_title(f"Target: {target_island_name}")
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = os.path.join(output_dir, f"shapley_bar_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved bar plot to {path}")


def _plot_add_curve(
    add_df: pd.DataFrame,
    target_island_name: str,
    output_dir: str,
    tag: str,
) -> None:
    import matplotlib.pyplot as plt

    _METHOD_STYLE = {
        "shapley_mean":      {"color": "#72B7B2", "marker": "s", "ls": "-",  "label": "Shapley (φ/n)"},
        "random_individual": {"color": "#E45756", "marker": None, "ls": ":",  "label": "Random individual (mean±SD)"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel in [
        (axes[0], "corr_eval", "Pearson r (pred adj vs orig pheno)"),
        (axes[1], "mse_adj", "MSE (pred adj vs actual adj)"),
    ]:
        # Random baselines (mean ± std)
        for method in ["random_individual"]:
            subset = add_df[add_df["method"] == method]
            if subset.empty:
                continue
            agg = subset.groupby("n_islands")[metric].agg(["mean", "std"]).reset_index()
            s = _METHOD_STYLE[method]
            ax.fill_between(
                agg["n_islands"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                alpha=0.15, color=s["color"],
            )
            ax.plot(agg["n_islands"], agg["mean"], color=s["color"], linestyle=s["ls"], label=s["label"])

        # Shapley curves
        for method in ["shapley_mean"]:
            subset = add_df[add_df["method"] == method]
            if subset.empty:
                continue
            s = _METHOD_STYLE[method]
            ax.plot(
                subset["n_islands"], subset[metric],
                color=s["color"], marker=s["marker"], markersize=5,
                linestyle=s["ls"], label=s["label"],
            )

        ax.set_xlabel("Number of islands added")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Add-islands curve — Target: {target_island_name}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, f"add_curve_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved add-curve plot to {path}")


def _plot_remove_curve(
    remove_df: pd.DataFrame,
    target_island_name: str,
    output_dir: str,
    tag: str,
) -> None:
    import matplotlib.pyplot as plt

    _METHOD_STYLE = {
        "shapley_mean":      {"color": "#72B7B2", "marker": "s", "ls": "-",  "label": "Shapley (φ/n)"},
        "random_individual": {"color": "#E45756", "marker": None, "ls": ":",  "label": "Random individual (mean±SD)"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel in [
        (axes[0], "corr_eval", "Pearson r (pred adj vs orig pheno)"),
        (axes[1], "mse_adj", "MSE (pred adj vs actual adj)"),
    ]:
        df_clean = remove_df.copy()
        df_clean[metric] = df_clean[metric].replace(float("inf"), float("nan"))

        for method in ["random_individual"]:
            subset = df_clean[df_clean["method"] == method]
            if subset.empty:
                continue
            agg = subset.groupby("n_removed")[metric].agg(["mean", "std"]).reset_index()
            s = _METHOD_STYLE[method]
            ax.fill_between(
                agg["n_removed"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                alpha=0.15, color=s["color"],
            )
            ax.plot(agg["n_removed"], agg["mean"], color=s["color"], linestyle=s["ls"], label=s["label"])

        for method in ["shapley_mean"]:
            subset = df_clean[df_clean["method"] == method]
            if subset.empty:
                continue
            s = _METHOD_STYLE[method]
            ax.plot(
                subset["n_removed"], subset[metric],
                color=s["color"], marker=s["marker"], markersize=5,
                linestyle=s["ls"], label=s["label"],
            )

        ax.set_xlabel("Number of islands removed (worst first)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Remove-islands curve — Target: {target_island_name}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, f"remove_curve_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved remove-curve plot to {path}")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_shapley_experiment(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    locality: np.ndarray,
    y_eval: np.ndarray,
    target_island_code: int,
    target_island_name: str,
    source_codes: List[int],
    source_names: Dict[int, str],
    train_params: Optional[TrainParams],
    cfg: ShapleyConfig,
    output_dir: str,
    device: torch.device,
    batch_size: int = 64,
    cal_fraction: float = 0.2,
    max_cal_fraction: Optional[float] = None,
    use_snp_selection: bool = False,
    num_snps: Optional[int] = None,
    snp_selection_mode: str = "random",
    compute_random_individual: bool = True,
    random_step_counts: Optional[np.ndarray] = None,
    cal_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
    snp_cols_override: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Orchestrate the full island-level Shapley experiment for one target island.

    Shapley values (phi) are computed using the target calibration set as the
    utility objective.  Add/remove curves are then evaluated on the held-out
    target test set so there is no selection leakage.

    Returns a dict with Shapley values and performance curves.
    """

    logger.info(
        f"[Shapley] Target island: {target_island_code} ({target_island_name}), "
        f"source islands: {len(source_codes)}"
    )

    # ---- Device detection ----------------------------------------------------
    devices = _detect_devices(device)

    # ---- Data split ----------------------------------------------------------
    split = split_target_island(
        X, y, ids, locality, target_island_code,
        y_eval=y_eval,
        cal_fraction=cal_fraction,
        seed=cfg.seed,
        max_cal_fraction=max_cal_fraction,
        cal_idx=cal_idx,
        test_idx=test_idx,
    )

    X_source = split["X_source"]
    y_source = split["y_source"]
    locality_source = split["locality_source"]

    # Cal set -> used as utility objective for Shapley computation
    X_cal = split["X_cal"]
    y_cal = split["y_cal"]
    y_eval_cal = split["y_eval_cal"]

    # Test set -> used only for reporting curves (never seen by Shapley)
    X_test = split["X_test"]
    y_test = split["y_test"]
    y_eval_test = split["y_eval_test"]

    # Filter source data to only included source islands
    in_source = np.isin(locality_source, source_codes)
    X_source = X_source[in_source]
    y_source = y_source[in_source]
    locality_source = locality_source[in_source]

    logger.info(
        f"Data: source={len(X_source)} (across {len(source_codes)} islands), "
        f"cal={len(X_cal)}, test={len(X_test)}"
    )

    # ---- SNP selection (on full source, deterministic) -----------------------
    snp_cols = None
    if snp_cols_override is not None:
        snp_cols = np.asarray(snp_cols_override, dtype=np.int64)
        logger.info(f"SNP selection: using fixed precomputed subset (n={len(snp_cols)})")
    elif use_snp_selection and num_snps is not None and num_snps < X_source.shape[1]:
        if snp_selection_mode.lower() == "random":
            rng = np.random.default_rng(cfg.seed)
            snp_cols = rng.choice(X_source.shape[1], size=int(num_snps), replace=False)
            logger.info(f"SNP selection: {num_snps} random SNPs (seed={cfg.seed})")
        else:
            from .utils import _select_top_snps_by_abs_corr
            snp_cols = _select_top_snps_by_abs_corr(X_source, y_source, num_snps)
            logger.info(f"SNP selection: top {num_snps} by correlation")

    # ---- Per-island sample counts --------------------------------------------
    n_individuals = np.array(
        [(locality_source == c).sum() for c in source_codes], dtype=np.int64
    )
    logger.info(
        "Island counts (capped to {cap} in training):\n{counts}".format(
            cap=cfg.cap_per_island,
            counts="\n".join(
                f"  {source_names.get(c, c)}: n={n}"
                for c, n in zip(source_codes, n_individuals)
            ),
        )
    )

    # ---- Utility cache (cal-based, for Shapley computation) ------------------
    cache_path = None
    if cfg.cache_dir:
        os.makedirs(cfg.cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cfg.cache_dir,
            f"utility_cache_target_{target_island_code}_seed_{cfg.seed}.json",
        )
        if cfg.permutation_state_path is None:
            cfg.permutation_state_path = os.path.join(
                cfg.cache_dir,
                f"tmc_perm_state_target_{target_island_code}_seed_{cfg.seed}.json",
            )
    cal_cache = UtilityCache(cache_path)

    # ---- Run TMC (utility = performance on cal set) --------------------------
    logger.info(f"Running TMC-Shapley with {cfg.n_permutations} permutations (utility on cal set)...")
    phi, v_full, tmc_stats = run_tmc_shapley(
        source_codes=source_codes,
        X_source=X_source,
        y_source=y_source,
        locality_source=locality_source,
        X_test=X_cal,
        y_test=y_cal,
        y_eval_test=y_eval_cal,
        train_params=train_params,
        device=device,
        cfg=cfg,
        cache=cal_cache,
        batch_size=batch_size,
        snp_cols=snp_cols,
        devices=devices,
    )

    logger.info("Shapley values (phi, computed on cal set):")
    for i, (code, name) in enumerate(zip(source_codes, [source_names.get(c, str(c)) for c in source_codes])):
        logger.info(f"  {name}: phi={phi[i]:.6f}, phi/n={phi[i]/max(n_individuals[i], 1):.8f}")

    # Curve retraining can use early stopping on calibration set
    curve_cfg = copy.deepcopy(cfg)
    curve_cfg.use_early_stopping = cfg.retrain_use_early_stopping
    curve_cfg.early_stopping_patience = cfg.retrain_early_stopping_patience
    logger.info(
        "Curve retraining early stopping: enabled=%s, patience=%d",
        curve_cfg.use_early_stopping,
        curve_cfg.early_stopping_patience,
    )

    # ---- Add curve (evaluated on held-out test set) --------------------------
    logger.info("Computing add-islands curve (evaluated on test set)...")
    add_curve_df = compute_add_curve(
        phi=phi,
        source_codes=source_codes,
        n_individuals=n_individuals,
        X_source=X_source,
        y_source=y_source,
        locality_source=locality_source,
        X_test=X_test,
        y_test=y_test,
        y_eval_test=y_eval_test,
        X_cal=X_cal,
        y_cal=y_cal,
        train_params=train_params,
        device=device,
        cfg=curve_cfg,
        snp_cols=snp_cols,
        n_random_orders=cfg.n_random_orders,
        seed=cfg.seed,
        batch_size=batch_size,
        devices=devices,
        include_shapley_mean=True,
        include_random_individual=compute_random_individual,
        random_step_counts=random_step_counts,
    )

    # ---- Remove curve (mirrored from add-curve) -----------------------------
    logger.info("Building mirrored remove-islands curve from add-islands curve...")
    remove_curve_df = mirror_add_to_remove_curve(add_curve_df, n_source_islands=len(source_codes))

    # ---- Save all results ----------------------------------------------------
    save_shapley_results(
        phi=phi,
        source_codes=source_codes,
        source_names=source_names,
        n_individuals=n_individuals,
        target_island_code=target_island_code,
        target_island_name=target_island_name,
        v_full=v_full,
        tmc_stats=tmc_stats,
        add_curve_df=add_curve_df,
        remove_curve_df=remove_curve_df,
        output_dir=output_dir,
        cfg=cfg,
    )

    return {
        "phi": phi,
        "source_codes": source_codes,
        "n_individuals": n_individuals,
        "v_full": v_full,
        "tmc_stats": tmc_stats,
        "add_curve_df": add_curve_df,
        "remove_curve_df": remove_curve_df,
    }
