"""
Removal curve experiment for TracIn-based training set optimization.

This module implements the main experiment:
1. Split target island into calibration and test sets
2. Train on source data with checkpoints
3. Compute TracIn scores
4. Generate removal curves (TracIn-guided vs random baseline)
"""
from __future__ import annotations
import gc
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .models import TrainParams, make_model
from .tracin import compute_tracin_scores, rank_by_influence
from .training import (
    evaluate_model,
    train_simple,
    train_with_checkpoints,
)
from .utils import _optimizer, make_loss, set_seed, _select_top_snps_by_abs_corr

logger = logging.getLogger(__name__)


@dataclass
class RemovalCurveResult:
    """Results from a single removal curve experiment."""
    target_island: int
    method: str  # "tracin" or "random"
    removal_fraction: float
    cal_fraction: float
    corr_eval: float  # Pearson r(predicted adjusted, original phenotype)
    mse_adj: float    # MSE(predicted adjusted, actual adjusted phenotype)
    n_train_remaining: int
    seed: int
    epochs_trained: int


@dataclass
class TracInExperimentResults:
    """Complete results from a TracIn experiment for one target island."""
    target_island: int
    target_island_name: str
    n_source_train: int
    n_target_cal: int
    n_target_test: int
    
    # Per-individual scores by seed
    scores_by_seed: Dict[int, pd.DataFrame]  # columns: ringnr, island, score, rank
    
    # Removal curves
    removal_curves: List[RemovalCurveResult] = field(default_factory=list)
    
    # TracIn score correlations across seeds
    score_corr: Optional[pd.DataFrame] = None
    
    # Config used
    config: Dict[str, Any] = field(default_factory=dict)


def split_target_island(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    locality: np.ndarray,
    target_island_code: int,
    y_eval: Optional[np.ndarray] = None,
    cal_fraction: float = 0.2,
    seed: int = 42,
    max_cal_fraction: Optional[float] = None,
    cal_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
    included_island_codes: Optional[List[int]] = None,
):
    """
    Split data into source (all non-target islands) and target (cal + test).
    
    Parameters
    ----------
    X, y, ids, locality : np.ndarray
        Full dataset (y is adjusted phenotype for training)
    target_island_code : int
        Code of the target island to hold out
    y_eval : np.ndarray, optional
        Original phenotypes for evaluation (y_mean)
    cal_fraction : float
        Fraction of target island to use for calibration (rest is test)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict with keys:
        X_source, y_source, ids_source, locality_source : source training data
        X_cal, y_cal, ids_cal, locality_cal : target calibration data
        X_test, y_test, ids_test, locality_test : target test data
        y_eval_source, y_eval_cal, y_eval_test : original phenotypes (if provided)
    """
    np.random.seed(seed)
    
    # If y_eval not provided, fall back to y
    if y_eval is None:
        y_eval = y.copy()
    
    # Separate source and target
    target_mask = (locality == target_island_code)
    source_mask = ~target_mask

    # Optional filtering of source islands used for training
    if included_island_codes is not None:
        included_set = set(int(v) for v in included_island_codes)
        source_mask = source_mask & np.isin(locality, list(included_set))
    
    X_source = X[source_mask]
    y_source = y[source_mask]
    ids_source = ids[source_mask]
    locality_source = locality[source_mask]
    y_eval_source = y_eval[source_mask]
    
    X_target = X[target_mask]
    y_target = y[target_mask]
    ids_target = ids[target_mask]
    locality_target = locality[target_mask]
    y_eval_target = y_eval[target_mask]
    
    # Split target into calibration and test
    n_target = len(X_target)
    n_cal = max(1, int(cal_fraction * n_target))
    
    if cal_idx is None or test_idx is None:
        # Random shuffle for split
        perm = np.random.permutation(n_target)
        if max_cal_fraction is not None:
            n_cal_max = max(1, int(max_cal_fraction * n_target))
            if n_cal > n_cal_max:
                raise ValueError(
                    f"cal_fraction={cal_fraction} exceeds max_cal_fraction={max_cal_fraction}"
                )
            cal_idx = perm[:n_cal]
            test_idx = perm[n_cal_max:]
        else:
            cal_idx = perm[:n_cal]
            test_idx = perm[n_cal:]
    else:
        cal_idx = np.asarray(cal_idx)
        test_idx = np.asarray(test_idx)
    
    X_cal = X_target[cal_idx]
    y_cal = y_target[cal_idx]
    ids_cal = ids_target[cal_idx]
    locality_cal = locality_target[cal_idx]
    y_eval_cal = y_eval_target[cal_idx]
    
    X_test = X_target[test_idx]
    y_test = y_target[test_idx]
    ids_test = ids_target[test_idx]
    locality_test = locality_target[test_idx]
    y_eval_test = y_eval_target[test_idx]
    
    return {
        "X_source": X_source, "y_source": y_source, "ids_source": ids_source, 
        "locality_source": locality_source, "y_eval_source": y_eval_source,
        "X_cal": X_cal, "y_cal": y_cal, "ids_cal": ids_cal,
        "locality_cal": locality_cal, "y_eval_cal": y_eval_cal,
        "X_test": X_test, "y_test": y_test, "ids_test": ids_test,
        "locality_test": locality_test, "y_eval_test": y_eval_test,
    }


def run_removal_curve_experiment(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    locality: np.ndarray,
    target_island_code: int,
    train_params: TrainParams,
    y_eval: Optional[np.ndarray] = None,
    code_to_label: Optional[Dict[int, str]] = None,
    cal_fraction: float = 0.2,
    experiment_mode: str = "removal_curve",
    removal_fractions: List[float] = None,
    n_random_seeds: int = 3,
    tracin_removal_mode: str = "lowest",
    iterative_removal_fraction: float = 0.05,
    iterative_n_iterations: int = 5,
    n_checkpoints: int = 5,
    tracin_mode: str = "lastlayer",
    batch_size: int = 64,
    seed: int = 42,
    max_cal_fraction: Optional[float] = None,
    output_dir: Optional[str] = None,
    use_snp_selection: bool = False,
    num_snps: Optional[int] = None,
    device: Optional[torch.device] = None,
    epoch_scaling: str = "none",
    early_stopping_patience: int = 0,
    snp_selection_mode: str = "corr",
    save_loss_curves: bool = False,
    save_retrain_loss_curves: bool = False,
    use_early_stopping: bool = True,
    retrain_epochs: Optional[int] = None,
    included_island_codes: Optional[List[int]] = None,
    checkpoint_training_every_iteration: bool = False,
    # iterative_addition parameters
    addition_initial_fraction: float = 0.2,
    addition_fraction_per_step: float = 0.05,
    addition_n_steps: int = 10,
) -> TracInExperimentResults:
    """
    Run the complete TracIn experiment for one target island.
    
    Four experiment modes are supported:
    1. "removal_curve": Test different removal fractions independently
    2. "weighted_sampling": Use TracIn scores as sample weights (no removal)
    3. "iterative_removal": Iteratively remove worst samples and recompute scores
    4. "iterative_addition": Start with small random subset, add best-scored samples
    
    For iterative_addition:
      - Train on ALL source data with checkpoints and compute TracIn scores once
      - Start with a random initial_fraction of source data (same for TracIn + random)
      - Each step, add addition_fraction_per_step of (original N) highest-scored
        remaining samples (TracIn) or random remaining samples (random baseline)
      - Retrain and evaluate at each step
    
    Parameters
    ----------
    X : np.ndarray
        SNP feature matrix
    y : np.ndarray
        Adjusted phenotypes (training target)
    ids : np.ndarray
        Sample IDs
    locality : np.ndarray
        Island codes for each sample
    target_island_code : int
        Target island to evaluate on
    train_params : TrainParams
        Hyperparameters for training
    y_eval : np.ndarray, optional
        Original phenotypes for evaluation (if None, uses y)
    code_to_label : Dict[int, str], optional
        Mapping from island codes to names
    cal_fraction : float
        Fraction of target island for calibration
    experiment_mode : str
        One of: "removal_curve", "weighted_sampling", "iterative_removal"
    removal_fractions : List[float]
        Fractions to test (for removal_curve mode)
    n_random_seeds : int
        Number of random baseline seeds
    tracin_removal_mode : str
        "lowest" or "highest" - which samples to remove
    iterative_removal_fraction : float
        Fraction to remove per iteration (for iterative_removal mode)
    iterative_n_iterations : int
        Number of iterations (for iterative_removal mode)
    checkpoint_training_every_iteration : bool
        If True in iterative_removal mode, retrain with checkpoints and recompute
        TracIn scores from checkpoints at every iteration.
    n_checkpoints : int
        Number of checkpoints for TracIn
    tracin_mode : str
        "full" or "lastlayer"
    batch_size : int
        Training batch size
    seed : int
        Main random seed
    output_dir : str, optional
        Directory to save results
    use_snp_selection : bool
        Whether to use SNP feature selection
    num_snps : int, optional
        Number of SNPs to select (if use_snp_selection)
    device : torch.device, optional
        Device for computation
    retrain_epochs : int, optional
        Number of epochs for retraining after pruning (if None, uses train_params.epochs)
    included_island_codes : List[int], optional
        Encoded island codes to include in source/training data. If None, includes all non-target islands.
        
    Returns
    -------
    results : TracInExperimentResults
        Complete experiment results
    """
    if removal_fractions is None:
        removal_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Use retrain_epochs if specified, otherwise fall back to train_params.epochs
    if retrain_epochs is None:
        retrain_epochs = train_params.epochs
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # If y_eval not provided, fall back to y
    if y_eval is None:
        y_eval = y.copy()
    
    set_seed(seed)
    
    target_name = code_to_label.get(target_island_code, str(target_island_code)) if code_to_label else str(target_island_code)
    logger.info(f"Running TracIn experiment for target island: {target_island_code} ({target_name})")
    
    # ========================================
    # Step 1: Split data
    # ========================================
    split = split_target_island(
        X, y, ids, locality, target_island_code, y_eval=y_eval, 
        cal_fraction=cal_fraction, seed=seed, max_cal_fraction=max_cal_fraction,
        included_island_codes=included_island_codes,
    )
    
    X_source = split["X_source"]
    y_source = split["y_source"]
    ids_source = split["ids_source"]
    locality_source = split["locality_source"]
    
    X_cal = split["X_cal"]
    y_cal = split["y_cal"]
    
    X_test = split["X_test"]
    y_test = split["y_test"]
    y_eval_test = split["y_eval_test"]

    if len(X_source) == 0:
        raise ValueError(
            f"No source samples left after filtering for target island {target_island_code}. "
            f"Check included_island_codes configuration."
        )
    
    logger.info(f"Data split: source={len(X_source)}, cal={len(X_cal)}, test={len(X_test)}")
    
    # ========================================
    # Step 2: Feature selection (fit on source only, deterministic across all training seeds)
    # ========================================
    # SNP selection is done ONCE here using the main seed.
    # ALL training seeds will use the SAME SNP indices (eliminates SNP selection randomness).
    snp_cols = None
    if use_snp_selection and num_snps is not None and num_snps < X_source.shape[1]:
        if (snp_selection_mode or "corr").lower() == "random":
            logger.info(f"Selecting {num_snps} random SNPs (deterministic, seed={seed})")
            rng = np.random.default_rng(seed)
            snp_cols = rng.choice(X_source.shape[1], size=int(num_snps), replace=False)
        else:
            logger.info(f"Selecting top {num_snps} SNPs by correlation")
            snp_cols = _select_top_snps_by_abs_corr(X_source, y_source, num_snps)
        snp_indices_hash = hash(tuple(snp_cols.tolist())) if snp_cols is not None else None
        logger.info(f"SNP selection complete: n_snps={len(snp_cols)}, hash={snp_indices_hash}")
        logger.info(f"These SNP indices will be REUSED for all {n_random_seeds} training seeds.")
        X_source = X_source[:, snp_cols]
        X_cal = X_cal[:, snp_cols]
        X_test = X_test[:, snp_cols]
    
    in_dim = X_source.shape[1]
    logger.info(f"Input dimension after feature selection: {in_dim}")
    
    # ========================================
    # Step 3: Train with checkpoints and compute TracIn scores per seed
    # ========================================
    x_source_t = torch.from_numpy(X_source).float()
    y_source_t = torch.from_numpy(y_source).float()
    x_cal_t = torch.from_numpy(X_cal).float().to(device)
    y_cal_t = torch.from_numpy(y_cal).float().to(device)
    x_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_adj_t = torch.from_numpy(y_test).float()  # Adjusted phenotype
    y_test_eval_t = torch.from_numpy(y_eval_test).float()  # Original phenotype
    
    loss_fn = make_loss(train_params.loss_name)
    scores_by_seed: Dict[int, pd.DataFrame] = {}
    removal_results: List[RemovalCurveResult] = []

    def _select_best_checkpoint_by_val_loss(
        checkpoints: List[Any],
        model_for_eval: torch.nn.Module,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> Tuple[Any, float]:
        """Select checkpoint with lowest validation loss on calibration set."""
        if len(checkpoints) == 0:
            raise ValueError("No checkpoints available for validation-based selection")

        best_ckpt = checkpoints[0]
        best_val_loss = float("inf")

        model_for_eval.to(device)
        model_for_eval.eval()
        with torch.no_grad():
            for ckpt in checkpoints:
                model_for_eval.load_state_dict(ckpt.state_dict)
                preds = model_for_eval(x_val)
                val_loss = float(loss_fn(preds.view(-1), y_val.view(-1)).item())
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt = ckpt

        return best_ckpt, best_val_loss
    
    logger.info("Computing TracIn scores and removal curves...")
    
    for tracin_seed in range(seed, seed + n_random_seeds):
        logger.info(f"Training model with checkpoints (seed={tracin_seed})...")
        
        set_seed(tracin_seed)
        model = make_model(in_dim, train_params)
        optimizer = _optimizer(
            train_params.optimizer,
            model.parameters(),
            train_params.lr,
            train_params.weight_decay
        )
        
        loss_plot_path = None
        if output_dir and save_loss_curves:
            cal_tag = _format_cal_fraction(cal_fraction)
            loss_plot_path = os.path.join(
                output_dir,
                f"loss_curve_island_{target_island_code}_cal_{cal_tag}_seed_{tracin_seed}.png",
            )

        model, checkpoints = train_with_checkpoints(
            model=model,
            x_train=x_source_t,
            y_train=y_source_t,
            epochs=train_params.epochs,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            n_checkpoints=n_checkpoints,
            batch_size=batch_size,
            seed=tracin_seed,
            x_val=x_cal_t,
            y_val=y_cal_t,
            x_test=x_test_t,
            y_test=y_test_adj_t,
            loss_plot_path=loss_plot_path,
        )
        logger.info(f"Training complete, saved {len(checkpoints)} checkpoints")
        
        # Compute TracIn scores for this seed
        logger.info(f"Computing TracIn scores (mode={tracin_mode}, seed={tracin_seed})...")
        
        x_source_t_dev = x_source_t.to(device)
        y_source_t_dev = y_source_t.to(device)
        scores = compute_tracin_scores(
            checkpoints=checkpoints,
            model=model,
            x_train=x_source_t_dev,
            y_train=y_source_t_dev,
            x_cal=x_cal_t,
            y_cal=y_cal_t,
            loss_fn=loss_fn,
            device=device,
            mode=tracin_mode,
        )
        
        rank_order = rank_by_influence(scores, ids_source)
        ranks = np.zeros(len(scores), dtype=int)
        for i, idx in enumerate(rank_order):
            ranks[idx] = i
        
        scores_df = pd.DataFrame({
            "ringnr": ids_source,
            "island": locality_source,
            "score": scores,
            "rank": ranks,
        }).sort_values("rank")
        
        scores_by_seed[tracin_seed] = scores_df
        logger.info(f"Scores computed (seed={tracin_seed}). Range: [{scores.min():.6f}, {scores.max():.6f}]")
        
        # ========================================
        # Step 4: Apply experiment mode
        # ========================================
        if experiment_mode == "weighted_sampling":
            # Mode 1: Weighted sampling (use TracIn scores as sample weights)
            logger.info(f"Using TracIn-weighted sampling (seed={tracin_seed})")
            # Clamp negative scores at 0
            weights = np.maximum(scores, 0.0)
            # Normalize so weights sum to n_samples (for proper weighted sampling)
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights * len(scores) / weight_sum
            else:
                # All scores negative/zero: uniform weights
                weights = np.ones(len(scores))
            logger.info(f"Weights: min={weights.min():.6f}, max={weights.max():.6f}, mean={weights.mean():.6f}")
            
            # Train with weighted sampling
            x_remaining_t = x_source_t
            y_remaining_t = y_source_t
            
            retrain_loss_plot_path = None
            if output_dir and save_retrain_loss_curves:
                cal_tag = _format_cal_fraction(cal_fraction)
                retrain_loss_plot_path = os.path.join(
                    output_dir,
                    f"loss_curve_weighted_tracin_island_{target_island_code}_cal_{cal_tag}_seed_{tracin_seed}.png",
                )
            
            retrain_model = make_model(in_dim, train_params)
            retrain_opt = _optimizer(
                train_params.optimizer,
                retrain_model.parameters(),
                train_params.lr,
                train_params.weight_decay
            )
            
            retrain_model, epochs_trained = train_simple(
                model=retrain_model,
                x_train=x_remaining_t,
                y_train=y_remaining_t,
                epochs=retrain_epochs,
                optimizer=retrain_opt,
                loss_fn=loss_fn,
                device=device,
                batch_size=batch_size,
                seed=tracin_seed,
                x_val=x_cal_t,
                y_val=y_cal_t,
                x_test=x_test_t,
                y_test=y_test_adj_t,
                patience=early_stopping_patience if use_early_stopping else 0,
                loss_plot_path=retrain_loss_plot_path,
                sample_weights=weights,
            )
            
            corr_eval, mse_adj = evaluate_model(
                retrain_model, x_test_t, y_test_adj_t, device, y_eval=y_test_eval_t
            )
            
            result = RemovalCurveResult(
                target_island=target_island_code,
                method="tracin_weighted",
                removal_fraction=0.0,  # Not applicable for weighted sampling
                cal_fraction=cal_fraction,
                corr_eval=corr_eval,
                mse_adj=mse_adj,
                n_train_remaining=len(X_source),  # All samples used with different weights
                seed=tracin_seed,
                epochs_trained=epochs_trained,
            )
            removal_results.append(result)
            logger.info(
                f"TracIn weighted sampling (seed={tracin_seed}): "
                f"corr_eval={corr_eval:.4f}, mse_adj={mse_adj:.6f}"
            )
            del retrain_model, x_remaining_t, y_remaining_t
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        elif experiment_mode == "iterative_removal":
            # Mode 2: Iterative removal (remove worst, recompute scores, repeat)
            logger.info(f"Using iterative removal (seed={tracin_seed})")
            logger.info(
                "Iterative mode uses fixed epochs with checkpoint training every iteration; "
                "best checkpoint selected by calibration loss for test evaluation"
            )
            
            # Start with full dataset
            X_current = X_source.copy()
            y_current = y_source.copy()
            ids_current = ids_source.copy()
            current_indices = np.arange(len(X_source))  # Track original indices
            n_source_initial = len(X_source)
            
            for iteration in range(iterative_n_iterations):
                logger.info(f"Iteration {iteration + 1}/{iterative_n_iterations}, n_samples={len(X_current)}")
                removed_fraction = min(iteration * iterative_removal_fraction, 1.0)
                
                # Retrain on current dataset
                set_seed(tracin_seed + iteration * 1000)  # Different seed per iteration
                x_current_t = torch.from_numpy(X_current).float()
                y_current_t = torch.from_numpy(y_current).float()
                
                retrain_model = make_model(in_dim, train_params)
                retrain_opt = _optimizer(
                    train_params.optimizer,
                    retrain_model.parameters(),
                    train_params.lr,
                    train_params.weight_decay
                )

                iter_epochs = train_params.epochs
                retrain_loss_plot_path = None
                if output_dir and save_retrain_loss_curves:
                    cal_tag = _format_cal_fraction(cal_fraction)
                    retrain_loss_plot_path = os.path.join(
                        output_dir,
                        f"loss_curve_iterative_island_{target_island_code}_cal_{cal_tag}_seed_{tracin_seed}_iter_{iteration + 1:02d}.png",
                    )

                retrain_model, iter_checkpoints = train_with_checkpoints(
                    model=retrain_model,
                    x_train=x_current_t,
                    y_train=y_current_t,
                    epochs=iter_epochs,
                    optimizer=retrain_opt,
                    loss_fn=loss_fn,
                    device=device,
                    n_checkpoints=n_checkpoints,
                    batch_size=batch_size,
                    seed=tracin_seed + iteration * 1000,
                    x_val=x_cal_t,
                    y_val=y_cal_t,
                    x_test=x_test_t,
                    y_test=y_test_adj_t,
                    loss_plot_path=retrain_loss_plot_path,
                )
                epochs_trained = iter_epochs

                best_ckpt, best_val_loss = _select_best_checkpoint_by_val_loss(
                    iter_checkpoints, retrain_model, x_cal_t, y_cal_t
                )
                retrain_model.load_state_dict(best_ckpt.state_dict)
                logger.info(
                    f"Iteration {iteration + 1}: selected checkpoint epoch {best_ckpt.epoch} "
                    f"with calibration loss {best_val_loss:.6f}"
                )
                
                # Evaluate on test set
                corr_eval, mse_adj = evaluate_model(
                    retrain_model, x_test_t, y_test_adj_t, device, y_eval=y_test_eval_t
                )
                
                # Store result for this iteration
                result = RemovalCurveResult(
                    target_island=target_island_code,
                    method="tracin_iterative",
                    removal_fraction=removed_fraction,
                    cal_fraction=cal_fraction,
                    corr_eval=corr_eval,
                    mse_adj=mse_adj,
                    n_train_remaining=len(X_current),
                    seed=tracin_seed,
                    epochs_trained=epochs_trained,
                )
                removal_results.append(result)
                logger.info(
                    f"Iteration {iteration + 1} (seed={tracin_seed}): "
                    f"corr_eval={corr_eval:.4f}, mse_adj={mse_adj:.6f} (n={len(X_current)})"
                )
                
                # If not the last iteration, recompute scores and remove worst samples
                if iteration < iterative_n_iterations - 1:
                    # Recompute TracIn scores on current dataset
                    logger.info(f"Recomputing TracIn scores for iteration {iteration + 2}...")

                    x_current_t_dev = x_current_t.to(device)
                    y_current_t_dev = y_current_t.to(device)
                    current_scores = compute_tracin_scores(
                        checkpoints=iter_checkpoints,
                        model=retrain_model,
                        x_train=x_current_t_dev,
                        y_train=y_current_t_dev,
                        x_cal=x_cal_t,
                        y_cal=y_cal_t,
                        loss_fn=loss_fn,
                        device=device,
                        mode=tracin_mode,
                    )
                    
                    # Determine how many samples to remove to match target cumulative
                    # removal relative to ORIGINAL source size (not compounding).
                    target_removed_next = min((iteration + 1) * iterative_removal_fraction, 1.0)
                    target_remaining_next = max(1, int(round((1.0 - target_removed_next) * n_source_initial)))
                    n_remove = len(X_current) - target_remaining_next
                    if n_remove <= 0:
                        logger.info("No additional removal needed to reach next target fraction")
                        continue
                    
                    # Rank samples by score
                    current_rank_order = rank_by_influence(current_scores, np.arange(len(X_current)))
                    
                    # Keep samples based on removal mode
                    if (tracin_removal_mode or "lowest").lower() == "highest":
                        keep_indices_local = current_rank_order[:-n_remove] if n_remove > 0 else current_rank_order
                    else:
                        keep_indices_local = current_rank_order[n_remove:]
                    
                    # Update dataset
                    X_current = X_current[keep_indices_local]
                    y_current = y_current[keep_indices_local]
                    ids_current = ids_current[keep_indices_local]
                    current_indices = current_indices[keep_indices_local]
                    
                    logger.info(f"Removed {n_remove} samples, {len(X_current)} remaining")
                
                del retrain_model, x_current_t, y_current_t
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        elif experiment_mode == "removal_curve":
            # Mode 3: Removal curve (test different removal fractions independently)
            logger.info(f"TracIn-guided removal curve with seed={tracin_seed}")
            for frac in removal_fractions:
                n_remove = int(frac * len(X_source))
                n_keep = len(X_source) - n_remove
                
                if n_keep < 10:  # Safety check
                    logger.warning(f"Skipping fraction {frac}: would keep only {n_keep} samples")
                    continue
                
                if (tracin_removal_mode or "lowest").lower() == "highest":
                    keep_indices = rank_order[:-n_remove] if n_remove > 0 else rank_order
                else:
                    keep_indices = rank_order[n_remove:]
                X_remaining = X_source[keep_indices]
                y_remaining = y_source[keep_indices]
                
                # Retrain with this seed
                set_seed(tracin_seed)
                x_remaining_t = torch.from_numpy(X_remaining).float()
                y_remaining_t = torch.from_numpy(y_remaining).float()
                
                retrain_model = make_model(in_dim, train_params)
                retrain_opt = _optimizer(
                    train_params.optimizer,
                    retrain_model.parameters(),
                    train_params.lr,
                    train_params.weight_decay
                )
                
                retrain_epochs_scaled = _scale_epochs(retrain_epochs, len(X_source), n_keep, epoch_scaling)
                retrain_loss_plot_path = None
                if output_dir and save_retrain_loss_curves:
                    cal_tag = _format_cal_fraction(cal_fraction)
                    frac_tag = int(round(float(frac) * 100))
                    retrain_loss_plot_path = os.path.join(
                        output_dir,
                        f"loss_curve_retrain_tracin_island_{target_island_code}_cal_{cal_tag}_seed_{tracin_seed}_rm_{frac_tag:02d}.png",
                    )

                retrain_model, epochs_trained = train_simple(
                    model=retrain_model,
                    x_train=x_remaining_t,
                    y_train=y_remaining_t,
                    epochs=retrain_epochs_scaled,
                    optimizer=retrain_opt,
                    loss_fn=loss_fn,
                    device=device,
                    batch_size=batch_size,
                    seed=tracin_seed,
                    x_val=x_cal_t,
                    y_val=y_cal_t,
                    x_test=x_test_t,
                    y_test=y_test_adj_t,
                    patience=early_stopping_patience if use_early_stopping else 0,
                    loss_plot_path=retrain_loss_plot_path,
                )
                
                # Evaluate on test set: corr vs original, MSE vs adjusted
                corr_eval, mse_adj = evaluate_model(
                    retrain_model, x_test_t, y_test_adj_t, device, y_eval=y_test_eval_t
                )
                
                result = RemovalCurveResult(
                    target_island=target_island_code,
                    method="tracin",
                    removal_fraction=frac,
                    cal_fraction=cal_fraction,
                    corr_eval=corr_eval,
                    mse_adj=mse_adj,
                    n_train_remaining=n_keep,
                    seed=tracin_seed,
                    epochs_trained=epochs_trained,
                )
                removal_results.append(result)
                
                logger.info(
                    f"TracIn removal {frac:.0%} (seed={tracin_seed}): "
                    f"corr_eval={corr_eval:.4f}, mse_adj={mse_adj:.6f} (n={n_keep})"
                )
                
                del retrain_model, x_remaining_t, y_remaining_t
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        elif experiment_mode == "iterative_addition":
            # Mode 4: Iterative addition – start small, add best-scored samples
            logger.info(f"Using iterative addition (seed={tracin_seed})")
            
            n_source = len(X_source)
            n_initial = max(1, int(round(addition_initial_fraction * n_source)))
            
            # Sort all source samples by TracIn score (highest = most beneficial)
            sorted_indices = np.argsort(-scores)  # descending by score
            
            # Deterministic random initial subset (same for TracIn and random,
            # seed depends only on tracin_seed so it's shared)
            rng_init = np.random.default_rng(tracin_seed)
            initial_indices = rng_init.choice(n_source, size=n_initial, replace=False)
            initial_set = set(initial_indices.tolist())
            
            # Remaining indices ordered by TracIn score (highest first)
            remaining_by_score = [idx for idx in sorted_indices if idx not in initial_set]
            
            current_indices = list(initial_indices)
            pointer = 0  # how far we've consumed from remaining_by_score
            
            for step in range(addition_n_steps + 1):
                added_fraction = addition_initial_fraction + step * addition_fraction_per_step
                added_fraction = min(added_fraction, 1.0)
                
                logger.info(
                    f"Addition step {step}/{addition_n_steps} "
                    f"(fraction={added_fraction:.0%}, n={len(current_indices)})"
                )
                
                # Retrain on current subset
                X_sub = X_source[current_indices]
                y_sub = y_source[current_indices]
                
                set_seed(tracin_seed + step * 1000)
                x_sub_t = torch.from_numpy(X_sub).float()
                y_sub_t = torch.from_numpy(y_sub).float()
                
                retrain_model = make_model(in_dim, train_params)
                retrain_opt = _optimizer(
                    train_params.optimizer,
                    retrain_model.parameters(),
                    train_params.lr,
                    train_params.weight_decay,
                )
                
                retrain_loss_plot_path = None
                if output_dir and save_retrain_loss_curves:
                    cal_tag = _format_cal_fraction(cal_fraction)
                    retrain_loss_plot_path = os.path.join(
                        output_dir,
                        f"loss_curve_addition_tracin_island_{target_island_code}_cal_{cal_tag}_seed_{tracin_seed}_step_{step:02d}.png",
                    )
                
                retrain_model, epochs_trained_step = train_simple(
                    model=retrain_model,
                    x_train=x_sub_t,
                    y_train=y_sub_t,
                    epochs=train_params.epochs,
                    optimizer=retrain_opt,
                    loss_fn=loss_fn,
                    device=device,
                    batch_size=batch_size,
                    seed=tracin_seed + step * 1000,
                    x_val=x_cal_t,
                    y_val=y_cal_t,
                    x_test=x_test_t,
                    y_test=y_test_adj_t,
                    patience=0,
                    loss_plot_path=retrain_loss_plot_path,
                    restore_best=True,
                )
                
                corr_eval, mse_adj = evaluate_model(
                    retrain_model, x_test_t, y_test_adj_t, device, y_eval=y_test_eval_t
                )
                
                result = RemovalCurveResult(
                    target_island=target_island_code,
                    method="tracin_addition",
                    removal_fraction=added_fraction,  # repurpose field as "fraction included"
                    cal_fraction=cal_fraction,
                    corr_eval=corr_eval,
                    mse_adj=mse_adj,
                    n_train_remaining=len(current_indices),
                    seed=tracin_seed,
                    epochs_trained=train_params.epochs,
                )
                removal_results.append(result)
                logger.info(
                    f"TracIn addition step {step} (seed={tracin_seed}): "
                    f"frac={added_fraction:.0%}, n={len(current_indices)}, "
                    f"corr_eval={corr_eval:.4f}, mse_adj={mse_adj:.6f}"
                )
                
                del retrain_model, x_sub_t, y_sub_t
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Add next batch of highest-scored samples for next step
                if step < addition_n_steps and added_fraction < 1.0:
                    target_n = min(n_source, int(round((added_fraction + addition_fraction_per_step) * n_source)))
                    n_to_add = target_n - len(current_indices)
                    if n_to_add > 0:
                        new_indices = remaining_by_score[pointer:pointer + n_to_add]
                        pointer += n_to_add
                        current_indices.extend(new_indices)
                        logger.info(f"Added {len(new_indices)} samples by TracIn score, total {len(current_indices)}")

        else:
            raise ValueError(f"Unknown experiment_mode: {experiment_mode}")
    
    # ========================================
    # Step 5: Random removal baseline (for removal_curve and iterative_removal modes)
    # ========================================
    if experiment_mode in ["removal_curve", "iterative_removal", "iterative_addition"]:
        logger.info(f"Running random baseline for {experiment_mode} mode...")
        
        if experiment_mode == "removal_curve":
            # Random baseline with different removal fractions
            for rand_seed in range(seed, seed + n_random_seeds):
                np.random.seed(rand_seed)
                random_order = np.random.permutation(len(X_source))
                
                for frac in removal_fractions:
                    n_remove = int(frac * len(X_source))
                    n_keep = len(X_source) - n_remove
                    
                    if n_keep < 10:
                        continue
                    
                    keep_indices = random_order[n_remove:]
                    
                    X_remaining = X_source[keep_indices]
                    y_remaining = y_source[keep_indices]
                    
                    x_remaining_t = torch.from_numpy(X_remaining).float()
                    y_remaining_t = torch.from_numpy(y_remaining).float()
                    
                    retrain_model = make_model(in_dim, train_params)
                    retrain_opt = _optimizer(
                        train_params.optimizer,
                        retrain_model.parameters(),
                        train_params.lr,
                        train_params.weight_decay
                    )
                    
                    retrain_epochs_scaled = _scale_epochs(retrain_epochs, len(X_source), n_keep, epoch_scaling)
                    retrain_loss_plot_path = None
                    if output_dir and save_retrain_loss_curves:
                        cal_tag = _format_cal_fraction(cal_fraction)
                        frac_tag = int(round(float(frac) * 100))
                        retrain_loss_plot_path = os.path.join(
                            output_dir,
                            f"loss_curve_retrain_random_island_{target_island_code}_cal_{cal_tag}_seed_{rand_seed}_rm_{frac_tag:02d}.png",
                        )

                    retrain_model, epochs_trained = train_simple(
                        model=retrain_model,
                        x_train=x_remaining_t,
                        y_train=y_remaining_t,
                        epochs=retrain_epochs_scaled,
                        optimizer=retrain_opt,
                        loss_fn=loss_fn,
                        device=device,
                        batch_size=batch_size,
                        seed=rand_seed,
                        x_val=x_cal_t,
                        y_val=y_cal_t,
                        x_test=x_test_t,
                        y_test=y_test_adj_t,
                        patience=early_stopping_patience if use_early_stopping else 0,
                        loss_plot_path=retrain_loss_plot_path,
                    )
                    
                    corr_eval, mse_adj = evaluate_model(
                        retrain_model, x_test_t, y_test_adj_t, device, y_eval=y_test_eval_t
                    )
                    
                    result = RemovalCurveResult(
                        target_island=target_island_code,
                        method="random",
                        removal_fraction=frac,
                        cal_fraction=cal_fraction,
                        corr_eval=corr_eval,
                        mse_adj=mse_adj,
                        n_train_remaining=n_keep,
                        seed=rand_seed,
                        epochs_trained=epochs_trained,
                    )
                    removal_results.append(result)
                    
                    del retrain_model, x_remaining_t, y_remaining_t
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                logger.info(f"Random baseline (seed={rand_seed}) complete")
        
        elif experiment_mode == "iterative_removal":
            # Random baseline with iterative removal
            for rand_seed in range(seed, seed + n_random_seeds):
                np.random.seed(rand_seed)
                
                # Start with full dataset
                X_current = X_source.copy()
                y_current = y_source.copy()
                n_source_initial = len(X_source)
                
                for iteration in range(iterative_n_iterations):
                    logger.info(f"Random baseline iter {iteration + 1}/{iterative_n_iterations}, seed={rand_seed}, n_samples={len(X_current)}")
                    removed_fraction = min(iteration * iterative_removal_fraction, 1.0)
                    
                    # Retrain on current dataset
                    x_current_t = torch.from_numpy(X_current).float()
                    y_current_t = torch.from_numpy(y_current).float()
                    
                    retrain_model = make_model(in_dim, train_params)
                    retrain_opt = _optimizer(
                        train_params.optimizer,
                        retrain_model.parameters(),
                        train_params.lr,
                        train_params.weight_decay
                    )

                    iter_epochs = train_params.epochs
                    retrain_loss_plot_path = None
                    if output_dir and save_retrain_loss_curves:
                        cal_tag = _format_cal_fraction(cal_fraction)
                        retrain_loss_plot_path = os.path.join(
                            output_dir,
                            f"loss_curve_random_iterative_island_{target_island_code}_cal_{cal_tag}_seed_{rand_seed}_iter_{iteration + 1:02d}.png",
                        )

                    retrain_model, iter_checkpoints = train_with_checkpoints(
                        model=retrain_model,
                        x_train=x_current_t,
                        y_train=y_current_t,
                        epochs=iter_epochs,
                        optimizer=retrain_opt,
                        loss_fn=loss_fn,
                        device=device,
                        n_checkpoints=n_checkpoints,
                        batch_size=batch_size,
                        seed=rand_seed + iteration * 1000,
                        x_val=x_cal_t,
                        y_val=y_cal_t,
                        x_test=x_test_t,
                        y_test=y_test_adj_t,
                        loss_plot_path=retrain_loss_plot_path,
                    )
                    epochs_trained = iter_epochs

                    best_ckpt, best_val_loss = _select_best_checkpoint_by_val_loss(
                        iter_checkpoints, retrain_model, x_cal_t, y_cal_t
                    )
                    retrain_model.load_state_dict(best_ckpt.state_dict)
                    logger.info(
                        f"Random iter {iteration + 1}: selected checkpoint epoch {best_ckpt.epoch} "
                        f"with calibration loss {best_val_loss:.6f}"
                    )
                    
                    corr_eval, mse_adj = evaluate_model(
                        retrain_model, x_test_t, y_test_adj_t, device, y_eval=y_test_eval_t
                    )
                    
                    result = RemovalCurveResult(
                        target_island=target_island_code,
                        method="random_iterative",
                        removal_fraction=removed_fraction,
                        cal_fraction=cal_fraction,
                        corr_eval=corr_eval,
                        mse_adj=mse_adj,
                        n_train_remaining=len(X_current),
                        seed=rand_seed,
                        epochs_trained=epochs_trained,
                    )
                    removal_results.append(result)
                    
                    # If not the last iteration, randomly remove samples
                    if iteration < iterative_n_iterations - 1:
                        # Remove to match target cumulative removal on original size.
                        target_removed_next = min((iteration + 1) * iterative_removal_fraction, 1.0)
                        target_remaining_next = max(1, int(round((1.0 - target_removed_next) * n_source_initial)))
                        n_remove = len(X_current) - target_remaining_next
                        if n_remove <= 0:
                            logger.info("No additional random removal needed to reach next target fraction")
                            continue
                        
                        # Random permutation and keep
                        random_order = np.random.permutation(len(X_current))
                        keep_indices_local = random_order[n_remove:]
                        
                        X_current = X_current[keep_indices_local]
                        y_current = y_current[keep_indices_local]
                        
                        logger.info(f"Randomly removed {n_remove} samples, {len(X_current)} remaining")
                    
                    del retrain_model, x_current_t, y_current_t
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                logger.info(f"Random iterative baseline (seed={rand_seed}) complete")
        
        elif experiment_mode == "iterative_addition":
            # Random baseline for iterative addition — adds randomly instead of by TracIn score
            for rand_seed in range(seed, seed + n_random_seeds):
                n_source = len(X_source)
                n_initial = max(1, int(round(addition_initial_fraction * n_source)))
                
                # SAME deterministic initial subset as TracIn (seeded by rand_seed)
                rng_init = np.random.default_rng(rand_seed)
                initial_indices = rng_init.choice(n_source, size=n_initial, replace=False)
                
                # Remaining indices in random order
                remaining = [i for i in range(n_source) if i not in set(initial_indices.tolist())]
                rng_remain = np.random.default_rng(rand_seed + 999_999)
                rng_remain.shuffle(remaining)
                
                current_indices = list(initial_indices)
                pointer = 0
                
                for step in range(addition_n_steps + 1):
                    added_fraction = addition_initial_fraction + step * addition_fraction_per_step
                    added_fraction = min(added_fraction, 1.0)
                    
                    logger.info(
                        f"Random addition step {step}/{addition_n_steps}, seed={rand_seed}, "
                        f"frac={added_fraction:.0%}, n={len(current_indices)}"
                    )
                    
                    X_sub = X_source[current_indices]
                    y_sub = y_source[current_indices]
                    
                    set_seed(rand_seed + step * 1000)
                    x_sub_t = torch.from_numpy(X_sub).float()
                    y_sub_t = torch.from_numpy(y_sub).float()
                    
                    retrain_model = make_model(in_dim, train_params)
                    retrain_opt = _optimizer(
                        train_params.optimizer,
                        retrain_model.parameters(),
                        train_params.lr,
                        train_params.weight_decay,
                    )
                    
                    retrain_loss_plot_path = None
                    if output_dir and save_retrain_loss_curves:
                        cal_tag = _format_cal_fraction(cal_fraction)
                        retrain_loss_plot_path = os.path.join(
                            output_dir,
                            f"loss_curve_addition_random_island_{target_island_code}_cal_{cal_tag}_seed_{rand_seed}_step_{step:02d}.png",
                        )
                    
                    retrain_model, epochs_trained_step = train_simple(
                        model=retrain_model,
                        x_train=x_sub_t,
                        y_train=y_sub_t,
                        epochs=train_params.epochs,
                        optimizer=retrain_opt,
                        loss_fn=loss_fn,
                        device=device,
                        batch_size=batch_size,
                        seed=rand_seed + step * 1000,
                        x_val=x_cal_t,
                        y_val=y_cal_t,
                        x_test=x_test_t,
                        y_test=y_test_adj_t,
                        patience=0,
                        loss_plot_path=retrain_loss_plot_path,
                        restore_best=True,
                    )
                    
                    corr_eval, mse_adj = evaluate_model(
                        retrain_model, x_test_t, y_test_adj_t, device, y_eval=y_test_eval_t
                    )
                    
                    result = RemovalCurveResult(
                        target_island=target_island_code,
                        method="random_addition",
                        removal_fraction=added_fraction,  # fraction included
                        cal_fraction=cal_fraction,
                        corr_eval=corr_eval,
                        mse_adj=mse_adj,
                        n_train_remaining=len(current_indices),
                        seed=rand_seed,
                        epochs_trained=train_params.epochs,
                    )
                    removal_results.append(result)
                    
                    del retrain_model, x_sub_t, y_sub_t
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Add random samples for next step
                    if step < addition_n_steps and added_fraction < 1.0:
                        target_n = min(n_source, int(round((added_fraction + addition_fraction_per_step) * n_source)))
                        n_to_add = target_n - len(current_indices)
                        if n_to_add > 0:
                            new_indices = remaining[pointer:pointer + n_to_add]
                            pointer += n_to_add
                            current_indices.extend(new_indices)
                            logger.info(f"Randomly added {len(new_indices)} samples, total {len(current_indices)}")
                
                logger.info(f"Random addition baseline (seed={rand_seed}) complete")
    
    # ========================================
    # Build results object
    # ========================================
    score_corr = _spearman_corr_scores(scores_by_seed)
    results = TracInExperimentResults(
        target_island=target_island_code,
        target_island_name=target_name,
        n_source_train=len(X_source),
        n_target_cal=len(X_cal),
        n_target_test=len(X_test),
        scores_by_seed=scores_by_seed,
        removal_curves=removal_results,
        score_corr=score_corr,
        config={
            "experiment_mode": experiment_mode,
            "cal_fraction": cal_fraction,
            "max_cal_fraction": max_cal_fraction,
            "removal_fractions": removal_fractions,
            "iterative_removal_fraction": iterative_removal_fraction,
            "iterative_n_iterations": iterative_n_iterations,
            "checkpoint_training_every_iteration": checkpoint_training_every_iteration,
            "n_checkpoints": n_checkpoints,
            "tracin_mode": tracin_mode,
            "batch_size": batch_size,
            "n_random_seeds": n_random_seeds,
            "seed": seed,
            "epoch_scaling": epoch_scaling,
            "early_stopping_patience": early_stopping_patience,
            "snp_selection_mode": snp_selection_mode,
            "tracin_removal_mode": tracin_removal_mode,
            "save_loss_curves": save_loss_curves,
            "save_retrain_loss_curves": save_retrain_loss_curves,
            "use_early_stopping": use_early_stopping,
            "use_snp_selection": use_snp_selection,
            "num_snps": num_snps,
            "retrain_epochs": retrain_epochs,
            "addition_initial_fraction": addition_initial_fraction,
            "addition_fraction_per_step": addition_fraction_per_step,
            "addition_n_steps": addition_n_steps,
            "train_params": {
                "lr": train_params.lr,
                "weight_decay": train_params.weight_decay,
                "epochs": train_params.epochs,
                "loss_name": train_params.loss_name,
                "optimizer": train_params.optimizer,
                "hidden_dims": train_params.hidden_dims,
                "dropout": train_params.dropout,
                "batch_norm": train_params.batch_norm,
            }
        }
    )
    
    # ========================================
    # Save results if output_dir provided
    # ========================================
    if output_dir:
        save_results(results, output_dir)
    
    return results


def save_results(results: TracInExperimentResults, output_dir: str):
    """Save experiment results to files."""
    os.makedirs(output_dir, exist_ok=True)
    cal_tag = _format_cal_fraction(results.config.get("cal_fraction", 0.0))
    
    # Save scores CSVs (per seed)
    for seed, df in results.scores_by_seed.items():
        scores_path = os.path.join(
            output_dir, f"tracin_scores_island_{results.target_island}_cal_{cal_tag}_seed_{seed}.csv"
        )
        df.to_csv(scores_path, index=False)
        logger.info(f"Saved scores to {scores_path}")
    
    # Save removal curves CSV
    curves_data = []
    for r in results.removal_curves:
        curves_data.append({
            "target_island": r.target_island,
            "method": r.method,
            "removal_fraction": r.removal_fraction,
            "cal_fraction": r.cal_fraction,
            "corr_eval": r.corr_eval,
            "mse_adj": r.mse_adj,
            "n_train_remaining": r.n_train_remaining,
            "seed": r.seed,
            "epochs_trained": r.epochs_trained,
        })
    curves_df = pd.DataFrame(curves_data)
    curves_path = os.path.join(
        output_dir, f"removal_curves_island_{results.target_island}_cal_{cal_tag}.csv"
    )
    curves_df.to_csv(curves_path, index=False)
    logger.info(f"Saved removal curves to {curves_path}")
    
    # Save summary JSON
    summary = {
        "target_island": results.target_island,
        "target_island_name": results.target_island_name,
        "n_source_train": results.n_source_train,
        "n_target_cal": results.n_target_cal,
        "n_target_test": results.n_target_test,
        "config": results.config,
    }
    if results.score_corr is not None:
        summary["tracin_score_corr"] = results.score_corr.to_dict()
    summary_path = os.path.join(
        output_dir, f"summary_island_{results.target_island}_cal_{cal_tag}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")
    
    # Generate and save plot
    try:
        plot_removal_curves(results, output_dir)
        plot_tracin_score_distributions(results, output_dir)
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")


def plot_removal_curves(results: TracInExperimentResults, output_dir: str):
    """Generate and save removal curve plots (boxplots or error bars)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Determine experiment mode from config
    experiment_mode = results.config.get("experiment_mode", "removal_curve")
    
    curves_df = pd.DataFrame([{
        "method": r.method,
        "removal_fraction": r.removal_fraction,
        "cal_fraction": r.cal_fraction,
        "corr_eval": r.corr_eval,
        "mse_adj": r.mse_adj,
        "seed": r.seed,
        "n_train_remaining": r.n_train_remaining,
    } for r in results.removal_curves])
    
    if experiment_mode == "weighted_sampling":
        # No plots for weighted sampling mode - just single point per seed
        return
    
    for cal_frac in sorted(curves_df["cal_fraction"].unique()):
        df_cal = curves_df[curves_df["cal_fraction"] == cal_frac]
        
        # Determine x-axis based on mode
        if experiment_mode == "iterative_removal":
            # X-axis is cumulative percent removed
            iterations = sorted(df_cal["removal_fraction"].unique())  # Using removal_fraction field for iteration
            positions = np.arange(len(iterations))
            x_labels = [f"{i:.0%}" for i in iterations]
            x_label_text = "Fraction of Training Data Removed"
            
            # Methods are tracin_iterative and random_iterative
            tracin_method = "tracin_iterative"
            random_method = "random_iterative"
        elif experiment_mode == "iterative_addition":
            # X-axis is fraction of data included
            x_vals = sorted(df_cal["removal_fraction"].unique())  # repurposed as fraction included
            positions = np.arange(len(x_vals))
            x_labels = [f"{v:.0%}" for v in x_vals]
            x_label_text = "Fraction of Training Data Included"
            
            tracin_method = "tracin_addition"
            random_method = "random_addition"
        else:  # removal_curve
            # X-axis is removal fraction
            fractions = sorted(df_cal["removal_fraction"].unique())
            positions = np.arange(len(fractions))
            x_labels = [f"{f:.0%}" for f in fractions]
            x_label_text = "Fraction of Training Data Removed"
            
            # Methods are tracin and random
            tracin_method = "tracin"
            random_method = "random"
        
        # Determine number of seeds
        n_seeds = df_cal["seed"].nunique()
        use_errorbar = n_seeds <= 5
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Collect x-axis values for iteration in _plot_metric
        if experiment_mode == "iterative_removal":
            _x_fracs = iterations
        elif experiment_mode == "iterative_addition":
            _x_fracs = x_vals
        else:
            _x_fracs = fractions
        
        def _plot_metric(ax, metric: str, ylabel: str, title: str):
            tracin_vals = [
                df_cal[(df_cal["method"] == tracin_method) & (df_cal["removal_fraction"] == f)][metric].to_numpy()
                for f in _x_fracs
            ]
            random_vals = [
                df_cal[(df_cal["method"] == random_method) & (df_cal["removal_fraction"] == f)][metric].to_numpy()
                for f in _x_fracs
            ]
            
            if use_errorbar:
                # Use error bars for n_seeds <= 5
                tracin_means = [np.mean(v) if len(v) > 0 else 0 for v in tracin_vals]
                tracin_stds = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in tracin_vals]
                random_means = [np.mean(v) if len(v) > 0 else 0 for v in random_vals]
                random_stds = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in random_vals]
                
                ax.errorbar(
                    positions - 0.1,
                    tracin_means,
                    yerr=tracin_stds,
                    fmt='o',
                    color="#4C78A8",
                    markersize=8,
                    capsize=5,
                    capthick=2,
                    label="TracIn",
                )
                ax.errorbar(
                    positions + 0.1,
                    random_means,
                    yerr=random_stds,
                    fmt='s',
                    color="#B0B0B0",
                    markersize=8,
                    capsize=5,
                    capthick=2,
                    label="Random",
                )
            else:
                # Use box plots for n_seeds > 5
                ax.boxplot(
                    tracin_vals,
                    positions=positions - 0.2,
                    widths=0.35,
                    patch_artist=True,
                    boxprops=dict(facecolor="#4C78A8", alpha=0.6),
                    medianprops=dict(color="#1F2D3D"),
                )
                ax.boxplot(
                    random_vals,
                    positions=positions + 0.2,
                    widths=0.35,
                    patch_artist=True,
                    boxprops=dict(facecolor="#B0B0B0", alpha=0.6),
                    medianprops=dict(color="#1F2D3D"),
                )
                ax.legend(
                    handles=[
                        Patch(facecolor="#4C78A8", label="TracIn"),
                        Patch(facecolor="#B0B0B0", label="Random"),
                    ],
                    loc="best",
                )
            
            ax.set_xticks(positions)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel(x_label_text, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.grid(True, alpha=0.3)
            if use_errorbar:
                ax.legend(loc="best")
        
        _plot_metric(
            axes[0],
            "corr_eval",
            "Pearson r (pred adj vs orig pheno)",
            f"Correlation with Original Phenotype\nIsland {results.target_island} ({results.target_island_name}), cal={cal_frac:.0%}",
        )
        _plot_metric(
            axes[1],
            "mse_adj",
            "MSE (pred adj vs actual adj)",
            f"MSE on Adjusted Phenotype\nIsland {results.target_island}, cal={cal_frac:.0%}",
        )
        
        plt.tight_layout()
        cal_tag = _format_cal_fraction(cal_frac)
        plot_path = os.path.join(
            output_dir, f"removal_curves_island_{results.target_island}_cal_{cal_tag}.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {plot_path}")


def plot_tracin_score_distributions(results: TracInExperimentResults, output_dir: str):
    """Plot TracIn score distributions per seed."""
    import matplotlib.pyplot as plt
    
    if not results.scores_by_seed:
        return
    
    plt.figure(figsize=(8, 5))
    for seed, df in results.scores_by_seed.items():
        scores = df["score"].to_numpy()
        plt.hist(scores, bins=50, alpha=0.4, label=f"seed {seed}")
    
    plt.xlabel("TracIn score")
    plt.ylabel("Count")
    cal_frac = results.config.get("cal_fraction", 0.0)
    plt.title(
        f"TracIn Score Distribution\n"
        f"Island {results.target_island} ({results.target_island_name}), cal={cal_frac:.0%}"
    )
    plt.legend()
    plt.tight_layout()
    
    cal_tag = _format_cal_fraction(results.config.get("cal_fraction", 0.0))
    plot_path = os.path.join(
        output_dir, f"tracin_score_dist_island_{results.target_island}_cal_{cal_tag}.png"
    )
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved score distribution plot to {plot_path}")


def _format_cal_fraction(cal_fraction: float) -> str:
    """Format calibration fraction for filenames."""
    pct = int(round(float(cal_fraction) * 100))
    return f"{pct:02d}"


def _scale_epochs(base_epochs: int, n_source: int, n_remaining: int, mode: str = "none") -> int:
    """Optionally scale epochs when data is removed.
    
    Parameters
    ----------
    mode : str
        "none"  – keep base_epochs unchanged (default)
        "sqrt"  – scale by sqrt(n_source / n_remaining)
        "linear" – scale by n_source / n_remaining
    """
    if mode == "none" or n_remaining <= 0 or n_remaining >= n_source:
        return base_epochs
    ratio = float(n_source) / float(n_remaining)
    if mode == "sqrt":
        scale = np.sqrt(ratio)
    elif mode == "linear":
        scale = ratio
    else:
        return base_epochs
    return max(1, int(round(base_epochs * scale)))


def _rank_array(x: np.ndarray) -> np.ndarray:
    """Simple rank transform (0..n-1) without tie handling."""
    order = np.argsort(x)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(x))
    return ranks


def _spearman_corr_scores(scores_by_seed: Dict[int, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Compute Spearman rank correlation across TracIn score vectors."""
    seeds = sorted(scores_by_seed.keys())
    if len(seeds) < 2:
        return None
    
    score_matrix = []
    for seed in seeds:
        score_matrix.append(scores_by_seed[seed]["score"].to_numpy())
    score_matrix = np.stack(score_matrix, axis=0)
    
    rank_matrix = np.vstack([_rank_array(row) for row in score_matrix])
    corr = np.corrcoef(rank_matrix)
    return pd.DataFrame(corr, index=seeds, columns=seeds)
