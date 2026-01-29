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
from typing import Any, Dict, List, Optional

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
    corr_eval: float  # Pearson r(predicted adjusted, original phenotype)
    mse_adj: float    # MSE(predicted adjusted, actual adjusted phenotype)
    n_train_remaining: int
    seed: int


@dataclass
class TracInExperimentResults:
    """Complete results from a TracIn experiment for one target island."""
    target_island: int
    target_island_name: str
    n_source_train: int
    n_target_cal: int
    n_target_test: int
    
    # Per-individual scores
    scores_df: pd.DataFrame  # columns: ringnr, island, score, rank
    
    # Removal curves
    removal_curves: List[RemovalCurveResult] = field(default_factory=list)
    
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
    
    # Random shuffle for split
    perm = np.random.permutation(n_target)
    cal_idx = perm[:n_cal]
    test_idx = perm[n_cal:]
    
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
    removal_fractions: List[float] = None,
    n_checkpoints: int = 5,
    tracin_mode: str = "lastlayer",
    batch_size: int = 64,
    n_random_seeds: int = 3,
    seed: int = 42,
    output_dir: Optional[str] = None,
    use_snp_selection: bool = False,
    num_snps: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> TracInExperimentResults:
    """
    Run the complete TracIn removal curve experiment for one target island.
    
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
    removal_fractions : List[float]
        Fractions of training data to remove
    n_checkpoints : int
        Number of checkpoints for TracIn
    tracin_mode : str
        "full" or "lastlayer"
    batch_size : int
        Training batch size
    n_random_seeds : int
        Number of random seeds for baseline
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
        
    Returns
    -------
    results : TracInExperimentResults
        Complete experiment results
    """
    if removal_fractions is None:
        removal_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
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
        cal_fraction=cal_fraction, seed=seed
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
    
    logger.info(f"Data split: source={len(X_source)}, cal={len(X_cal)}, test={len(X_test)}")
    
    # ========================================
    # Step 2: Feature selection (fit on source only)
    # ========================================
    snp_cols = None
    if use_snp_selection and num_snps is not None and num_snps < X_source.shape[1]:
        logger.info(f"Selecting top {num_snps} SNPs based on source data")
        snp_cols = _select_top_snps_by_abs_corr(X_source, y_source, num_snps)
        X_source = X_source[:, snp_cols]
        X_cal = X_cal[:, snp_cols]
        X_test = X_test[:, snp_cols]
    
    in_dim = X_source.shape[1]
    logger.info(f"Input dimension after feature selection: {in_dim}")
    
    # ========================================
    # Step 3: Train with checkpoints
    # ========================================
    logger.info("Training model with checkpoints...")
    
    x_source_t = torch.from_numpy(X_source).float()
    y_source_t = torch.from_numpy(y_source).float()
    x_cal_t = torch.from_numpy(X_cal).float().to(device)
    y_cal_t = torch.from_numpy(y_cal).float().to(device)
    x_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_adj_t = torch.from_numpy(y_test).float()  # Adjusted phenotype
    y_test_eval_t = torch.from_numpy(y_eval_test).float()  # Original phenotype
    
    model = make_model(in_dim, train_params)
    optimizer = _optimizer(
        train_params.optimizer,
        model.parameters(),
        train_params.lr,
        train_params.weight_decay
    )
    loss_fn = make_loss(train_params.loss_name)
    
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
        seed=seed,
    )
    
    logger.info(f"Training complete, saved {len(checkpoints)} checkpoints")
    
    # ========================================
    # Step 4: Compute TracIn scores
    # ========================================
    logger.info(f"Computing TracIn scores (mode={tracin_mode})...")
    
    x_source_t = x_source_t.to(device)
    y_source_t = y_source_t.to(device)
    
    scores = compute_tracin_scores(
        checkpoints=checkpoints,
        model=model,
        x_train=x_source_t,
        y_train=y_source_t,
        x_cal=x_cal_t,
        y_cal=y_cal_t,
        loss_fn=loss_fn,
        device=device,
        mode=tracin_mode,
    )
    
    # Create scores dataframe
    rank_order = rank_by_influence(scores, ids_source)
    ranks = np.zeros(len(scores), dtype=int)
    for i, idx in enumerate(rank_order):
        ranks[idx] = i
    
    scores_df = pd.DataFrame({
        "ringnr": ids_source,
        "island": locality_source,
        "score": scores,
        "rank": ranks,
    })
    scores_df = scores_df.sort_values("rank")
    
    logger.info(f"Scores computed. Range: [{scores.min():.6f}, {scores.max():.6f}]")
    
    # ========================================
    # Step 5: Generate removal curves
    # ========================================
    logger.info("Generating removal curves...")
    
    removal_results = []
    
    # TracIn-guided removal
    for frac in removal_fractions:
        n_remove = int(frac * len(X_source))
        n_keep = len(X_source) - n_remove
        
        if n_keep < 10:  # Safety check
            logger.warning(f"Skipping fraction {frac}: would keep only {n_keep} samples")
            continue
        
        # Remove lowest-scoring individuals (ascending order)
        keep_indices = rank_order[n_remove:]
        
        X_remaining = X_source[keep_indices]
        y_remaining = y_source[keep_indices]
        
        # Retrain
        x_remaining_t = torch.from_numpy(X_remaining).float()
        y_remaining_t = torch.from_numpy(y_remaining).float()
        
        retrain_model = make_model(in_dim, train_params)
        retrain_opt = _optimizer(
            train_params.optimizer,
            retrain_model.parameters(),
            train_params.lr,
            train_params.weight_decay
        )
        
        retrain_model = train_simple(
            model=retrain_model,
            x_train=x_remaining_t,
            y_train=y_remaining_t,
            epochs=train_params.epochs,
            optimizer=retrain_opt,
            loss_fn=loss_fn,
            device=device,
            batch_size=batch_size,
            seed=seed,
        )
        
        # Evaluate on test set: corr vs original, MSE vs adjusted
        corr_eval, mse_adj = evaluate_model(
            retrain_model, x_test_t, y_test_adj_t, device, y_eval=y_test_eval_t
        )
        
        result = RemovalCurveResult(
            target_island=target_island_code,
            method="tracin",
            removal_fraction=frac,
            corr_eval=corr_eval,
            mse_adj=mse_adj,
            n_train_remaining=n_keep,
            seed=seed,
        )
        removal_results.append(result)
        
        logger.info(f"TracIn removal {frac:.0%}: corr_eval={corr_eval:.4f}, mse_adj={mse_adj:.6f} (n={n_keep})")
        
        del retrain_model, x_remaining_t, y_remaining_t
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Random removal baseline (multiple seeds)
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
            
            retrain_model = train_simple(
                model=retrain_model,
                x_train=x_remaining_t,
                y_train=y_remaining_t,
                epochs=train_params.epochs,
                optimizer=retrain_opt,
                loss_fn=loss_fn,
                device=device,
                batch_size=batch_size,
                seed=rand_seed,
            )
            
            corr_eval, mse_adj = evaluate_model(
                retrain_model, x_test_t, y_test_adj_t, device, y_eval=y_test_eval_t
            )
            
            result = RemovalCurveResult(
                target_island=target_island_code,
                method="random",
                removal_fraction=frac,
                corr_eval=corr_eval,
                mse_adj=mse_adj,
                n_train_remaining=n_keep,
                seed=rand_seed,
            )
            removal_results.append(result)
            
            del retrain_model, x_remaining_t, y_remaining_t
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"Random baseline (seed={rand_seed}) complete")
    
    # ========================================
    # Build results object
    # ========================================
    results = TracInExperimentResults(
        target_island=target_island_code,
        target_island_name=target_name,
        n_source_train=len(X_source),
        n_target_cal=len(X_cal),
        n_target_test=len(X_test),
        scores_df=scores_df,
        removal_curves=removal_results,
        config={
            "cal_fraction": cal_fraction,
            "removal_fractions": removal_fractions,
            "n_checkpoints": n_checkpoints,
            "tracin_mode": tracin_mode,
            "batch_size": batch_size,
            "n_random_seeds": n_random_seeds,
            "seed": seed,
            "use_snp_selection": use_snp_selection,
            "num_snps": num_snps,
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
    
    # Save scores CSV
    scores_path = os.path.join(output_dir, f"tracin_scores_island_{results.target_island}.csv")
    results.scores_df.to_csv(scores_path, index=False)
    logger.info(f"Saved scores to {scores_path}")
    
    # Save removal curves CSV
    curves_data = []
    for r in results.removal_curves:
        curves_data.append({
            "target_island": r.target_island,
            "method": r.method,
            "removal_fraction": r.removal_fraction,
            "corr_eval": r.corr_eval,
            "mse_adj": r.mse_adj,
            "n_train_remaining": r.n_train_remaining,
            "seed": r.seed,
        })
    curves_df = pd.DataFrame(curves_data)
    curves_path = os.path.join(output_dir, f"removal_curves_island_{results.target_island}.csv")
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
    summary_path = os.path.join(output_dir, f"summary_island_{results.target_island}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")
    
    # Generate and save plot
    try:
        plot_removal_curves(results, output_dir)
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")


def plot_removal_curves(results: TracInExperimentResults, output_dir: str):
    """Generate and save removal curve plots."""
    import matplotlib.pyplot as plt
    
    curves_df = pd.DataFrame([{
        "method": r.method,
        "removal_fraction": r.removal_fraction,
        "corr_eval": r.corr_eval,
        "mse_adj": r.mse_adj,
        "seed": r.seed,
    } for r in results.removal_curves])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Correlation plot (corr_eval: predicted vs original phenotype)
    ax = axes[0]
    
    # TracIn
    tracin_data = curves_df[curves_df["method"] == "tracin"]
    ax.plot(tracin_data["removal_fraction"], tracin_data["corr_eval"], 
            "o-", color="blue", linewidth=2, markersize=8, label="TracIn-guided")
    
    # Random (aggregate with mean and std)
    random_data = curves_df[curves_df["method"] == "random"]
    random_agg = random_data.groupby("removal_fraction").agg({
        "corr_eval": ["mean", "std"]
    }).reset_index()
    random_agg.columns = ["removal_fraction", "corr_mean", "corr_std"]
    
    ax.errorbar(random_agg["removal_fraction"], random_agg["corr_mean"],
                yerr=random_agg["corr_std"], fmt="s--", color="gray", 
                linewidth=2, markersize=8, capsize=3, label="Random")
    
    ax.set_xlabel("Fraction of Training Data Removed", fontsize=12)
    ax.set_ylabel("Pearson r (pred adj vs orig pheno)", fontsize=12)
    ax.set_title(f"Correlation with Original Phenotype\nIsland {results.target_island} ({results.target_island_name})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MSE plot (mse_adj: predicted vs adjusted phenotype)
    ax = axes[1]
    
    ax.plot(tracin_data["removal_fraction"], tracin_data["mse_adj"],
            "o-", color="blue", linewidth=2, markersize=8, label="TracIn-guided")
    
    random_agg_mse = random_data.groupby("removal_fraction").agg({
        "mse_adj": ["mean", "std"]
    }).reset_index()
    random_agg_mse.columns = ["removal_fraction", "mse_mean", "mse_std"]
    
    ax.errorbar(random_agg_mse["removal_fraction"], random_agg_mse["mse_mean"],
                yerr=random_agg_mse["mse_std"], fmt="s--", color="gray",
                linewidth=2, markersize=8, capsize=3, label="Random")
    
    ax.set_xlabel("Fraction of Training Data Removed", fontsize=12)
    ax.set_ylabel("MSE (pred adj vs actual adj)", fontsize=12)
    ax.set_title(f"MSE on Adjusted Phenotype\nIsland {results.target_island}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"removal_curves_island_{results.target_island}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {plot_path}")
