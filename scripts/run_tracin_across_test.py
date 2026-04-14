#!/usr/bin/env python
"""
Run TracIn-based training set optimization for one target island.

This script:
1. Loads data and splits target island into calibration and test sets
2. Trains an MLP on source data with checkpoints
3. Computes TracIn influence scores for each source individual
4. Generates removal curves comparing TracIn-guided vs random removal
5. Saves scores, curves, and plots

Usage:
    python scripts/run_tracin_loio_one_island.py --config config/tracin_config.json

All settings are controlled via the config file.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.data import load_data
from src.models import TrainParams
from src.removal_curve import run_removal_curve_experiment
from src.cv_utils import ISLAND_ID_TO_NAME
from src.utils import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_island_code(target_island, code_to_label: dict, present_codes: set) -> int:
    """
    Resolve target island specification to an encoded code.
    
    Accepts:
    - Encoded code directly (0-15)
    - Original island label (20, 22, 23, etc.)
    - Island name (string like "Træna")
    
    The function checks:
    1. If the value is already a valid encoded code, use it directly
    2. If it's an original label, find the encoded code that maps to it
    3. If it's an island name, resolve via ISLAND_ID_TO_NAME
    """
    # Convert to int if possible
    try:
        val = int(target_island)
    except (ValueError, TypeError):
        val = None
    
    if val is not None:
        # Check if it's already a valid encoded code
        if val in present_codes:
            return val
        
        # Check if it's an original label that needs to be mapped to encoded code
        for code, label in code_to_label.items():
            if int(label) == val:
                return code
        
        # Not found
        raise ValueError(f"Island {val} not found as code or label")
    
    # String name matching
    if isinstance(target_island, str):
        target_lower = target_island.lower()
        
        # Try matching by island name via ISLAND_ID_TO_NAME
        for orig_label, name in ISLAND_ID_TO_NAME.items():
            if name.lower() == target_lower:
                # Find the encoded code for this original label
                for code, lbl in code_to_label.items():
                    if int(lbl) == orig_label:
                        return code
        
        # Try matching by label string in code_to_label
        for code, label in code_to_label.items():
            if str(label).lower() == target_lower:
                return code
    
    raise ValueError(f"Could not resolve target island: {target_island}")


def main():
    parser = argparse.ArgumentParser(
        description="Run TracIn training set optimization for one target island"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)
    
    logger.info(f"Loaded config from {args.config}")
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Set device
    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    
    # Load data
    paths = config["paths"]
    target_column = config.get("target_column", "y_adjusted")
    eval_target_column = config.get("eval_target_column", "y_mean")
    
    logger.info("Loading data...")
    X, y, ids, GRM_df, locality, code_to_label, y_eval = load_data(
        paths,
        target_column=target_column,
        standardize_features=False,
        return_locality=True,
        min_count=20,
        return_eval=True,
        eval_target_column=eval_target_column,
    )
    
    if y_eval is None:
        logger.warning(f"y_eval ('{eval_target_column}') not found in data; using y for evaluation")
        y_eval = y.copy()
    
    logger.info(f"Loaded {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Islands present: {sorted(np.unique(locality).tolist())}")
    
    # Print island mapping for reference
    if code_to_label:
        logger.info("Island code -> label mapping:")
        for code in sorted(code_to_label.keys()):
            label = code_to_label[code]
            name = ISLAND_ID_TO_NAME.get(int(label), label)
            count = (locality == code).sum()
            logger.info(f"  {code} -> {label} ({name}): n={count}")
    
    # Get set of present encoded codes
    present_codes = set(np.unique(locality).astype(int).tolist())

    # Resolve optional included islands (for source/training filtering)
    included_specs = config.get("included_islands", None)
    included_island_codes = None
    if included_specs not in (None, []):
        if not isinstance(included_specs, list):
            included_specs = [included_specs]
        included_island_codes = []
        for included_spec in included_specs:
            included_code = resolve_island_code(included_spec, code_to_label, present_codes)
            included_island_codes.append(included_code)
        included_island_codes = sorted(set(int(v) for v in included_island_codes))
        logger.info(f"Included islands (source filter): {included_island_codes}")
    
    # Resolve target islands (single, list, or random selection)
    target_specs = config.get("target_islands")
    if target_specs is None or target_specs == []:
        n_target_islands = config.get("n_target_islands")
        if n_target_islands is not None:
            n_target_islands = int(n_target_islands)
            if n_target_islands <= 0:
                raise ValueError("n_target_islands must be > 0")
            if n_target_islands > len(present_codes):
                raise ValueError("n_target_islands exceeds number of available islands")
            rng = np.random.default_rng(seed)
            target_island_codes = rng.choice(
                sorted(present_codes), size=n_target_islands, replace=False
            ).tolist()
        else:
            target_specs = [config.get("target_island")]
            if not isinstance(target_specs, list):
                target_specs = [target_specs]
            target_island_codes = []
            for target_spec in target_specs:
                if target_spec is None:
                    target_island_code = int(np.unique(locality)[0])
                    logger.warning(f"No target island specified, using first: {target_island_code}")
                else:
                    target_island_code = resolve_island_code(target_spec, code_to_label, present_codes)
                if target_island_code not in present_codes:
                    raise ValueError(f"Target island code {target_island_code} not found in data")
                target_island_codes.append(target_island_code)
    else:
        if not isinstance(target_specs, list):
            target_specs = [target_specs]
        target_island_codes = []
        for target_spec in target_specs:
            if target_spec is None:
                target_island_code = int(np.unique(locality)[0])
                logger.warning(f"No target island specified, using first: {target_island_code}")
            else:
                target_island_code = resolve_island_code(target_spec, code_to_label, present_codes)
            if target_island_code not in present_codes:
                raise ValueError(f"Target island code {target_island_code} not found in data")
            target_island_codes.append(target_island_code)
    
    # Log resolved targets
    for target_island_code in target_island_codes:
        target_label = code_to_label.get(target_island_code, target_island_code)
        target_name = ISLAND_ID_TO_NAME.get(int(target_label), target_label)
        logger.info(f"Target island: code={target_island_code}, label={target_label}, name={target_name}")
    
    # Build TrainParams
    tp_config = config.get("train_params", {})
    train_params = TrainParams(
        lr=float(tp_config.get("lr", 1e-4)),
        weight_decay=float(tp_config.get("weight_decay", 1e-5)),
        epochs=int(tp_config.get("epochs", 100)),
        loss_name=tp_config.get("loss_name", "mse"),
        optimizer=tp_config.get("optimizer", "adam"),
        hidden_dims=tp_config.get("hidden_dims", [256, 64]),
        dropout=float(tp_config.get("dropout", 0.3)),
        batch_norm=bool(tp_config.get("batch_norm", False)),
    )
    
    logger.info(f"Train params: epochs={train_params.epochs}, lr={train_params.lr}, hidden={train_params.hidden_dims}")
    
    # TracIn config
    tracin_config = config.get("tracin", {})
    n_checkpoints = int(tracin_config.get("n_checkpoints", 5))
    tracin_mode = tracin_config.get("mode", "lastlayer")

    # Score backend config
    scoring_cfg = config.get("scoring", {})
    score_method = str(scoring_cfg.get("method", "tracin"))
    score_window = str(scoring_cfg.get("window", "best_cal"))
    score_weight_mode = str(scoring_cfg.get("weight_mode", "improvement"))
    score_flip_sign = bool(scoring_cfg.get("flip_sign", True))
    score_prefix = "inrun" if score_method.lower() in {"inrun_firstorder", "inrun"} else "tracin"
    
    # Other config
    cal_fractions = config.get("cal_fractions")
    if cal_fractions is None:
        cal_fractions = [float(config.get("cal_fraction", 0.2))]
    if not isinstance(cal_fractions, list):
        cal_fractions = [float(cal_fractions)]
    cal_fractions = [float(v) for v in cal_fractions]
    max_cal_fraction = max(cal_fractions)
    
    # Experiment mode
    experiment_mode = config.get("experiment_mode", "removal_curve")
    logger.info(f"Experiment mode: {experiment_mode}")

    iterative_addition_cfg = config.get("iterative_addition", {})
    addition_initial_fraction = float(iterative_addition_cfg.get("initial_fraction", 0.2))
    addition_fraction_per_step = float(iterative_addition_cfg.get("fraction_per_step", 0.05))
    addition_n_steps = int(iterative_addition_cfg.get("n_steps", 10))
    addition_fraction_schedule = iterative_addition_cfg.get("fraction_schedule", None)
    if addition_fraction_schedule is not None:
        if not isinstance(addition_fraction_schedule, list):
            raise ValueError("iterative_addition.fraction_schedule must be a list of fractions")
        addition_fraction_schedule = [float(v) for v in addition_fraction_schedule]
    
    # Mode-specific config
    checkpoint_training_every_iteration = False
    if experiment_mode == "removal_curve":
        mode_config = config.get("removal_curve", {})
        removal_fractions = mode_config.get("removal_fractions", [0, 0.1, 0.2, 0.3, 0.4, 0.5])
        n_random_seeds = int(mode_config.get("n_random_seeds", 3))
        tracin_removal_mode = mode_config.get("tracin_removal_mode", "lowest")
        iterative_removal_fraction = 0.05  # Not used
        iterative_n_iterations = 5  # Not used
    elif experiment_mode == "weighted_sampling":
        mode_config = config.get("weighted_sampling", {})
        removal_fractions = [0.0]  # Not used
        n_random_seeds = 1  # Only one seed needed
        tracin_removal_mode = "lowest"  # Not used
        iterative_removal_fraction = 0.05  # Not used
        iterative_n_iterations = 5  # Not used
    elif experiment_mode == "iterative_removal":
        mode_config = config.get("iterative_removal", {})
        removal_fractions = []  # Not used
        n_random_seeds = int(mode_config.get("n_random_seeds", 3))
        tracin_removal_mode = mode_config.get("tracin_removal_mode", "lowest")
        iterative_removal_fraction = float(mode_config.get("removal_fraction_per_iteration", 0.05))
        iterative_n_iterations = int(mode_config.get("n_iterations", 5))
        checkpoint_training_every_iteration = bool(mode_config.get("checkpoint_training_every_iteration", False))
    elif experiment_mode == "iterative_addition":
        mode_config = config.get("iterative_addition", {})
        removal_fractions = []  # Not used
        n_random_seeds = int(mode_config.get("n_random_seeds", 3))
        tracin_removal_mode = "lowest"  # Not used
        iterative_removal_fraction = 0.05  # Not used
        iterative_n_iterations = 5  # Not used
    else:
        raise ValueError(f"Unknown experiment_mode: {experiment_mode}")
    
    # Backward compatibility: support old config format
    if "removal_fractions" in config:
        removal_fractions = config["removal_fractions"]
    if "n_random_seeds" in config:
        n_random_seeds = int(config["n_random_seeds"])
    if "tracin_removal_mode" in config:
        tracin_removal_mode = config["tracin_removal_mode"]
    if "use_tracin_weighted_sampling" in config and config["use_tracin_weighted_sampling"]:
        experiment_mode = "weighted_sampling"
        logger.warning("Using deprecated 'use_tracin_weighted_sampling' - please use 'experiment_mode': 'weighted_sampling'")
    
    batch_size = int(config.get("batch_size", 64))
    use_snp_selection = config.get("use_snp_selection", False)
    num_snps = config.get("num_snps", None)
    snp_selection_mode = config.get("snp_selection_mode", "corr")
    epoch_scaling = config.get("epoch_scaling", "none")
    early_stopping_patience = int(config.get("early_stopping_patience", 0))
    save_loss_curves = bool(config.get("save_loss_curves", False))
    save_retrain_loss_curves = bool(config.get("save_retrain_loss_curves", False))
    use_early_stopping = bool(config.get("use_early_stopping", True))
    retrain_epochs = config.get("retrain_epochs", None)
    if retrain_epochs is not None:
        retrain_epochs = int(retrain_epochs)
    
    output_dir = paths.get("output_dir", "outputs/tracin")
    
    # Run experiment
    logger.info("=" * 60)
    logger.info(
        "Starting %s %s experiment",
        "In-Run" if score_prefix == "inrun" else "TracIn",
        experiment_mode,
    )
    logger.info("=" * 60)
    
    all_results = []
    for target_island_code in target_island_codes:
        for cal_fraction in cal_fractions:
            results = run_removal_curve_experiment(
                X=X,
                y=y,
                ids=ids,
                locality=locality,
                target_island_code=target_island_code,
                train_params=train_params,
                y_eval=y_eval,
                code_to_label=code_to_label,
                cal_fraction=cal_fraction,
                max_cal_fraction=max_cal_fraction,
                experiment_mode=experiment_mode,
                removal_fractions=removal_fractions,
                n_random_seeds=n_random_seeds,
                tracin_removal_mode=tracin_removal_mode,
                iterative_removal_fraction=iterative_removal_fraction,
                iterative_n_iterations=iterative_n_iterations,
                checkpoint_training_every_iteration=checkpoint_training_every_iteration,
                n_checkpoints=n_checkpoints,
                tracin_mode=tracin_mode,
                score_method=score_method,
                score_window=score_window,
                score_weight_mode=score_weight_mode,
                score_flip_sign=score_flip_sign,
                batch_size=batch_size,
                seed=seed,
                output_dir=output_dir,
                use_snp_selection=use_snp_selection,
                num_snps=num_snps,
                device=device,
                epoch_scaling=epoch_scaling,
                early_stopping_patience=early_stopping_patience,
                snp_selection_mode=snp_selection_mode,
                save_loss_curves=save_loss_curves,
                save_retrain_loss_curves=save_retrain_loss_curves,
                use_early_stopping=use_early_stopping,
                retrain_epochs=retrain_epochs,
                included_island_codes=included_island_codes,
                # iterative_addition params
                addition_initial_fraction=addition_initial_fraction,
                addition_fraction_per_step=addition_fraction_per_step,
                addition_n_steps=addition_n_steps,
                addition_fraction_schedule=addition_fraction_schedule,
            )
            all_results.append(results)
    
    logger.info("=" * 60)
    logger.info("Experiment complete!")
    logger.info("=" * 60)
    
    # Print summary
    for results in all_results:
        logger.info(f"Target island: {results.target_island} ({results.target_island_name})")
        logger.info(f"Source training samples: {results.n_source_train}")
        logger.info(f"Target calibration samples: {results.n_target_cal}")
        logger.info(f"Target test samples: {results.n_target_test}")
    
    import pandas as pd
    
    # Print score-guided removal curve results (mean ± std across seeds)
    for results in all_results:
        exp_mode_local = str(results.config.get("experiment_mode", experiment_mode))
        if exp_mode_local == "iterative_addition":
            guided_method = f"{score_prefix}_addition"
            random_method = "random_addition"
        elif exp_mode_local == "iterative_removal":
            guided_method = f"{score_prefix}_iterative"
            random_method = "random_iterative"
        elif exp_mode_local == "weighted_sampling":
            guided_method = f"{score_prefix}_weighted"
            random_method = "random"
        else:
            guided_method = score_prefix
            random_method = "random"

        tracin_df = pd.DataFrame([{
            "frac": r.removal_fraction,
            "cal_fraction": r.cal_fraction,
            "corr_eval": r.corr_eval,
            "mse_adj": r.mse_adj
        } for r in results.removal_curves if r.method == guided_method])
        
        logger.info("\n%s-guided removal curve (mean ± std):", "In-Run" if score_prefix == "inrun" else "TracIn")
        logger.info("  (corr_eval = corr(pred_adj, orig_pheno), mse_adj = mse(pred_adj, adj_pheno))")
        if len(tracin_df) > 0:
            tracin_agg = tracin_df.groupby(["cal_fraction", "frac"]).agg({
                "corr_eval": ["mean", "std"],
                "mse_adj": ["mean", "std"]
            })
            for cal_frac in sorted(tracin_df["cal_fraction"].unique()):
                for frac in sorted(tracin_df["frac"].unique()):
                    row = tracin_agg.loc[(cal_frac, frac)]
                    logger.info(
                        f"  cal={cal_frac:.0%}, {frac:.0%} removed -> "
                        f"corr_eval={row['corr_eval']['mean']:.4f}±{row['corr_eval']['std']:.4f}, "
                        f"mse_adj={row['mse_adj']['mean']:.6f}±{row['mse_adj']['std']:.6f}"
                    )
        
        # Random baseline summary
        random_df = pd.DataFrame([{
            "frac": r.removal_fraction,
            "cal_fraction": r.cal_fraction,
            "corr_eval": r.corr_eval,
            "mse_adj": r.mse_adj
        } for r in results.removal_curves if r.method == random_method])
        
        if len(random_df) > 0:
            random_agg = random_df.groupby(["cal_fraction", "frac"]).agg({
                "corr_eval": ["mean", "std"],
                "mse_adj": ["mean", "std"]
            })
            logger.info("\nRandom removal baseline (mean ± std):")
            for cal_frac in sorted(random_df["cal_fraction"].unique()):
                for frac in sorted(random_df["frac"].unique()):
                    row = random_agg.loc[(cal_frac, frac)]
                    logger.info(
                        f"  cal={cal_frac:.0%}, {frac:.0%} removed -> "
                        f"corr_eval={row['corr_eval']['mean']:.4f}±{row['corr_eval']['std']:.4f}, "
                        f"mse_adj={row['mse_adj']['mean']:.6f}±{row['mse_adj']['std']:.6f}"
                    )
    
    # Save combined removal curves across islands
    if len(all_results) > 1:
        combined_rows = []
        for results in all_results:
            for r in results.removal_curves:
                combined_rows.append({
                    "target_island": results.target_island,
                    "target_island_name": results.target_island_name,
                    "method": r.method,
                    "removal_fraction": r.removal_fraction,
                    "cal_fraction": r.cal_fraction,
                    "corr_eval": r.corr_eval,
                    "mse_adj": r.mse_adj,
                    "seed": r.seed,
                    "epochs_trained": r.epochs_trained,
                })
        combined_df = pd.DataFrame(combined_rows)
        combined_path = f"{output_dir}/combined_removal_curves.csv"
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"Saved combined curves to: {combined_path}")
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
            
            def _cal_tag(value: float) -> str:
                pct = int(round(float(value) * 100))
                return f"{pct:02d}"
            
            for cal_frac in sorted(combined_df["cal_fraction"].unique()):
                df_cal = combined_df[combined_df["cal_fraction"] == cal_frac]
                fractions = sorted(df_cal["removal_fraction"].unique())
                positions = np.arange(len(fractions))

                exp_mode_plot = str(experiment_mode)
                if exp_mode_plot == "iterative_addition":
                    guided_method_plot = f"{score_prefix}_addition"
                    random_method_plot = "random_addition"
                elif exp_mode_plot == "iterative_removal":
                    guided_method_plot = f"{score_prefix}_iterative"
                    random_method_plot = "random_iterative"
                elif exp_mode_plot == "weighted_sampling":
                    guided_method_plot = f"{score_prefix}_weighted"
                    random_method_plot = "random"
                else:
                    guided_method_plot = score_prefix
                    random_method_plot = "random"
                
                # Determine number of seeds
                n_seeds = df_cal["seed"].nunique()
                use_errorbar = n_seeds <= 5
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                def _plot_metric(ax, metric: str, ylabel: str, title: str):
                    tracin_vals = [
                        df_cal[(df_cal["method"] == guided_method_plot) & (df_cal["removal_fraction"] == f)][metric].to_numpy()
                        for f in fractions
                    ]
                    random_vals = [
                        df_cal[(df_cal["method"] == random_method_plot) & (df_cal["removal_fraction"] == f)][metric].to_numpy()
                        for f in fractions
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
                            label="In-Run" if score_prefix == "inrun" else "TracIn",
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
                                Patch(facecolor="#4C78A8", label="In-Run" if score_prefix == "inrun" else "TracIn"),
                                Patch(facecolor="#B0B0B0", label="Random"),
                            ],
                            loc="best",
                        )
                    
                    ax.set_xticks(positions)
                    ax.set_xticklabels([f"{f:.0%}" for f in fractions])
                    ax.set_xlabel("Fraction of Training Data Removed")
                    ax.set_ylabel(ylabel)
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3)
                    if use_errorbar:
                        ax.legend(loc="best")
                
                _plot_metric(
                    axes[0],
                    "corr_eval",
                    "Pearson r (pred adj vs orig pheno)",
                    f"Combined Correlation (Across Islands), cal={cal_frac:.0%}",
                )
                _plot_metric(
                    axes[1],
                    "mse_adj",
                    "MSE (pred adj vs actual adj)",
                    f"Combined MSE (Across Islands), cal={cal_frac:.0%}",
                )
                
                plt.tight_layout()
                combined_plot = f"{output_dir}/combined_removal_curves_cal_{_cal_tag(cal_frac)}.png"
                plt.savefig(combined_plot, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"Saved combined plot to: {combined_plot}")
        except Exception as e:
            logger.warning(f"Could not generate combined plot: {e}")
    
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
