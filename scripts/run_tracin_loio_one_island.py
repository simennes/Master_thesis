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
    
    # Resolve target island
    target_spec = config.get("target_island")
    if target_spec is None:
        # Default to first island
        target_island_code = int(np.unique(locality)[0])
        logger.warning(f"No target island specified, using first: {target_island_code}")
    else:
        target_island_code = resolve_island_code(target_spec, code_to_label, present_codes)
    
    if target_island_code not in present_codes:
        raise ValueError(f"Target island code {target_island_code} not found in data")
    
    # Log resolved target
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
    
    # Other config
    cal_fraction = float(config.get("cal_fraction", 0.2))
    removal_fractions = config.get("removal_fractions", [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    batch_size = int(config.get("batch_size", 64))
    n_random_seeds = int(config.get("n_random_seeds", 3))
    use_snp_selection = config.get("use_snp_selection", False)
    num_snps = config.get("num_snps", None)
    
    output_dir = paths.get("output_dir", "outputs/tracin")
    
    # Run experiment
    logger.info("=" * 60)
    logger.info("Starting TracIn removal curve experiment")
    logger.info("=" * 60)
    
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
        removal_fractions=removal_fractions,
        n_checkpoints=n_checkpoints,
        tracin_mode=tracin_mode,
        batch_size=batch_size,
        n_random_seeds=n_random_seeds,
        seed=seed,
        output_dir=output_dir,
        use_snp_selection=use_snp_selection,
        num_snps=num_snps,
        device=device,
    )
    
    logger.info("=" * 60)
    logger.info("Experiment complete!")
    logger.info("=" * 60)
    
    # Print summary
    logger.info(f"Target island: {results.target_island} ({results.target_island_name})")
    logger.info(f"Source training samples: {results.n_source_train}")
    logger.info(f"Target calibration samples: {results.n_target_cal}")
    logger.info(f"Target test samples: {results.n_target_test}")
    
    # Print removal curve results
    tracin_results = [r for r in results.removal_curves if r.method == "tracin"]
    logger.info("\nTracIn-guided removal curve:")
    logger.info("  (corr_eval = corr(pred_adj, orig_pheno), mse_adj = mse(pred_adj, adj_pheno))")
    for r in sorted(tracin_results, key=lambda x: x.removal_fraction):
        logger.info(f"  {r.removal_fraction:.0%} removed -> corr_eval={r.corr_eval:.4f}, mse_adj={r.mse_adj:.6f}")
    
    # Random baseline summary
    import pandas as pd
    random_df = pd.DataFrame([{
        "frac": r.removal_fraction,
        "corr_eval": r.corr_eval,
        "mse_adj": r.mse_adj
    } for r in results.removal_curves if r.method == "random"])
    
    if len(random_df) > 0:
        random_agg = random_df.groupby("frac").agg({"corr_eval": ["mean", "std"], "mse_adj": ["mean", "std"]})
        logger.info("\nRandom removal baseline (mean ± std):")
        for frac in sorted(random_df["frac"].unique()):
            row = random_agg.loc[frac]
            logger.info(f"  {frac:.0%} removed -> corr_eval={row['corr_eval']['mean']:.4f}±{row['corr_eval']['std']:.4f}")
    
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
