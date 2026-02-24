#!/usr/bin/env python
"""
Plot train/cal/test loss curves for different numbers of top-ranked source islands.

Ranking is taken from TMC-Shapley output (phi_per_individual):
- preferred: shapley_uncertainty_target_<id>.csv (column phi_per_ind_mean)
- fallback:  shapley_island_values_target_<id>.csv (column phi_per_individual)

Example
-------
python scripts/plot_loss_curves_by_island_count.py \
  --config config/tmc_shapley_islands.json \
  --target_island 0 \
  --island_counts 1 3 7 10 15
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.models import TrainParams, make_model
from src.removal_curve import split_target_island
from src.utils import _optimizer, _select_top_snps_by_abs_corr, make_loss, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_island_code(target_island, code_to_label: Dict[int, str], present_codes: set) -> int:
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


def infer_ranking_csv(output_dir_base: str, target_code: int) -> Path:
    base = Path(output_dir_base) / f"island_{target_code}"
    candidates = [
        base / f"shapley_uncertainty_target_{target_code}.csv",
        base / f"shapley_island_values_target_{target_code}.csv",
        base / "repeat_0" / f"shapley_island_values_target_{target_code}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not infer ranking CSV. Expected one of:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_ranked_source_islands(
    ranking_csv: Path,
    source_codes: List[int],
) -> List[int]:
    df = pd.read_csv(ranking_csv)
    if "source_island" not in df.columns:
        raise ValueError(f"Ranking CSV missing 'source_island': {ranking_csv}")

    if "phi_per_ind_mean" in df.columns:
        score_col = "phi_per_ind_mean"
    elif "phi_per_individual" in df.columns:
        score_col = "phi_per_individual"
    elif "phi" in df.columns:
        score_col = "phi"
    else:
        raise ValueError(
            f"Ranking CSV missing expected score columns in {ranking_csv}"
        )

    rank_df = df[["source_island", score_col]].copy()
    rank_df["source_island"] = rank_df["source_island"].astype(int)
    rank_df = rank_df[rank_df["source_island"].isin(source_codes)]
    rank_df = rank_df.sort_values(score_col, ascending=False)

    ranked = rank_df["source_island"].tolist()
    if not ranked:
        raise ValueError("No ranked source islands found after filtering.")
    return ranked


def compute_loss_curves(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_params: TrainParams,
    batch_size: int,
    device: torch.device,
    seed: int,
    use_early_stopping: bool = False,
    patience: int = 20,
) -> pd.DataFrame:
    set_seed(seed)

    loss_fn = make_loss(train_params.loss_name)
    model = make_model(X_train.shape[1], train_params)
    optimizer = _optimizer(
        train_params.optimizer,
        model.parameters(),
        train_params.lr,
        train_params.weight_decay,
    )

    x_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    x_cal_t = torch.from_numpy(X_cal).float().to(device)
    y_cal_t = torch.from_numpy(y_cal).float().to(device)
    x_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.to(device)
    rows = []
    best_cal_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, train_params.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            n_batches += 1

        train_loss = epoch_loss_sum / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            cal_loss = loss_fn(model(x_cal_t), y_cal_t).item()
            test_loss = loss_fn(model(x_test_t), y_test_t).item()

        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "cal_loss": float(cal_loss),
                "test_loss": float(test_loss),
            }
        )

        if use_early_stopping:
            if cal_loss < best_cal_loss:
                best_cal_loss = cal_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch} (patience={patience}, best_cal_loss={best_cal_loss:.6f})"
                )
                break

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss curves by number of source islands")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument(
        "--target_island",
        default=None,
        help="Target island (encoded code, original label, or name). Defaults to first in config.",
    )
    parser.add_argument(
        "--island_counts",
        nargs="+",
        type=int,
        default=[1, 3, 7, 10, 15],
        help="Island counts to evaluate",
    )
    parser.add_argument(
        "--ranking_csv",
        default=None,
        help="Optional path to ranking CSV. If omitted, inferred from output_dir/island_<target>/...",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Split/training base seed override (default: config seed)",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping based on calibration loss",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience in epochs (default: 20)",
    )
    args = parser.parse_args()

    if args.patience < 1:
        raise ValueError("--patience must be >= 1")

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    paths = cfg["paths"]
    output_dir_base = paths.get("output_dir", "outputs/tmc_shapley")

    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    base_seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)

    X, y, ids, _GRM, locality, code_to_label, y_eval = load_data(
        paths=paths,
        target_column=cfg.get("target_column", "y_adjusted"),
        standardize_features=cfg.get("standardize_features", True),
        return_locality=True,
        min_count=cfg.get("min_count", 20),
        return_eval=True,
        eval_target_column=cfg.get("eval_target_column", "y_mean"),
    )

    present_codes = set(int(c) for c in np.unique(locality))
    raw_targets = cfg.get("target_islands", [0])
    target_raw = args.target_island if args.target_island is not None else raw_targets[0]
    target_code = resolve_island_code(target_raw, code_to_label, present_codes)
    target_name = island_label(target_code, code_to_label)

    included_raw = cfg.get("included_islands", None)
    if included_raw is not None:
        included_island_codes = [
            resolve_island_code(v, code_to_label, present_codes)
            for v in included_raw
        ]
    else:
        included_island_codes = sorted(present_codes)
    source_codes = [c for c in included_island_codes if c != target_code]

    ranking_csv = Path(args.ranking_csv) if args.ranking_csv else infer_ranking_csv(output_dir_base, target_code)
    logger.info(f"Using ranking CSV: {ranking_csv}")
    ranked_source_codes = load_ranked_source_islands(ranking_csv, source_codes)

    split = split_target_island(
        X,
        y,
        ids,
        locality,
        target_code,
        y_eval=y_eval,
        cal_fraction=cfg.get("cal_fraction", 0.2),
        seed=base_seed,
        max_cal_fraction=cfg.get("max_cal_fraction", None),
    )

    X_source = split["X_source"]
    y_source = split["y_source"]
    locality_source = split["locality_source"]
    X_cal = split["X_cal"]
    y_cal = split["y_cal"]
    X_test = split["X_test"]
    y_test = split["y_test"]

    in_source = np.isin(locality_source, source_codes)
    X_source = X_source[in_source]
    y_source = y_source[in_source]
    locality_source = locality_source[in_source]

    tp_cfg = cfg["train_params"]
    train_params = TrainParams(
        lr=tp_cfg["lr"],
        weight_decay=tp_cfg.get("weight_decay", 1e-5),
        epochs=tp_cfg["epochs"],
        loss_name=tp_cfg.get("loss_name", "mse"),
        optimizer=tp_cfg.get("optimizer", "adam"),
        hidden_dims=tp_cfg.get("hidden_dims", [512, 64]),
        dropout=tp_cfg.get("dropout", 0.5),
        batch_norm=tp_cfg.get("batch_norm", False),
    )
    batch_size = int(tp_cfg.get("batch_size", 64))

    use_snp_selection = bool(cfg.get("use_snp_selection", False))
    num_snps = cfg.get("num_snps", None)
    snp_selection_mode = str(cfg.get("snp_selection_mode", "random")).lower()

    all_curves = []
    summary_rows = []

    requested_counts = [int(k) for k in args.island_counts if int(k) > 0]
    requested_counts = sorted(set(requested_counts))

    for k_req in requested_counts:
        k = min(k_req, len(ranked_source_codes))
        selected_islands = ranked_source_codes[:k]
        m = np.isin(locality_source, selected_islands)

        X_train = X_source[m]
        y_train = y_source[m]

        if len(X_train) < 2:
            logger.warning(f"Skipping k={k_req}: too few training samples ({len(X_train)})")
            continue

        snp_cols = None
        if use_snp_selection and num_snps is not None and int(num_snps) < X_train.shape[1]:
            if snp_selection_mode == "random":
                rng = np.random.default_rng(base_seed + 10000 + k)
                snp_cols = rng.choice(X_train.shape[1], size=int(num_snps), replace=False)
            else:
                snp_cols = _select_top_snps_by_abs_corr(X_train, y_train, int(num_snps))

        if snp_cols is not None:
            X_train_fit = X_train[:, snp_cols]
            X_cal_fit = X_cal[:, snp_cols]
            X_test_fit = X_test[:, snp_cols]
        else:
            X_train_fit = X_train
            X_cal_fit = X_cal
            X_test_fit = X_test

        logger.info(
            f"Training k={k_req} (effective k={k}) islands, n_train={len(X_train_fit)}, "
            f"n_cal={len(X_cal_fit)}, n_test={len(X_test_fit)}"
        )

        curve_df = compute_loss_curves(
            X_train=X_train_fit,
            y_train=y_train,
            X_cal=X_cal_fit,
            y_cal=y_cal,
            X_test=X_test_fit,
            y_test=y_test,
            train_params=train_params,
            batch_size=batch_size,
            device=device,
            seed=base_seed + 100 * k,
            use_early_stopping=args.early_stopping,
            patience=args.patience,
        )
        curve_df["n_islands"] = int(k_req)
        curve_df["effective_n_islands"] = int(k)
        curve_df["n_train"] = int(len(X_train_fit))
        all_curves.append(curve_df)

        best_idx = int(curve_df["cal_loss"].idxmin())
        summary_rows.append(
            {
                "n_islands": int(k_req),
                "effective_n_islands": int(k),
                "n_train": int(len(X_train_fit)),
                "best_cal_epoch": int(curve_df.loc[best_idx, "epoch"]),
                "best_cal_loss": float(curve_df.loc[best_idx, "cal_loss"]),
                "test_loss_at_best_cal": float(curve_df.loc[best_idx, "test_loss"]),
                "final_epoch": int(curve_df["epoch"].iloc[-1]),
                "final_cal_loss": float(curve_df["cal_loss"].iloc[-1]),
                "final_test_loss": float(curve_df["test_loss"].iloc[-1]),
                "used_early_stopping": bool(args.early_stopping),
                "patience": int(args.patience if args.early_stopping else 0),
                "stopped_early": bool(int(curve_df["epoch"].iloc[-1]) < int(train_params.epochs)),
            }
        )

    if not all_curves:
        raise RuntimeError("No curves were computed. Check island counts and data availability.")

    curves_df = pd.concat(all_curves, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("n_islands")

    out_dir = Path(output_dir_base) / f"island_{target_code}" / "loss_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_es{args.patience}" if args.early_stopping else ""
    csv_path = out_dir / f"loss_curves_seed_{base_seed}{suffix}.csv"
    summary_path = out_dir / f"loss_curve_summary_seed_{base_seed}{suffix}.csv"
    png_path = out_dir / f"loss_curves_seed_{base_seed}{suffix}.png"

    curves_df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    metrics = [
        ("train_loss", "Train loss"),
        ("cal_loss", "Calibration loss"),
        ("test_loss", "Test loss"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for k in sorted(curves_df["n_islands"].unique().tolist()):
            d = curves_df[curves_df["n_islands"] == k]
            ax.plot(d["epoch"], d[metric], label=f"k={k}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Loss")
    axes[-1].legend(title="Islands", fontsize=9)
    fig.suptitle(f"Loss curves by island count — Target {target_code} ({target_name})", y=1.02)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved curves CSV: {csv_path}")
    logger.info(f"Saved summary CSV: {summary_path}")
    logger.info(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
