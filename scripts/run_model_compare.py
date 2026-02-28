#!/usr/bin/env python
"""
Compare MLP vs Ridge regression on target-island prediction.

Protocol
--------
- Train on source islands only.
- Test on the full selected target island(s) (no target calibration split).
- Evaluate multiple random source-removal fractions to compare performance
  as training set size decreases.

Ridge lambda (alpha) is selected automatically with source-only CV (cv.glmnet-like).

Usage
-----
python scripts/run_model_compare.py --config config/model_compare_config.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.models import TrainParams, make_model
from src.training import evaluate_model, train_simple
from src.utils import _optimizer, _select_top_snps_by_abs_corr, make_loss, set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _save_plots(summary_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"Could not import matplotlib for plotting: {e}")
        return

    if summary_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    metrics = [
        ("corr_eval_mean", "corr_eval_std", "Pearson r", "Accuracy (corr) vs removal"),
        ("mse_adj_mean", "mse_adj_std", "MSE", "Error (MSE) vs removal"),
        ("fit_time_mean_s", "fit_time_std_s", "Fit time (s)", "Runtime vs removal"),
    ]

    for ax, (mean_col, std_col, ylabel, title) in zip(axes, metrics):
        for model in sorted(summary_df["model"].unique().tolist()):
            d = summary_df[summary_df["model"] == model].sort_values("removal_fraction")
            if d.empty:
                continue
            x = d["removal_fraction"].to_numpy()
            y = d[mean_col].to_numpy()
            e = d[std_col].fillna(0.0).to_numpy()
            ax.plot(x, y, marker="o", label=model)
            ax.fill_between(x, y - e, y + e, alpha=0.15)
        ax.set_xlabel("Removal fraction")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[-1].legend(title="Model", fontsize=9)
    plt.tight_layout()
    out_path = output_dir / "model_compare_overview.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison plot: {out_path}")


def _save_alpha_diagnostics_plots(alpha_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"Could not import matplotlib for alpha diagnostics plotting: {e}")
        return

    try:
        if alpha_df.empty:
            return

        agg = (
            alpha_df.groupby("alpha", as_index=False)
            .agg(
                cv_mse_mean=("cv_mse", "mean"),
                cv_mse_std=("cv_mse", "std"),
                cv_time_mean_s=("cv_time_seconds", "mean"),
                cv_time_std_s=("cv_time_seconds", "std"),
                n_trials=("cv_mse", "size"),
            )
            .sort_values("alpha")
        )

        # Keep only finite, strictly positive alphas for log-scale plotting
        agg = agg[np.isfinite(agg["alpha"]) & (agg["alpha"] > 0)].copy()
        if agg.empty:
            logger.warning("Skipping ridge alpha diagnostics plot: no positive finite alpha values.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        x = agg["alpha"].to_numpy(dtype=float)

        y1 = agg["cv_mse_mean"].to_numpy(dtype=float)
        e1 = agg["cv_mse_std"].fillna(0.0).to_numpy(dtype=float)
        axes[0].plot(x, y1, marker="o")
        axes[0].fill_between(x, y1 - e1, y1 + e1, alpha=0.15)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Alpha (lambda)")
        axes[0].set_ylabel("CV MSE")
        axes[0].set_title("Ridge alpha vs CV MSE")
        axes[0].grid(True, alpha=0.3)

        y2 = agg["cv_time_mean_s"].to_numpy(dtype=float)
        e2 = agg["cv_time_std_s"].fillna(0.0).to_numpy(dtype=float)
        axes[1].plot(x, y2, marker="o")
        axes[1].fill_between(x, y2 - e2, y2 + e2, alpha=0.15)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Alpha (lambda)")
        axes[1].set_ylabel("CV time (s)")
        axes[1].set_title("Ridge alpha vs CV runtime")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / "ridge_alpha_diagnostics.png"
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved ridge alpha diagnostics plot: {out_path}")
    except Exception as e:
        logger.warning(f"Could not generate ridge alpha diagnostics plot: {e}")


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


def _build_alpha_grid(alpha_cfg: Dict) -> np.ndarray:
    if "alphas" in alpha_cfg and alpha_cfg["alphas"] is not None:
        vals = np.array(alpha_cfg["alphas"], dtype=float)
        if np.any(vals <= 0):
            raise ValueError("All explicit ridge alphas must be > 0")
        return np.unique(np.sort(vals))

    a_min = float(alpha_cfg.get("alpha_min", 1e-4))
    a_max = float(alpha_cfg.get("alpha_max", 1e4))
    n_alphas = int(alpha_cfg.get("n_alphas", 50))
    if a_min <= 0 or a_max <= 0 or a_max <= a_min:
        raise ValueError("Invalid alpha range for ridge")
    if n_alphas < 2:
        raise ValueError("n_alphas must be >= 2")
    return np.logspace(math.log10(a_min), math.log10(a_max), n_alphas)


def fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    source_groups: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    ridge_cfg: Dict,
    seed: int,
) -> Dict:
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold, LeaveOneGroupOut
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("scikit-learn is required for ridge comparison") from e

    mode = str(ridge_cfg.get("alpha_selection", "cv_source")).lower()
    if mode != "cv_source":
        raise ValueError("With no target calibration set, ridge.alpha_selection must be 'cv_source'.")

    alphas = _build_alpha_grid(ridge_cfg)
    using_explicit_alpha_list = ridge_cfg.get("alphas", None) is not None

    # In high-dimensional settings (p >> n), very small alphas can cause
    # ill-conditioned/singular systems in CV folds. Apply a safe floor.
    alpha_floor_cfg = ridge_cfg.get("alpha_floor_high_dim", 1e-2)
    alpha_floor = float(alpha_floor_cfg) if alpha_floor_cfg is not None else None
    if alpha_floor is not None and X_train.shape[1] > X_train.shape[0]:
        n_before = len(alphas)
        alphas = alphas[alphas >= alpha_floor]
        if len(alphas) == 0:
            alphas = np.array([alpha_floor], dtype=float)
        if len(alphas) != n_before:
            logger.info(
                "Ridge alpha grid clipped for high-dimensional regime (p=%d, n=%d): "
                "kept %d/%d alphas with alpha >= %.3g",
                X_train.shape[1], X_train.shape[0], len(alphas), n_before, alpha_floor,
            )

    logger.info(
        "Ridge alpha grid (%s): n=%d, min=%.6g, max=%.6g",
        "explicit list" if using_explicit_alpha_list else "logspace",
        len(alphas),
        float(np.min(alphas)),
        float(np.max(alphas)),
    )

    # Fast path: if user gave exactly one explicit alpha, skip CV entirely
    if using_explicit_alpha_list and len(alphas) == 1:
        alpha_best = float(alphas[0])
        logger.info("Ridge single alpha provided (%.6g): skipping CV and using this value directly.", alpha_best)

        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=alpha_best)),
        ])

        suppress_solver_warnings = bool(ridge_cfg.get("suppress_solver_warnings", True))
        if suppress_solver_warnings:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Ill-conditioned matrix.*")
                warnings.filterwarnings("ignore", message=".*Singular matrix.*")
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        pred_test = model.predict(X_test)
        mse_adj = float(np.mean((pred_test - y_test) ** 2))

        corr_eval = 0.0
        if len(pred_test) >= 2:
            p = pred_test - pred_test.mean()
            t = y_eval_test - y_eval_test.mean()
            denom = np.sqrt((p * p).sum()) * np.sqrt((t * t).sum())
            if denom > 1e-12:
                corr_eval = float((p * t).sum() / denom)

        alpha_diagnostics = [{
            "alpha": alpha_best,
            "cv_mse": float("nan"),
            "cv_time_seconds": 0.0,
            "n_folds_scored": 0,
        }]

        return {
            "model": "ridge",
            "alpha_best": alpha_best,
            "alpha_cv_mse_best": float("nan"),
            "corr_eval": float(corr_eval),
            "mse_adj": float(mse_adj),
            "alpha_diagnostics": alpha_diagnostics,
        }

    cv_strategy = str(ridge_cfg.get("cv_strategy", "loio_source")).lower()
    n_folds = int(ridge_cfg.get("cv_folds", 5))

    if cv_strategy == "loio_source":
        unique_groups = np.unique(source_groups)
        if len(unique_groups) < 2:
            logger.warning(
                "Ridge LOIO requested but only %d source island group(s) available; falling back to KFold.",
                len(unique_groups),
            )
            cv = KFold(n_splits=min(max(2, n_folds), len(X_train)), shuffle=True, random_state=seed)
            split_iter = list(cv.split(X_train))
            cv_used = f"kfold_fallback_{len(split_iter)}"
        else:
            logo = LeaveOneGroupOut()
            split_iter = list(logo.split(X_train, y_train, groups=source_groups))
            cv_used = f"loio_source_{len(split_iter)}"
    elif cv_strategy == "kfold":
        cv = KFold(n_splits=min(max(2, n_folds), len(X_train)), shuffle=True, random_state=seed)
        split_iter = list(cv.split(X_train))
        cv_used = f"kfold_{len(split_iter)}"
    else:
        raise ValueError("ridge.cv_strategy must be 'loio_source' or 'kfold'")

    logger.info("Ridge alpha CV strategy: %s", cv_used)

    best_alpha = None
    best_cv_mse = float("inf")
    alpha_diagnostics = []

    suppress_solver_warnings = bool(ridge_cfg.get("suppress_solver_warnings", True))

    log_alpha_search = bool(ridge_cfg.get("log_alpha_search", True))

    for alpha in alphas:
        t_alpha_start = time.perf_counter()
        fold_mses = []
        for tr_idx, va_idx in split_iter:
            if len(np.unique(y_train[tr_idx])) < 2 or len(va_idx) == 0:
                continue

            model_fold = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=float(alpha))),
            ])

            if suppress_solver_warnings:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Ill-conditioned matrix.*")
                    warnings.filterwarnings("ignore", message=".*Singular matrix.*")
                    model_fold.fit(X_train[tr_idx], y_train[tr_idx])
            else:
                model_fold.fit(X_train[tr_idx], y_train[tr_idx])

            pred_va = model_fold.predict(X_train[va_idx])
            fold_mses.append(float(np.mean((pred_va - y_train[va_idx]) ** 2)))

        if len(fold_mses) == 0:
            alpha_diagnostics.append(
                {
                    "alpha": float(alpha),
                    "cv_mse": float("nan"),
                    "cv_time_seconds": float(time.perf_counter() - t_alpha_start),
                    "n_folds_scored": 0,
                }
            )
            continue
        cv_mse = float(np.mean(fold_mses))
        cv_time_s = float(time.perf_counter() - t_alpha_start)
        alpha_diagnostics.append(
            {
                "alpha": float(alpha),
                "cv_mse": cv_mse,
                "cv_time_seconds": cv_time_s,
                "n_folds_scored": int(len(fold_mses)),
            }
        )
        if log_alpha_search:
            logger.info(
                "Ridge alpha trial | alpha=%.6g cv_mse=%.6f cv_time=%.3fs folds=%d",
                float(alpha),
                cv_mse,
                cv_time_s,
                len(fold_mses),
            )
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_alpha = float(alpha)

    if best_alpha is None:
        best_alpha = float(alphas[len(alphas) // 2])
        logger.warning("Ridge CV failed to score folds; using median alpha=%.6g", best_alpha)

    alpha_best = float(best_alpha)
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=alpha_best)),
    ])
    if suppress_solver_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Ill-conditioned matrix.*")
            warnings.filterwarnings("ignore", message=".*Singular matrix.*")
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    mse_adj = float(np.mean((pred_test - y_test) ** 2))

    # keep metric definition aligned with existing pipeline
    corr_eval = 0.0
    if len(pred_test) >= 2:
        p = pred_test - pred_test.mean()
        t = y_eval_test - y_eval_test.mean()
        denom = np.sqrt((p * p).sum()) * np.sqrt((t * t).sum())
        if denom > 1e-12:
            corr_eval = float((p * t).sum() / denom)

    return {
        "model": "ridge",
        "alpha_best": alpha_best,
        "alpha_cv_mse_best": float(best_cv_mse),
        "corr_eval": float(corr_eval),
        "mse_adj": float(mse_adj),
        "alpha_diagnostics": alpha_diagnostics,
    }


def fit_predict_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_eval_test: np.ndarray,
    train_params: TrainParams,
    mlp_cfg: Dict,
    device: torch.device,
    batch_size: int,
    seed: int,
    verbose_training: bool = False,
) -> Dict:
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
    x_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float()
    y_eval_test_t = torch.from_numpy(y_eval_test).float()

    use_es = bool(mlp_cfg.get("use_early_stopping", False))
    if use_es:
        logger.warning("mlp.use_early_stopping=true but no calibration set is used; early stopping is disabled.")

    model, epochs_trained = train_simple(
        model=model,
        x_train=x_train_t,
        y_train=y_train_t,
        epochs=train_params.epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        batch_size=batch_size,
        seed=seed,
        verbose=verbose_training,
        patience=0,
        restore_best=False,
    )

    corr_eval, mse_adj = evaluate_model(
        model,
        x_test_t,
        y_test_t,
        device,
        y_eval=y_eval_test_t,
    )

    return {
        "model": "mlp",
        "epochs_trained": int(epochs_trained),
        "corr_eval": float(corr_eval),
        "mse_adj": float(mse_adj),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MLP vs Ridge on target islands")
    parser.add_argument("--config", required=True, help="Path to model comparison config JSON")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    paths = cfg["paths"]
    output_dir = Path(cfg.get("output", {}).get("output_dir", "outputs/model_compare"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    global_seed = int(cfg.get("seed", 42))
    set_seed(global_seed)

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
    target_raw = cfg.get("target_islands", [0])
    target_codes = [resolve_island_code(v, code_to_label, present_codes) for v in target_raw]

    included_raw = cfg.get("included_islands", None)
    if included_raw is not None:
        included_island_codes = [
            resolve_island_code(v, code_to_label, present_codes) for v in included_raw
        ]
    else:
        included_island_codes = sorted(present_codes)

    tp_cfg = cfg["mlp"]["train_params"]
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

    mlp_enabled = bool(cfg.get("mlp", {}).get("enabled", True))
    ridge_enabled = bool(cfg.get("ridge", {}).get("enabled", True))
    if not mlp_enabled and not ridge_enabled:
        raise ValueError("Both mlp.enabled and ridge.enabled are false; enable at least one model.")
    logger.info("Model switches | mlp.enabled=%s ridge.enabled=%s", mlp_enabled, ridge_enabled)

    use_snp_selection = bool(cfg.get("use_snp_selection", False))
    num_snps = cfg.get("num_snps", None)
    snp_selection_mode = str(cfg.get("snp_selection_mode", "random")).lower()

    n_repeats = int(cfg.get("evaluation", {}).get("n_repeats", 1))
    seed_stride = int(cfg.get("evaluation", {}).get("seed_stride", 1000))
    removal_fractions = cfg.get("evaluation", {}).get("removal_fractions", [0.0])
    removal_fractions = [float(r) for r in removal_fractions]
    verbose_training = bool(cfg.get("logging", {}).get("mlp_verbose_training", False))
    log_every_fraction = int(cfg.get("logging", {}).get("log_every_fraction", 1))
    if log_every_fraction < 1:
        log_every_fraction = 1
    for rf in removal_fractions:
        if rf < 0.0 or rf >= 1.0:
            raise ValueError("evaluation.removal_fractions must be in [0.0, 1.0)")

    total_steps = len(target_codes) * n_repeats * len(removal_fractions)
    step_counter = 0

    rows = []
    alpha_diag_rows = []

    for target_code in target_codes:
        target_name = island_label(target_code, code_to_label)

        target_mask = locality == target_code
        source_mask = ~target_mask

        source_codes = [c for c in included_island_codes if c != target_code]
        if len(source_codes) == 0:
            logger.warning(f"No source islands for target {target_code}; skipping")
            continue

        source_mask = source_mask & np.isin(locality, source_codes)

        X_source_all = X[source_mask]
        y_source_all = y[source_mask]
        locality_source_all = locality[source_mask]

        X_test = X[target_mask]
        y_test = y[target_mask]
        y_eval_test = y_eval[target_mask]

        if len(X_source_all) < 2 or len(X_test) < 2:
            logger.warning(
                f"Insufficient samples for target {target_code}: "
                f"n_source={len(X_source_all)}, n_test={len(X_test)}"
            )
            continue

        logger.info(
            f"Target {target_code} ({target_name}) | "
            f"n_source_full={len(X_source_all)}, n_test={len(X_test)}"
        )

        for rep in range(n_repeats):
            run_seed = global_seed + rep * seed_stride
            rng = np.random.default_rng(run_seed)
            logger.info(
                "Starting repeat %d/%d for target %s (seed=%d)",
                rep + 1,
                n_repeats,
                target_code,
                run_seed,
            )

            for frac_idx, rf in enumerate(removal_fractions):
                step_counter += 1
                n_total = len(X_source_all)
                n_keep = max(2, int(round((1.0 - rf) * n_total)))
                keep_idx = rng.choice(n_total, size=n_keep, replace=False)

                X_source = X_source_all[keep_idx]
                y_source = y_source_all[keep_idx]
                locality_source = locality_source_all[keep_idx]

                if use_snp_selection and num_snps is not None and int(num_snps) < X_source.shape[1]:
                    if snp_selection_mode == "random":
                        snp_rng = np.random.default_rng(run_seed + int(rf * 1_000_000))
                        snp_cols = snp_rng.choice(X_source.shape[1], size=int(num_snps), replace=False)
                    else:
                        snp_cols = _select_top_snps_by_abs_corr(X_source, y_source, int(num_snps))
                    X_source_fit = X_source[:, snp_cols]
                    X_test_fit = X_test[:, snp_cols]
                else:
                    X_source_fit = X_source
                    X_test_fit = X_test

                if (frac_idx % log_every_fraction) == 0:
                    logger.info(
                        "Progress %d/%d | target=%s rep=%d/%d rf=%.2f | n_train=%d n_test=%d",
                        step_counter,
                        total_steps,
                        target_code,
                        rep + 1,
                        n_repeats,
                        rf,
                        len(X_source_fit),
                        len(X_test_fit),
                    )

                mlp_out = None
                if mlp_enabled:
                    logger.info(
                        "Fitting MLP | target=%s rep=%d rf=%.2f",
                        target_code,
                        rep + 1,
                        rf,
                    )

                    t0 = time.perf_counter()
                    mlp_out = fit_predict_mlp(
                        X_train=X_source_fit,
                        y_train=y_source,
                        X_test=X_test_fit,
                        y_test=y_test,
                        y_eval_test=y_eval_test,
                        train_params=train_params,
                        mlp_cfg=cfg.get("mlp", {}),
                        device=device,
                        batch_size=batch_size,
                        seed=run_seed + int(rf * 10_000) + 1,
                        verbose_training=verbose_training,
                    )
                    mlp_time_s = time.perf_counter() - t0
                    logger.info(
                        "Done MLP | target=%s rep=%d rf=%.2f | corr=%.4f mse=%.4f epochs=%d time=%.2fs",
                        target_code,
                        rep + 1,
                        rf,
                        mlp_out["corr_eval"],
                        mlp_out["mse_adj"],
                        mlp_out["epochs_trained"],
                        mlp_time_s,
                    )
                    rows.append(
                        {
                            "target_island": target_code,
                            "target_island_name": target_name,
                            "repeat": rep,
                            "seed": run_seed,
                            "removal_fraction": rf,
                            "n_source_train": int(len(X_source_fit)),
                            "n_test": int(len(X_test_fit)),
                            "fit_time_seconds": float(mlp_time_s),
                            **mlp_out,
                        }
                    )

                ridge_out = None
                if ridge_enabled:
                    logger.info(
                        "Fitting RidgeCV | target=%s rep=%d rf=%.2f",
                        target_code,
                        rep + 1,
                        rf,
                    )

                    t0 = time.perf_counter()
                    ridge_out = fit_predict_ridge(
                        X_train=X_source_fit,
                        y_train=y_source,
                        source_groups=locality_source,
                        X_test=X_test_fit,
                        y_test=y_test,
                        y_eval_test=y_eval_test,
                        ridge_cfg=cfg.get("ridge", {}),
                        seed=run_seed + int(rf * 10_000) + 2,
                    )
                    ridge_alpha_diag = ridge_out.pop("alpha_diagnostics", [])
                    ridge_time_s = time.perf_counter() - t0
                    logger.info(
                        "Done RidgeCV | target=%s rep=%d rf=%.2f | corr=%.4f mse=%.4f alpha=%.6g time=%.2fs",
                        target_code,
                        rep + 1,
                        rf,
                        ridge_out["corr_eval"],
                        ridge_out["mse_adj"],
                        ridge_out["alpha_best"],
                        ridge_time_s,
                    )
                    rows.append(
                        {
                            "target_island": target_code,
                            "target_island_name": target_name,
                            "repeat": rep,
                            "seed": run_seed,
                            "removal_fraction": rf,
                            "n_source_train": int(len(X_source_fit)),
                            "n_test": int(len(X_test_fit)),
                            "fit_time_seconds": float(ridge_time_s),
                            **ridge_out,
                        }
                    )
                    for d in ridge_alpha_diag:
                        alpha_diag_rows.append(
                            {
                                "target_island": target_code,
                                "target_island_name": target_name,
                                "repeat": rep,
                                "seed": run_seed,
                                "removal_fraction": rf,
                                "n_source_train": int(len(X_source_fit)),
                                **d,
                            }
                        )

                if mlp_out is not None and ridge_out is not None:
                    logger.info(
                        f"target={target_code} rep={rep} rf={rf:.2f} | "
                        f"n_train={len(X_source_fit)} | "
                        f"MLP corr={mlp_out['corr_eval']:.4f}, "
                        f"Ridge corr={ridge_out['corr_eval']:.4f}, "
                        f"ridge_alpha={ridge_out['alpha_best']:.6g}"
                    )
                elif mlp_out is not None:
                    logger.info(
                        f"target={target_code} rep={rep} rf={rf:.2f} | "
                        f"n_train={len(X_source_fit)} | "
                        f"MLP corr={mlp_out['corr_eval']:.4f}"
                    )
                elif ridge_out is not None:
                    logger.info(
                        f"target={target_code} rep={rep} rf={rf:.2f} | "
                        f"n_train={len(X_source_fit)} | "
                        f"Ridge corr={ridge_out['corr_eval']:.4f}, "
                        f"ridge_alpha={ridge_out['alpha_best']:.6g}"
                    )

    if len(rows) == 0:
        raise RuntimeError("No comparison results produced.")

    results_df = pd.DataFrame(rows)
    results_path = output_dir / "model_compare_results.csv"
    results_df.to_csv(results_path, index=False)

    summary_df = (
        results_df.groupby([
            "target_island", "target_island_name", "model", "removal_fraction"
        ], as_index=False)
        .agg(
            corr_eval_mean=("corr_eval", "mean"),
            corr_eval_std=("corr_eval", "std"),
            mse_adj_mean=("mse_adj", "mean"),
            mse_adj_std=("mse_adj", "std"),
            fit_time_mean_s=("fit_time_seconds", "mean"),
            fit_time_std_s=("fit_time_seconds", "std"),
            n_source_train_mean=("n_source_train", "mean"),
            n_runs=("corr_eval", "size"),
        )
    )
    summary_path = output_dir / "model_compare_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    _save_plots(summary_df, output_dir)

    if len(alpha_diag_rows) > 0:
        alpha_df = pd.DataFrame(alpha_diag_rows)
        alpha_path = output_dir / "ridge_alpha_diagnostics.csv"
        alpha_df.to_csv(alpha_path, index=False)

        alpha_summary_df = (
            alpha_df.groupby(["alpha", "removal_fraction"], as_index=False)
            .agg(
                cv_mse_mean=("cv_mse", "mean"),
                cv_mse_std=("cv_mse", "std"),
                cv_time_mean_s=("cv_time_seconds", "mean"),
                cv_time_std_s=("cv_time_seconds", "std"),
                n_trials=("cv_mse", "size"),
            )
            .sort_values(["removal_fraction", "alpha"])
        )
        alpha_summary_path = output_dir / "ridge_alpha_diagnostics_summary.csv"
        alpha_summary_df.to_csv(alpha_summary_path, index=False)

        _save_alpha_diagnostics_plots(alpha_df, output_dir)
        logger.info(f"Saved ridge alpha diagnostics: {alpha_path}")
        logger.info(f"Saved ridge alpha diagnostics summary: {alpha_summary_path}")

    logger.info(f"Saved detailed results: {results_path}")
    logger.info(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
