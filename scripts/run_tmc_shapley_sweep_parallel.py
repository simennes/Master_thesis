#!/usr/bin/env python
"""
Parallel TMC-Shapley sweep runner (new script).

This script supports two modes:
1) worker: run a shard of repeats (for SLURM array jobs)
2) merge : combine shard outputs and build final summaries/plots

Designed to keep the original sweep script unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.models import TrainParams
from src.removal_curve import split_target_island
from src.tmc_shapley_islands import ShapleyConfig, run_shapley_experiment
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


def _as_float_list(v: Any, default: List[float]) -> List[float]:
    if v is None:
        return default
    if isinstance(v, list):
        return [float(x) for x in v]
    return [float(v)]


def _as_int_list(v: Any, default: List[int]) -> List[int]:
    if v is None:
        return default
    if isinstance(v, list):
        return [int(x) for x in v]
    return [int(v)]


def _make_repeat_seed(global_seed: int, target_code: int, cal_frac: float, repeat_idx: int) -> int:
    token = f"{int(global_seed)}|{int(target_code)}|{float(cal_frac):.12g}|{int(repeat_idx)}"
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % 2_147_483_647


def _plot_remove_curve_grid(
    remove_summary: pd.DataFrame,
    n_perm_grid: List[int],
    cal_frac_grid: List[float],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    methods = ["shapley_mean", "random_individual"]
    n_rows = len(n_perm_grid)
    n_cols = len(cal_frac_grid)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for i, n_perm in enumerate(n_perm_grid):
        for j, cal_frac in enumerate(cal_frac_grid):
            ax = axes[i, j]
            cell = remove_summary[
                (remove_summary["n_permutations"] == n_perm)
                & (np.isclose(remove_summary["cal_fraction"], cal_frac))
            ]
            if cell.empty:
                ax.set_title(f"perm={n_perm}, cal={cal_frac:.2f}\n(no data)")
                ax.grid(True, alpha=0.3)
                continue

            for method in methods:
                sub = cell[cell["method"] == method].sort_values("n_removed")
                if sub.empty:
                    continue
                y = sub["corr_mean"].to_numpy(dtype=float)
                yerr = sub["corr_std"].fillna(0.0).to_numpy(dtype=float)
                x = sub["n_removed"].to_numpy(dtype=int)
                if method == "shapley_mean":
                    ax.plot(x, y, color="#2A9D8F", marker="o", linewidth=1.8, label="Shapley")
                    ax.fill_between(x, y - yerr, y + yerr, color="#2A9D8F", alpha=0.12)
                else:
                    ax.plot(x, y, color="#E76F51", linestyle="--", linewidth=1.4, label="Random")
                    ax.fill_between(x, y - yerr, y + yerr, color="#E76F51", alpha=0.10)

            ax.set_title(f"perm={n_perm}, cal={cal_frac:.2f}")
            ax.grid(True, alpha=0.3)
            if i == n_rows - 1:
                ax.set_xlabel("n_removed")
            if j == 0:
                ax.set_ylabel("corr_eval")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Remove curves by n_permutations x cal_fraction", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_gain_3d_surface(gain_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if gain_df.empty:
        return

    col = "gain_over_baseline"
    if col not in gain_df.columns:
        col = "avg_gain_over_v_full"  # fallback for old data

    x_vals = sorted(gain_df["n_permutations"].unique().tolist())
    y_vals = sorted(gain_df["cal_fraction"].unique().tolist())

    z_mat = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
    for yi, yv in enumerate(y_vals):
        for xi, xv in enumerate(x_vals):
            row = gain_df[(gain_df["n_permutations"] == xv) & (np.isclose(gain_df["cal_fraction"], yv))]
            if not row.empty:
                z_mat[yi, xi] = float(row.iloc[0][col])

    X, Y = np.meshgrid(np.array(x_vals, dtype=float), np.array(y_vals, dtype=float))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    vmin, vmax = np.nanmin(z_mat), np.nanmax(z_mat)
    if vmin < 0 < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = "RdYlGn"
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = "viridis"

    if np.isfinite(z_mat).all():
        surf = ax.plot_surface(X, Y, z_mat, cmap=cmap, norm=norm, edgecolor="grey",
                               linewidth=0.3, alpha=0.92, antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.10, label="gain (corr_best − corr_0)")
    else:
        valid = gain_df[np.isfinite(gain_df[col])].copy()
        sc = ax.scatter(
            valid["n_permutations"], valid["cal_fraction"], valid[col],
            c=valid[col], cmap=cmap, norm=norm, s=60, edgecolors="k", linewidths=0.5,
        )
        fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.10, label="gain (corr_best − corr_0)")

    ax.set_xlabel("n_permutations", labelpad=10)
    ax.set_ylabel("cal_fraction", labelpad=10)
    ax.set_zlabel("gain over baseline", labelpad=8)
    ax.set_title("Gain: best remove-curve point vs k=0\n(test-set metric only)", pad=16)
    ax.view_init(elev=28, azim=-52)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_common(config_path: Path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    sweep_cfg = cfg.get("sweep", {})
    n_perm_grid = _as_int_list(
        sweep_cfg.get("n_permutations", cfg.get("tmc", {}).get("n_permutations", 50)),
        default=[50],
    )
    cal_frac_grid = _as_float_list(
        sweep_cfg.get("cal_fractions", cfg.get("cal_fraction", 0.2)),
        default=[0.2],
    )
    n_perm_grid = list(dict.fromkeys(int(x) for x in n_perm_grid))
    cal_frac_grid = list(dict.fromkeys(float(x) for x in cal_frac_grid))

    if any(x < 1 for x in n_perm_grid):
        raise ValueError("All sweep n_permutations must be >= 1")
    if any((x <= 0.0 or x >= 1.0) for x in cal_frac_grid):
        raise ValueError("All sweep cal_fractions must be in (0, 1)")

    return cfg, sweep_cfg, n_perm_grid, cal_frac_grid


def run_worker(config_path: Path, shard_index: int, num_shards: int, target_override: list[str] | None) -> None:
    cfg, sweep_cfg, n_perm_grid, cal_frac_grid = _load_common(config_path)

    paths = cfg["paths"]
    base_output = Path(paths.get("output_dir", "outputs/tmc_shapley"))
    sweep_output = Path(sweep_cfg.get("output_dir", str(base_output / "sweep_parallel")))
    save_cell_artifacts = bool(sweep_cfg.get("save_cell_artifacts", False))

    shard_dir = sweep_output / "shards" / f"shard_{shard_index:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    global_seed = int(cfg.get("seed", 42))
    set_seed(global_seed)

    logger.info("Loading data...")
    X, y, ids, _, locality, code_to_label, y_eval = load_data(
        paths=paths,
        target_column=cfg.get("target_column", "y_adjusted"),
        standardize_features=cfg.get("standardize_features", True),
        return_locality=True,
        min_count=cfg.get("min_count", 20),
        return_eval=True,
        eval_target_column=cfg.get("eval_target_column", "y_mean"),
    )
    logger.info(
        f"Data loaded: {X.shape[0]} samples, {X.shape[1]} SNPs, {len(np.unique(locality))} islands"
    )

    present_codes = set(int(c) for c in np.unique(locality))

    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "mlp")).lower()
    if model_type not in {"mlp", "ridge"}:
        raise ValueError("model.type must be one of ['mlp', 'ridge']")

    if model_type == "mlp":
        tp_cfg = cfg.get("train_params")
        if tp_cfg is None:
            raise ValueError("train_params is required when model.type='mlp'")
        train_params = TrainParams(
            lr=tp_cfg["lr"],
            weight_decay=tp_cfg.get("weight_decay", 1e-5),
            epochs=tp_cfg["epochs"],
            loss_name=tp_cfg.get("loss_name", "mse"),
            optimizer=tp_cfg.get("optimizer", "adam"),
            hidden_dims=tp_cfg.get("hidden_dims", [512, 64]),
            dropout=tp_cfg.get("dropout", 0.3),
            batch_norm=tp_cfg.get("batch_norm", False),
        )
        batch_size = tp_cfg.get("batch_size", cfg.get("batch_size", 64))
    else:
        train_params = None
        batch_size = int(cfg.get("batch_size", 64))

    tmc_cfg_raw = cfg.get("tmc", {})
    base_shapley_cfg = ShapleyConfig(
        n_permutations=int(tmc_cfg_raw.get("n_permutations", 50)),
        eps_trunc=float(tmc_cfg_raw.get("eps_trunc", 0.005)),
        min_prefix_islands=int(tmc_cfg_raw.get("min_prefix_islands", 3)),
        use_truncation=bool(tmc_cfg_raw.get("use_truncation", False)),
        island_sampling_mode=tmc_cfg_raw.get("island_sampling_mode", "mean"),
        cap_per_island=tmc_cfg_raw.get("cap_per_island", None),
        n_train_seeds_per_eval=int(tmc_cfg_raw.get("n_train_seeds_per_eval", 1)),
        cache_dir=tmc_cfg_raw.get("cache_dir", None),
        n_random_orders=int(cfg.get("baselines", {}).get("n_random_orders", 1)),
        use_early_stopping=bool(tmc_cfg_raw.get("use_early_stopping", False)),
        early_stopping_patience=int(tmc_cfg_raw.get("early_stopping_patience", 0)),
        retrain_use_early_stopping=bool(tmc_cfg_raw.get("retrain_use_early_stopping", True)),
        retrain_early_stopping_patience=int(tmc_cfg_raw.get("retrain_early_stopping_patience", 20)),
        snp_selection_mode=cfg.get("snp_selection_mode", "random"),
        use_snp_selection=bool(cfg.get("use_snp_selection", False)),
        num_snps=cfg.get("num_snps", None),
        model_type=model_type,
        ridge_alpha=float(model_cfg.get("alpha", 1e5)),
        seed=global_seed,
    )

    uncertainty_cfg = cfg.get("uncertainty", {})
    n_repeats = int(sweep_cfg.get("n_target_split_repeats", uncertainty_cfg.get("n_target_split_repeats", 5)))
    if n_repeats < 1:
        raise ValueError("sweep.n_target_split_repeats must be >= 1")

    repeat_indices = [r for r in range(n_repeats) if (r % num_shards) == shard_index]
    if len(repeat_indices) == 0:
        logger.warning("Shard %d/%d has no assigned repeats. Exiting.", shard_index, num_shards)
        return

    included_raw = cfg.get("included_islands", None)
    if included_raw is not None:
        included_island_codes = [resolve_island_code(v, code_to_label, present_codes) for v in included_raw]
    else:
        included_island_codes = sorted(present_codes)

    raw_targets = target_override if target_override else cfg.get("target_islands", [0])
    target_codes = [resolve_island_code(t, code_to_label, present_codes) for t in raw_targets]

    logger.info(
        "Worker shard=%d/%d | repeats=%s | perms=%s | cal_fracs=%s | targets=%s",
        shard_index,
        num_shards,
        repeat_indices,
        n_perm_grid,
        cal_frac_grid,
        target_codes,
    )

    # Fixed SNP subset for whole experiment (same across all shards)
    fixed_snp_cols: np.ndarray | None = None
    use_snp_selection = bool(cfg.get("use_snp_selection", False))
    num_snps = cfg.get("num_snps", None)
    snp_mode = str(cfg.get("snp_selection_mode", "random")).lower()
    if use_snp_selection and num_snps is not None and int(num_snps) < int(X.shape[1]):
        if snp_mode != "random":
            raise ValueError("Parallel sweep currently supports fixed global SNP subset only for snp_selection_mode='random'.")
        rng_snp = np.random.default_rng(global_seed)
        fixed_snp_cols = np.sort(rng_snp.choice(X.shape[1], size=int(num_snps), replace=False).astype(np.int64))
        np.save(shard_dir / "fixed_snp_columns.npy", fixed_snp_cols)

    # Precompute split plans for this shard's repeats
    max_cal_fraction = cfg.get("max_cal_fraction", None)
    split_index_map: Dict[tuple, Dict[str, np.ndarray]] = {}
    split_plan_rows: List[Dict[str, Any]] = []

    for target_code in target_codes:
        target_mask = (locality == target_code)
        ids_target = ids[target_mask]
        n_target = int(target_mask.sum())
        for cal_frac in cal_frac_grid:
            n_cal = max(1, int(cal_frac * n_target))
            for repeat_idx in repeat_indices:
                repeat_seed = (
                    _make_repeat_seed(
                        global_seed=global_seed,
                        target_code=int(target_code),
                        cal_frac=float(cal_frac),
                        repeat_idx=int(repeat_idx),
                    )
                )
                rng = np.random.default_rng(repeat_seed)
                perm = rng.permutation(n_target)
                if max_cal_fraction is not None:
                    n_cal_max = max(1, int(float(max_cal_fraction) * n_target))
                    if n_cal > n_cal_max:
                        raise ValueError(
                            f"cal_fraction={cal_frac} exceeds max_cal_fraction={max_cal_fraction}"
                        )
                    cal_idx = perm[:n_cal]
                    test_idx = perm[n_cal_max:]
                else:
                    cal_idx = perm[:n_cal]
                    test_idx = perm[n_cal:]

                split_index_map[(int(target_code), float(cal_frac), int(repeat_idx))] = {
                    "cal_idx": cal_idx,
                    "test_idx": test_idx,
                    "repeat_seed": int(repeat_seed),
                }

                for rid in ids_target[cal_idx]:
                    split_plan_rows.append(
                        {
                            "target_island": int(target_code),
                            "cal_fraction": float(cal_frac),
                            "repeat": int(repeat_idx),
                            "split_seed": int(repeat_seed),
                            "subset": "cal",
                            "ringnr": str(rid),
                        }
                    )
                for rid in ids_target[test_idx]:
                    split_plan_rows.append(
                        {
                            "target_island": int(target_code),
                            "cal_fraction": float(cal_frac),
                            "repeat": int(repeat_idx),
                            "split_seed": int(repeat_seed),
                            "subset": "test",
                            "ringnr": str(rid),
                        }
                    )

    if split_plan_rows:
        pd.DataFrame(split_plan_rows).to_csv(shard_dir / "target_split_plan.csv", index=False)

    remove_rows: List[pd.DataFrame] = []
    run_rows: List[Dict[str, Any]] = []

    total_jobs = len(target_codes) * len(cal_frac_grid) * len(n_perm_grid) * len(repeat_indices)
    done_jobs = 0

    for target_code in target_codes:
        target_name = island_label(target_code, code_to_label)
        source_codes = [c for c in included_island_codes if c != target_code]
        if len(source_codes) == 0:
            logger.warning(f"No source islands for target {target_code}, skipping")
            continue
        source_names = {c: island_label(c, code_to_label) for c in source_codes}

        for cal_frac in cal_frac_grid:
            for n_perm in n_perm_grid:
                for repeat_idx in repeat_indices:
                    split_key = (int(target_code), float(cal_frac), int(repeat_idx))
                    split_plan = split_index_map[split_key]
                    repeat_seed = int(split_plan["repeat_seed"])

                    repeat_cfg = copy.deepcopy(base_shapley_cfg)
                    repeat_cfg.n_permutations = int(n_perm)
                    repeat_cfg.seed = int(repeat_seed)

                    if save_cell_artifacts:
                        run_output_dir = (
                            shard_dir
                            / f"target_{target_code}"
                            / f"perm_{n_perm}"
                            / f"cal_{cal_frac:.3f}"
                            / f"repeat_{repeat_idx}"
                        )
                    else:
                        run_output_dir = shard_dir / "tmp"

                    logger.info(
                        "Shard %d/%d progress %d/%d | target=%s cal=%.3f perm=%d repeat=%d",
                        shard_index,
                        num_shards,
                        done_jobs + 1,
                        total_jobs,
                        target_code,
                        cal_frac,
                        n_perm,
                        repeat_idx,
                    )

                    try:
                        result = run_shapley_experiment(
                            X=X,
                            y=y,
                            ids=ids,
                            locality=locality,
                            y_eval=y_eval,
                            target_island_code=target_code,
                            target_island_name=target_name,
                            source_codes=source_codes,
                            source_names=source_names,
                            train_params=train_params,
                            cfg=repeat_cfg,
                            output_dir=str(run_output_dir),
                            device=device,
                            batch_size=batch_size,
                            cal_fraction=float(cal_frac),
                            max_cal_fraction=cfg.get("max_cal_fraction", None),
                            use_snp_selection=cfg.get("use_snp_selection", False),
                            num_snps=cfg.get("num_snps", None),
                            snp_selection_mode=cfg.get("snp_selection_mode", "random"),
                            cal_idx=np.asarray(split_plan["cal_idx"], dtype=np.int64),
                            test_idx=np.asarray(split_plan["test_idx"], dtype=np.int64),
                            snp_cols_override=fixed_snp_cols,
                        )
                    except Exception as e:
                        logger.exception(
                            "Failed cell target=%s perm=%s cal=%s repeat=%s: %s",
                            target_code,
                            n_perm,
                            cal_frac,
                            repeat_idx,
                            e,
                        )
                        done_jobs += 1
                        continue

                    remove_df = result["remove_curve_df"].copy()
                    remove_df["target_island"] = int(target_code)
                    remove_df["target_island_name"] = str(target_name)
                    remove_df["repeat"] = int(repeat_idx)
                    remove_df["split_seed"] = int(repeat_seed)
                    remove_df["n_permutations"] = int(n_perm)
                    remove_df["cal_fraction"] = float(cal_frac)
                    remove_rows.append(remove_df)

                    run_rows.append(
                        {
                            "target_island": int(target_code),
                            "target_island_name": str(target_name),
                            "repeat": int(repeat_idx),
                            "split_seed": int(repeat_seed),
                            "n_permutations": int(n_perm),
                            "cal_fraction": float(cal_frac),
                            "v_full": float(result["v_full"]),
                            "n_source_islands": int(len(source_codes)),
                        }
                    )

                    done_jobs += 1

    if len(remove_rows) == 0:
        logger.warning("No successful rows produced by shard %d", shard_index)
        return

    remove_out = pd.concat(remove_rows, ignore_index=True)
    runs_out = pd.DataFrame(run_rows)
    remove_out.to_csv(shard_dir / "remove_curve_rows.csv", index=False)
    runs_out.to_csv(shard_dir / "run_metadata.csv", index=False)

    with open(shard_dir / "shard_summary.json", "w") as f:
        json.dump(
            {
                "shard_index": int(shard_index),
                "num_shards": int(num_shards),
                "n_rows_remove": int(len(remove_out)),
                "n_rows_runs": int(len(runs_out)),
                "repeat_indices": [int(r) for r in repeat_indices],
            },
            f,
            indent=2,
        )

    logger.info("Shard %d complete. Output: %s", shard_index, shard_dir)


def run_merge(config_path: Path) -> None:
    cfg, sweep_cfg, n_perm_grid, cal_frac_grid = _load_common(config_path)

    paths = cfg["paths"]
    base_output = Path(paths.get("output_dir", "outputs/tmc_shapley"))
    sweep_output = Path(sweep_cfg.get("output_dir", str(base_output / "sweep_parallel")))

    shards_root = sweep_output / "shards"
    shard_dirs = sorted([p for p in shards_root.glob("shard_*") if p.is_dir()])
    if len(shard_dirs) == 0:
        raise RuntimeError(f"No shard directories found in {shards_root}")

    remove_parts = []
    run_parts = []

    for shard in shard_dirs:
        remove_path = shard / "remove_curve_rows.csv"
        run_path = shard / "run_metadata.csv"
        if remove_path.exists():
            remove_parts.append(pd.read_csv(remove_path))
        if run_path.exists():
            run_parts.append(pd.read_csv(run_path))

    if len(remove_parts) == 0 or len(run_parts) == 0:
        raise RuntimeError("No shard output CSVs found to merge.")

    remove_all_df = pd.concat(remove_parts, ignore_index=True)
    runs_df = pd.concat(run_parts, ignore_index=True)

    remove_all_path = sweep_output / "remove_curve_sweep_rows.csv"
    runs_path = sweep_output / "sweep_run_metadata.csv"
    remove_all_df.to_csv(remove_all_path, index=False)
    runs_df.to_csv(runs_path, index=False)

    remove_summary = (
        remove_all_df.groupby(["n_permutations", "cal_fraction", "method", "n_removed"], as_index=False)
        .agg(
            corr_mean=("corr_eval", "mean"),
            corr_std=("corr_eval", "std"),
            mse_mean=("mse_adj", "mean"),
            mse_std=("mse_adj", "std"),
            n_rows=("corr_eval", "size"),
        )
    )
    remove_summary_path = sweep_output / "remove_curve_sweep_summary.csv"
    remove_summary.to_csv(remove_summary_path, index=False)

    shapley_summary = remove_summary[remove_summary["method"] == "shapley_mean"].copy()
    idx_best = shapley_summary.groupby(["n_permutations", "cal_fraction"])["corr_mean"].idxmax()
    best_rows = shapley_summary.loc[idx_best].copy()

    # Baseline: n_removed=0 from the same remove curve (test-set metric)
    baseline_rows = (
        shapley_summary[shapley_summary["n_removed"] == 0]
        .rename(columns={"corr_mean": "corr_baseline"})
        [["n_permutations", "cal_fraction", "corr_baseline"]]
    )

    vfull_summary = (
        runs_df.groupby(["n_permutations", "cal_fraction"], as_index=False)
        .agg(v_full_mean=("v_full", "mean"), n_runs=("v_full", "size"), n_source_islands=("n_source_islands", "mean"))
    )

    gain_df = best_rows.merge(vfull_summary, on=["n_permutations", "cal_fraction"], how="left")
    gain_df = gain_df.merge(baseline_rows, on=["n_permutations", "cal_fraction"], how="left")
    gain_df["best_n_removed"] = gain_df["n_removed"].astype(int)
    gain_df["best_n_islands"] = (gain_df["n_source_islands"].round().astype(int) - gain_df["best_n_removed"]).clip(lower=1)
    gain_df["best_corr_mean"] = gain_df["corr_mean"]
    gain_df["gain_over_baseline"] = gain_df["best_corr_mean"] - gain_df["corr_baseline"]

    gain_cols = [
        "n_permutations",
        "cal_fraction",
        "best_n_removed",
        "best_n_islands",
        "best_corr_mean",
        "corr_baseline",
        "v_full_mean",
        "gain_over_baseline",
        "n_runs",
    ]
    gain_df = gain_df[gain_cols].sort_values(["n_permutations", "cal_fraction"]).reset_index(drop=True)

    gain_path = sweep_output / "gain_over_baseline_surface_data.csv"
    gain_df.to_csv(gain_path, index=False)

    grid_plot_path = sweep_output / "remove_curve_grid.png"
    _plot_remove_curve_grid(remove_summary, n_perm_grid=n_perm_grid, cal_frac_grid=cal_frac_grid, output_path=grid_plot_path)

    surface_plot_path = sweep_output / "gain_over_baseline_surface_3d.png"
    _plot_gain_3d_surface(gain_df, output_path=surface_plot_path)

    summary_payload = {
        "n_shards_found": int(len(shard_dirs)),
        "n_permutations": [int(x) for x in n_perm_grid],
        "cal_fractions": [float(x) for x in cal_frac_grid],
        "n_successful_runs": int(len(runs_df)),
        "outputs": {
            "remove_curve_sweep_rows": str(remove_all_path),
            "remove_curve_sweep_summary": str(remove_summary_path),
            "sweep_run_metadata": str(runs_path),
            "gain_over_baseline_surface_data": str(gain_path),
            "remove_curve_grid_plot": str(grid_plot_path),
            "gain_surface_3d_plot": str(surface_plot_path),
        },
    }
    with open(sweep_output / "sweep_summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2)

    logger.info("Merge complete. Final outputs in %s", sweep_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel TMC-Shapley sweep runner")
    parser.add_argument("--config", required=True, help="Path to sweep config JSON")
    parser.add_argument("--mode", choices=["worker", "merge"], default="worker")
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--target_islands", nargs="+", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if args.mode == "merge":
        run_merge(config_path)
        return

    shard_index = args.shard_index
    if shard_index is None:
        shard_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    num_shards = args.num_shards
    if num_shards is None:
        env_n = os.environ.get("SWEEP_NUM_SHARDS")
        num_shards = int(env_n) if env_n is not None else 1

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}]")

    run_worker(config_path, shard_index=shard_index, num_shards=num_shards, target_override=args.target_islands)


if __name__ == "__main__":
    main()
