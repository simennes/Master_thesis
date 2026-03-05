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


def _build_trait_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            "target_column": t.get("target_column", cfg.get("target_column", "y_adjusted")),
            "eval_target_column": t.get("eval_target_column", cfg.get("eval_target_column", "y_mean")),
            "standardize_features": t.get("standardize_features", cfg.get("standardize_features", True)),
            "min_count": int(t.get("min_count", cfg.get("min_count", 20))),
        })
    return specs


def _plot_remove_curve_grid(
    remove_summary: pd.DataFrame,
    n_perm_grid: List[int],
    cal_values: List[float],
    cal_key: str,
    cal_label: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    methods = ["shapley_mean", "random_individual"]
    n_rows = len(n_perm_grid)
    n_cols = len(cal_values)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for i, n_perm in enumerate(n_perm_grid):
        for j, cal_val in enumerate(cal_values):
            ax = axes[i, j]
            cell = remove_summary[
                (remove_summary["n_permutations"] == n_perm)
                & (np.isclose(remove_summary[cal_key], cal_val))
            ]
            if cell.empty:
                if cal_key == "n_cal_samples":
                    cal_txt = f"n_cal={int(round(cal_val))}"
                else:
                    cal_txt = f"cal={cal_val:.2f}"
                ax.set_title(f"perm={n_perm}, {cal_txt}\n(no data)")
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

            if cal_key == "n_cal_samples":
                cal_txt = f"n_cal={int(round(cal_val))}"
            else:
                cal_txt = f"cal={cal_val:.2f}"
            ax.set_title(f"perm={n_perm}, {cal_txt}")
            ax.grid(True, alpha=0.3)
            if i == n_rows - 1:
                ax.set_xlabel("n_removed")
            if j == 0:
                ax.set_ylabel("corr_eval")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Remove curves by n_permutations x {cal_label}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_gain_3d_surface(gain_df: pd.DataFrame, output_path: Path, cal_key: str, cal_label: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if gain_df.empty:
        return

    col = "avg_gain_over_v_full"

    x_vals = sorted(gain_df["n_permutations"].unique().tolist())
    y_vals = sorted(gain_df[cal_key].unique().tolist())

    z_mat = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
    for yi, yv in enumerate(y_vals):
        for xi, xv in enumerate(x_vals):
            row = gain_df[(gain_df["n_permutations"] == xv) & (np.isclose(gain_df[cal_key], yv))]
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
            valid["n_permutations"], valid[cal_key], valid[col],
            c=valid[col], cmap=cmap, norm=norm, s=60, edgecolors="k", linewidths=0.5,
        )
        fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.10, label="gain (corr_best − corr_0)")

    ax.set_xlabel("n_permutations", labelpad=10)
    ax.set_ylabel(cal_label, labelpad=10)
    ax.set_zlabel("avg gain over v_full", labelpad=8)
    ax.set_title("Best average gain over v_full by sweep cell", pad=16)
    ax.view_init(elev=28, azim=-52)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _cal_value_text(cal_key: str, cal_val: float) -> str:
    if cal_key == "n_cal_samples":
        return f"n_cal={int(round(cal_val))}"
    return f"cal={float(cal_val):.2f}"


def _cal_value_token(cal_key: str, cal_val: float) -> str:
    if cal_key == "n_cal_samples":
        return f"ncal_{int(round(cal_val))}"
    return f"cal_{float(cal_val):.3f}".replace(".", "p")


def _plot_running_rank_heatmap(
    rank_summary_df: pd.DataFrame,
    output_path: Path,
    selected_permutations: List[int],
    target_name: str,
    cal_text: str,
) -> None:
    import matplotlib.pyplot as plt

    if rank_summary_df.empty:
        return

    max_perm = int(rank_summary_df["permutation_index"].max())
    final_slice = rank_summary_df[rank_summary_df["permutation_index"] == max_perm].copy()
    if final_slice.empty:
        return

    final_slice = final_slice.sort_values("phi_per_ind_mean", ascending=False)
    ordered_codes = [int(x) for x in final_slice["source_island"].tolist()]

    heat = rank_summary_df.pivot_table(
        index="permutation_index",
        columns="source_island",
        values="rank_mean",
        aggfunc="mean",
    ).sort_index(axis=0)

    heat = heat.reindex(columns=ordered_codes)
    if heat.empty:
        return

    label_map = (
        rank_summary_df[["source_island", "source_island_name"]]
        .drop_duplicates()
        .set_index("source_island")["source_island_name"]
        .to_dict()
    )
    x_labels = [str(label_map.get(c, c)) for c in ordered_codes]

    n_rows, n_cols = heat.shape
    fig_w = max(10.0, 0.7 * n_cols + 2.5)
    fig_h = max(5.0, 0.06 * n_rows + 3.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    im = ax.imshow(
        heat.to_numpy(dtype=float),
        aspect="auto",
        cmap="viridis_r",
        vmin=1,
        vmax=max(1, n_cols),
        origin="lower",
    )

    ax.set_title(f"Running rank (phi/n) | target={target_name}, {cal_text}")
    ax.set_xlabel("source island (ordered by final mean phi/n)")
    ax.set_ylabel("permutation index")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)

    y_vals = heat.index.to_numpy(dtype=int)
    if len(y_vals) <= 15:
        y_tick_vals = y_vals.tolist()
    else:
        y_tick_vals = sorted(set(y_vals[:: max(1, len(y_vals) // 12)].tolist() + [int(y_vals[-1])]))
    y_lookup = {int(v): idx for idx, v in enumerate(y_vals)}
    y_tick_pos = [y_lookup[v] for v in y_tick_vals if v in y_lookup]
    ax.set_yticks(y_tick_pos)
    ax.set_yticklabels([str(v) for v in y_tick_vals])

    for p in sorted(set(int(x) for x in selected_permutations)):
        if p in y_lookup:
            row_idx = y_lookup[p]
            ax.axhline(row_idx - 0.5, color="white", linestyle="--", linewidth=1.2, alpha=0.9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("mean running rank (1 = most important)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_common(config_path: Path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    sweep_cfg = cfg.get("sweep", {})
    n_perm_grid = _as_int_list(
        sweep_cfg.get("n_permutations", cfg.get("tmc", {}).get("n_permutations", 50)),
        default=[50],
    )
    cal_samples_grid_raw = sweep_cfg.get("n_cal_samples", None)
    if cal_samples_grid_raw is not None:
        cal_mode = "samples"
        cal_values = list(dict.fromkeys(int(x) for x in _as_int_list(cal_samples_grid_raw, default=[])))
        if any(x < 1 for x in cal_values):
            raise ValueError("All sweep n_cal_samples must be >= 1")
        cal_key = "n_cal_samples"
        cal_label = "n_cal_samples"
    else:
        cal_mode = "fraction"
        cal_values = list(dict.fromkeys(float(x) for x in _as_float_list(
            sweep_cfg.get("cal_fractions", cfg.get("cal_fraction", 0.2)),
            default=[0.2],
        )))
        if any((x <= 0.0 or x >= 1.0) for x in cal_values):
            raise ValueError("All sweep cal_fractions must be in (0, 1)")
        cal_key = "cal_fraction"
        cal_label = "cal_fraction"

    n_perm_grid = list(dict.fromkeys(int(x) for x in n_perm_grid))

    if any(x < 1 for x in n_perm_grid):
        raise ValueError("All sweep n_permutations must be >= 1")

    return cfg, sweep_cfg, n_perm_grid, cal_mode, cal_values, cal_key, cal_label


def run_worker(config_path: Path, shard_index: int, num_shards: int, target_override: list[str] | None) -> None:
    cfg, sweep_cfg, n_perm_grid, cal_mode, cal_values, cal_key, cal_label = _load_common(config_path)

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

    all_trait_specs = _build_trait_specs(cfg)
    n_traits = len(all_trait_specs)

    # --- Determine sharding mode: repeat-only vs repeat+trait ----------------
    if num_shards <= n_repeats:
        # Classic mode: shard only across repeats
        repeat_indices = [r for r in range(n_repeats) if (r % num_shards) == shard_index]
        trait_specs = all_trait_specs  # all traits
        logger.info(
            "Sharding mode: repeat-only (num_shards=%d <= n_repeats=%d). "
            "This shard handles repeats=%s, all %d traits.",
            num_shards, n_repeats, repeat_indices, n_traits,
        )
    else:
        # Extended mode: shard across repeats AND traits
        trait_multiplier = num_shards // n_repeats
        if num_shards != n_repeats * trait_multiplier:
            raise ValueError(
                f"num_shards ({num_shards}) must be divisible by n_repeats ({n_repeats}) "
                f"when num_shards > n_repeats"
            )
        if n_traits % trait_multiplier != 0:
            raise ValueError(
                f"num_shards/n_repeats = {trait_multiplier}, but the number of traits "
                f"({n_traits}) is not divisible by {trait_multiplier}. "
                f"Choose num_shards that is a multiple of n_repeats and whose quotient "
                f"divides the number of traits."
            )
        traits_per_shard = n_traits // trait_multiplier
        # Decompose shard_index into (trait_group, repeat_shard)
        # shard layout: shard_index = repeat_shard * trait_multiplier + trait_group
        # so repeat_shards 0..n_repeats-1 each have trait_multiplier sub-shards
        repeat_shard = shard_index // trait_multiplier
        trait_group = shard_index % trait_multiplier
        trait_start = trait_group * traits_per_shard
        trait_end = trait_start + traits_per_shard
        trait_specs = all_trait_specs[trait_start:trait_end]
        repeat_indices = [r for r in range(n_repeats) if (r % n_repeats) == repeat_shard]
        logger.info(
            "Sharding mode: repeat+trait (num_shards=%d, n_repeats=%d, "
            "trait_multiplier=%d). This shard handles repeats=%s, traits=%s.",
            num_shards, n_repeats, trait_multiplier,
            repeat_indices, [t["name"] for t in trait_specs],
        )

    if len(repeat_indices) == 0:
        logger.warning("Shard %d/%d has no assigned repeats. Exiting.", shard_index, num_shards)
        return
    if len(trait_specs) == 0:
        logger.warning("Shard %d/%d has no assigned traits. Exiting.", shard_index, num_shards)
        return

    shard_summary: Dict[str, Any] = {
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "repeat_indices": [int(r) for r in repeat_indices],
        "traits": {},
    }

    for trait_spec in trait_specs:
        trait_name = trait_spec["name"]
        trait_dir = shard_dir / trait_name
        trait_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading data for trait '%s'...", trait_name)
        X, y, ids, _, locality, code_to_label, y_eval = load_data(
            paths=trait_spec["paths"],
            target_column=trait_spec["target_column"],
            standardize_features=trait_spec["standardize_features"],
            return_locality=True,
            min_count=trait_spec["min_count"],
            return_eval=True,
            eval_target_column=trait_spec["eval_target_column"],
        )
        logger.info(
            "Trait '%s': %d samples, %d SNPs, %d islands",
            trait_name,
            X.shape[0],
            X.shape[1],
            len(np.unique(locality)),
        )

        present_codes = set(int(c) for c in np.unique(locality))

        included_raw = cfg.get("included_islands", None)
        if included_raw is not None:
            included_island_codes = [resolve_island_code(v, code_to_label, present_codes) for v in included_raw]
        else:
            included_island_codes = sorted(present_codes)

        raw_targets = target_override if target_override else cfg.get("target_islands", [0])
        target_codes = [resolve_island_code(t, code_to_label, present_codes) for t in raw_targets]

        logger.info(
            "Worker shard=%d/%d trait=%s | repeats=%s | perms=%s | %s=%s | targets=%s",
            shard_index,
            num_shards,
            trait_name,
            repeat_indices,
            n_perm_grid,
            cal_label,
            cal_values,
            target_codes,
        )

        fixed_snp_cols: np.ndarray | None = None
        use_snp_selection = bool(cfg.get("use_snp_selection", False))
        num_snps = cfg.get("num_snps", None)
        snp_mode = str(cfg.get("snp_selection_mode", "random")).lower()
        if use_snp_selection and num_snps is not None and int(num_snps) < int(X.shape[1]):
            if snp_mode != "random":
                raise ValueError("Parallel sweep currently supports fixed global SNP subset only for snp_selection_mode='random'.")
            rng_snp = np.random.default_rng(global_seed)
            fixed_snp_cols = np.sort(rng_snp.choice(X.shape[1], size=int(num_snps), replace=False).astype(np.int64))
            np.save(trait_dir / "fixed_snp_columns.npy", fixed_snp_cols)

        max_cal_fraction_cfg = cfg.get("max_cal_fraction", None)
        split_index_map: Dict[tuple, Dict[str, np.ndarray]] = {}
        split_plan_rows: List[Dict[str, Any]] = []

        for target_code in target_codes:
            target_mask = (locality == target_code)
            ids_target = ids[target_mask]
            n_target = int(target_mask.sum())

            if cal_mode == "samples":
                max_n_cal_samples = int(max(cal_values))
                if max_n_cal_samples >= n_target:
                    raise ValueError(f"max n_cal_samples ({max_n_cal_samples}) must be < n_target ({n_target}) for target {target_code}")
                max_cal_fraction_eff = float(max_n_cal_samples / n_target)
            else:
                max_cal_fraction_eff = float(max_cal_fraction_cfg) if max_cal_fraction_cfg is not None else None

            for cal_val in cal_values:
                if cal_mode == "samples":
                    n_cal = int(cal_val)
                    if n_cal >= n_target:
                        raise ValueError(f"n_cal_samples ({n_cal}) must be < n_target ({n_target}) for target {target_code}")
                    cal_frac = float(n_cal / n_target)
                else:
                    cal_frac = float(cal_val)
                    n_cal = max(1, int(cal_frac * n_target))

                for repeat_idx in repeat_indices:
                    repeat_seed = _make_repeat_seed(
                        global_seed=global_seed,
                        target_code=int(target_code),
                        cal_frac=float(cal_frac),
                        repeat_idx=int(repeat_idx),
                    )
                    rng = np.random.default_rng(repeat_seed)
                    perm = rng.permutation(n_target)

                    if max_cal_fraction_eff is not None:
                        n_cal_max = max(1, int(float(max_cal_fraction_eff) * n_target))
                        if n_cal > n_cal_max:
                            raise ValueError(f"cal setting n_cal={n_cal} exceeds max n_cal={n_cal_max}")
                        cal_idx = perm[:n_cal]
                        test_idx = perm[n_cal_max:]
                    else:
                        cal_idx = perm[:n_cal]
                        test_idx = perm[n_cal:]

                    split_index_map[(int(target_code), float(cal_val), int(repeat_idx))] = {
                        "cal_idx": cal_idx,
                        "test_idx": test_idx,
                        "repeat_seed": int(repeat_seed),
                        "cal_fraction": float(cal_frac),
                        "n_cal_samples": int(n_cal),
                    }

                    for rid in ids_target[cal_idx]:
                        split_plan_rows.append({
                            "trait": trait_name,
                            "target_island": int(target_code),
                            "cal_fraction": float(cal_frac),
                            "n_cal_samples": int(n_cal),
                            "repeat": int(repeat_idx),
                            "split_seed": int(repeat_seed),
                            "subset": "cal",
                            "ringnr": str(rid),
                        })
                    for rid in ids_target[test_idx]:
                        split_plan_rows.append({
                            "trait": trait_name,
                            "target_island": int(target_code),
                            "cal_fraction": float(cal_frac),
                            "n_cal_samples": int(n_cal),
                            "repeat": int(repeat_idx),
                            "split_seed": int(repeat_seed),
                            "subset": "test",
                            "ringnr": str(rid),
                        })

        if split_plan_rows:
            pd.DataFrame(split_plan_rows).to_csv(trait_dir / "target_split_plan.csv", index=False)

        remove_rows: List[pd.DataFrame] = []
        run_rows: List[Dict[str, Any]] = []
        rank_rows: List[Dict[str, Any]] = []

        total_jobs = len(target_codes) * len(cal_values) * len(n_perm_grid) * len(repeat_indices)
        done_jobs = 0

        for target_code in target_codes:
            target_name = island_label(target_code, code_to_label)
            source_codes = [c for c in included_island_codes if c != target_code]
            if len(source_codes) == 0:
                logger.warning(f"No source islands for target {target_code}, skipping")
                continue
            source_names = {c: island_label(c, code_to_label) for c in source_codes}

            for cal_val in cal_values:
                for n_perm in n_perm_grid:
                    for repeat_idx in repeat_indices:
                        split_key = (int(target_code), float(cal_val), int(repeat_idx))
                        split_plan = split_index_map[split_key]
                        repeat_seed = int(split_plan["repeat_seed"])

                        repeat_cfg = copy.deepcopy(base_shapley_cfg)
                        repeat_cfg.n_permutations = int(n_perm)
                        repeat_cfg.seed = int(repeat_seed)

                        # Make cache_dir trait-specific to avoid collisions
                        # across traits that share the same target/seed combo.
                        if repeat_cfg.cache_dir:
                            repeat_cfg.cache_dir = os.path.join(
                                repeat_cfg.cache_dir, trait_name
                            )

                        if save_cell_artifacts:
                            run_output_dir = (
                                trait_dir
                                / f"target_{target_code}"
                                / f"perm_{n_perm}"
                                / f"cal_{split_plan['cal_fraction']:.3f}"
                                / f"repeat_{repeat_idx}"
                            )
                        else:
                            run_output_dir = trait_dir / "tmp"

                        logger.info(
                            "Shard %d/%d trait=%s progress %d/%d | target=%s %s=%s perm=%d repeat=%d",
                            shard_index,
                            num_shards,
                            trait_name,
                            done_jobs + 1,
                            total_jobs,
                            target_code,
                            cal_label,
                            int(cal_val) if cal_mode == "samples" else f"{float(cal_val):.3f}",
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
                                cal_fraction=float(split_plan["cal_fraction"]),
                                max_cal_fraction=float(split_plan["cal_fraction"]) if cal_mode == "samples" else cfg.get("max_cal_fraction", None),
                                use_snp_selection=cfg.get("use_snp_selection", False),
                                num_snps=cfg.get("num_snps", None),
                                snp_selection_mode=cfg.get("snp_selection_mode", "random"),
                                cal_idx=np.asarray(split_plan["cal_idx"], dtype=np.int64),
                                test_idx=np.asarray(split_plan["test_idx"], dtype=np.int64),
                                snp_cols_override=fixed_snp_cols,
                            )
                        except Exception as e:
                            logger.exception(
                                "Failed cell trait=%s target=%s perm=%s %s=%s repeat=%s: %s",
                                trait_name,
                                target_code,
                                n_perm,
                                cal_label,
                                cal_val,
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
                        remove_df["trait"] = trait_name
                        remove_df["cal_fraction"] = float(split_plan["cal_fraction"])
                        remove_df["n_cal_samples"] = int(split_plan["n_cal_samples"])
                        remove_rows.append(remove_df)

                        run_rows.append(
                            {
                                "target_island": int(target_code),
                                "target_island_name": str(target_name),
                                "repeat": int(repeat_idx),
                                "split_seed": int(repeat_seed),
                                "n_permutations": int(n_perm),
                                "trait": trait_name,
                                "cal_fraction": float(split_plan["cal_fraction"]),
                                "n_cal_samples": int(split_plan["n_cal_samples"]),
                                "v_full": float(result["v_full"]),
                                "n_source_islands": int(len(source_codes)),
                            }
                        )

                        inter_rows = result.get("tmc_stats", {}).get("intermediate_rankings", [])
                        if inter_rows:
                            source_codes_arr = [int(c) for c in result.get("source_codes", [])]
                            n_ind_arr = np.asarray(result.get("n_individuals", []), dtype=float)
                            n_map = {
                                int(code): max(1.0, float(n_ind_arr[idx]))
                                for idx, code in enumerate(source_codes_arr)
                                if idx < len(n_ind_arr)
                            }
                            for row in inter_rows:
                                source_island = int(row.get("source_island"))
                                phi_running = float(row.get("phi_running", np.nan))
                                if not np.isfinite(phi_running):
                                    continue
                                n_src = n_map.get(source_island, 1.0)
                                rank_rows.append(
                                    {
                                        "trait": trait_name,
                                        "target_island": int(target_code),
                                        "target_island_name": str(target_name),
                                        "repeat": int(repeat_idx),
                                        "split_seed": int(repeat_seed),
                                        "n_permutations": int(n_perm),
                                        "permutation_index": int(row.get("permutation_index", 0)),
                                        cal_key: float(cal_val),
                                        "cal_fraction": float(split_plan["cal_fraction"]),
                                        "n_cal_samples": int(split_plan["n_cal_samples"]),
                                        "source_island": int(source_island),
                                        "source_island_name": str(source_names.get(source_island, source_island)),
                                        "phi_running": float(phi_running),
                                        "phi_per_ind_running": float(phi_running / n_src),
                                        "rank_running": int(row.get("rank_running", 0)),
                                    }
                                )

                        done_jobs += 1

        if len(remove_rows) == 0:
            logger.warning("No successful rows produced by shard %d for trait '%s'", shard_index, trait_name)
            continue

        remove_out = pd.concat(remove_rows, ignore_index=True)
        runs_out = pd.DataFrame(run_rows)
        rank_out = pd.DataFrame(rank_rows)
        remove_out.to_csv(trait_dir / "remove_curve_rows.csv", index=False)
        runs_out.to_csv(trait_dir / "run_metadata.csv", index=False)
        if not rank_out.empty:
            rank_out.to_csv(trait_dir / "running_rank_rows.csv", index=False)

        shard_summary["traits"][trait_name] = {
            "n_rows_remove": int(len(remove_out)),
            "n_rows_runs": int(len(runs_out)),
            "n_rows_running_rank": int(len(rank_out)),
        }

    with open(shard_dir / "shard_summary.json", "w") as f:
        json.dump(shard_summary, f, indent=2)

    logger.info("Shard %d complete. Output: %s", shard_index, shard_dir)


def run_merge(config_path: Path) -> None:
    cfg, sweep_cfg, n_perm_grid, cal_mode, cal_values, cal_key, cal_label = _load_common(config_path)

    paths = cfg["paths"]
    base_output = Path(paths.get("output_dir", "outputs/tmc_shapley"))
    sweep_output = Path(sweep_cfg.get("output_dir", str(base_output / "sweep_parallel")))

    shards_root = sweep_output / "shards"
    shard_dirs = sorted([p for p in shards_root.glob("shard_*") if p.is_dir()])
    if len(shard_dirs) == 0:
        raise RuntimeError(f"No shard directories found in {shards_root}")

    trait_specs = _build_trait_specs(cfg)
    summary_payload: Dict[str, Any] = {
        "n_shards_found": int(len(shard_dirs)),
        "n_permutations": [int(x) for x in n_perm_grid],
        "cal_mode": cal_mode,
        "cal_values": [float(x) for x in cal_values],
        "traits": {},
    }

    for trait_spec in trait_specs:
        trait_name = trait_spec["name"]
        trait_output = sweep_output / trait_name
        trait_output.mkdir(parents=True, exist_ok=True)

        remove_parts = []
        run_parts = []
        rank_parts = []

        for shard in shard_dirs:
            remove_path = shard / trait_name / "remove_curve_rows.csv"
            run_path = shard / trait_name / "run_metadata.csv"
            rank_path = shard / trait_name / "running_rank_rows.csv"
            if remove_path.exists():
                remove_parts.append(pd.read_csv(remove_path))
            if run_path.exists():
                run_parts.append(pd.read_csv(run_path))
            if rank_path.exists():
                rank_parts.append(pd.read_csv(rank_path))

        if len(remove_parts) == 0 or len(run_parts) == 0:
            logger.warning("No shard output CSVs found to merge for trait '%s'.", trait_name)
            continue

        remove_all_df = pd.concat(remove_parts, ignore_index=True)
        runs_df = pd.concat(run_parts, ignore_index=True)

        remove_all_path = trait_output / "remove_curve_sweep_rows.csv"
        runs_path = trait_output / "sweep_run_metadata.csv"
        remove_all_df.to_csv(remove_all_path, index=False)
        runs_df.to_csv(runs_path, index=False)

        remove_summary = (
            remove_all_df.groupby(["n_permutations", cal_key, "method", "n_removed"], as_index=False)
            .agg(
                corr_mean=("corr_eval", "mean"),
                corr_std=("corr_eval", "std"),
                mse_mean=("mse_adj", "mean"),
                mse_std=("mse_adj", "std"),
                n_rows=("corr_eval", "size"),
            )
        )
        remove_summary_path = trait_output / "remove_curve_sweep_summary.csv"
        remove_summary.to_csv(remove_summary_path, index=False)

        shapley_summary = remove_summary[remove_summary["method"] == "shapley_mean"].copy()
        idx_best = shapley_summary.groupby(["n_permutations", cal_key])["corr_mean"].idxmax()
        best_rows = shapley_summary.loc[idx_best].copy()

        vfull_summary = (
            runs_df.groupby(["n_permutations", cal_key], as_index=False)
            .agg(v_full_mean=("v_full", "mean"), n_runs=("v_full", "size"), n_source_islands=("n_source_islands", "mean"))
        )

        gain_df = best_rows.merge(vfull_summary, on=["n_permutations", cal_key], how="left")
        gain_df["best_n_removed"] = gain_df["n_removed"].astype(int)
        gain_df["best_n_islands"] = (gain_df["n_source_islands"].round().astype(int) - gain_df["best_n_removed"]).clip(lower=1)
        gain_df["best_corr_mean"] = gain_df["corr_mean"]
        gain_df["avg_gain_over_v_full"] = gain_df["best_corr_mean"] - gain_df["v_full_mean"]

        gain_cols = [
            "n_permutations",
            cal_key,
            "best_n_removed",
            "best_n_islands",
            "best_corr_mean",
            "v_full_mean",
            "avg_gain_over_v_full",
            "n_runs",
        ]
        gain_df = gain_df[gain_cols].sort_values(["n_permutations", cal_key]).reset_index(drop=True)

        gain_path = trait_output / "gain_over_vfull_surface_data.csv"
        gain_df.to_csv(gain_path, index=False)

        grid_plot_path = trait_output / "remove_curve_grid.png"
        _plot_remove_curve_grid(
            remove_summary,
            n_perm_grid=n_perm_grid,
            cal_values=[float(x) for x in cal_values],
            cal_key=cal_key,
            cal_label=cal_label,
            output_path=grid_plot_path,
        )

        surface_plot_path = trait_output / "gain_over_vfull_surface_3d.png"
        _plot_gain_3d_surface(gain_df, output_path=surface_plot_path, cal_key=cal_key, cal_label=cal_label)

        rank_summary = pd.DataFrame()
        if rank_parts:
            rank_raw = pd.concat(rank_parts, ignore_index=True)
            rank_summary = (
                rank_raw.groupby(
                    ["target_island", cal_key, "permutation_index", "source_island", "source_island_name"],
                    as_index=False,
                )
                .agg(
                    rank_mean=("rank_running", "mean"),
                    rank_std=("rank_running", "std"),
                    phi_per_ind_mean=("phi_per_ind_running", "mean"),
                    n_repeats=("repeat", "nunique"),
                )
            )

        cache_dir_base = cfg.get("tmc", {}).get("cache_dir", None)
        if rank_summary.empty and cache_dir_base:
            try:
                _, _, _, _, locality_meta, code_to_label_meta, _ = load_data(
                    paths=trait_spec["paths"],
                    target_column=trait_spec["target_column"],
                    standardize_features=trait_spec["standardize_features"],
                    return_locality=True,
                    min_count=trait_spec["min_count"],
                    return_eval=True,
                    eval_target_column=trait_spec["eval_target_column"],
                )
                present_codes = sorted(int(c) for c in np.unique(locality_meta))
                included_raw = cfg.get("included_islands", None)
                if included_raw is not None:
                    included_island_codes = [
                        resolve_island_code(v, code_to_label_meta, set(present_codes))
                        for v in included_raw
                    ]
                else:
                    included_island_codes = present_codes
                island_counts = {
                    int(c): int((locality_meta == c).sum())
                    for c in included_island_codes
                }
                island_names = {
                    int(c): island_label(int(c), code_to_label_meta)
                    for c in included_island_codes
                }

                cache_dir = Path(cache_dir_base) / trait_name
                unique_runs = (
                    runs_df[["target_island", cal_key, "repeat", "split_seed"]]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                running_rows: List[Dict[str, Any]] = []

                for _, run in unique_runs.iterrows():
                    target_code = int(run["target_island"])
                    split_seed = int(run["split_seed"])
                    repeat_idx = int(run["repeat"])
                    cal_val = float(run[cal_key])

                    state_path = cache_dir / f"tmc_perm_state_target_{target_code}_seed_{split_seed}.json"
                    if not state_path.exists():
                        continue

                    with open(state_path, "r") as f:
                        state = json.load(f)

                    phi_list = state.get("local_phi_by_perm", [])
                    if not isinstance(phi_list, list) or len(phi_list) == 0:
                        continue

                    source_codes = [int(c) for c in included_island_codes if int(c) != target_code]
                    n_islands_state = int(state.get("n_islands", len(source_codes)))
                    if n_islands_state != len(source_codes):
                        logger.warning(
                            "Skipping ranking state %s (n_islands=%d does not match expected=%d).",
                            state_path,
                            n_islands_state,
                            len(source_codes),
                        )
                        continue

                    n_source = len(source_codes)
                    n_by_source = np.array([max(1, island_counts.get(c, 1)) for c in source_codes], dtype=float)
                    phi_sum = np.zeros(n_source, dtype=float)

                    for t, arr in enumerate(phi_list, start=1):
                        if arr is None:
                            break
                        local_phi = np.asarray(arr, dtype=float)
                        if local_phi.shape[0] != n_source:
                            logger.warning(
                                "Skipping malformed ranking entry in %s (expected %d islands, got %d).",
                                state_path,
                                n_source,
                                local_phi.shape[0],
                            )
                            phi_sum = None
                            break
                        phi_sum += local_phi
                        phi_running = phi_sum / float(t)
                        phi_per_ind_running = phi_running / n_by_source

                        order = np.argsort(-phi_per_ind_running)
                        ranks = np.empty(n_source, dtype=np.int64)
                        ranks[order] = np.arange(1, n_source + 1)

                        for idx, source_code in enumerate(source_codes):
                            running_rows.append(
                                {
                                    "target_island": int(target_code),
                                    cal_key: float(cal_val),
                                    "repeat": int(repeat_idx),
                                    "split_seed": int(split_seed),
                                    "permutation_index": int(t),
                                    "source_island": int(source_code),
                                    "source_island_name": str(island_names.get(source_code, source_code)),
                                    "phi_per_ind_running": float(phi_per_ind_running[idx]),
                                    "rank_running": int(ranks[idx]),
                                }
                            )

                if running_rows:
                    running_df = pd.DataFrame(running_rows)
                    rank_summary = (
                        running_df.groupby(
                            ["target_island", cal_key, "permutation_index", "source_island", "source_island_name"],
                            as_index=False,
                        )
                        .agg(
                            rank_mean=("rank_running", "mean"),
                            rank_std=("rank_running", "std"),
                            phi_per_ind_mean=("phi_per_ind_running", "mean"),
                            n_repeats=("repeat", "nunique"),
                        )
                    )
            except Exception as e:
                logger.warning(
                    "Could not build running-ranking summaries for trait '%s': %s",
                    trait_name,
                    e,
                )

        per_target_outputs: Dict[str, Any] = {}
        target_codes = sorted(int(x) for x in remove_all_df["target_island"].dropna().unique().tolist())
        for target_code in target_codes:
            target_rows = remove_all_df[remove_all_df["target_island"] == target_code].copy()
            target_runs = runs_df[runs_df["target_island"] == target_code].copy()
            if target_rows.empty or target_runs.empty:
                continue

            target_name = str(
                target_rows["target_island_name"].dropna().iloc[0]
                if "target_island_name" in target_rows.columns and target_rows["target_island_name"].notna().any()
                else target_code
            )

            target_dir = trait_output / f"target_{target_code}"
            target_dir.mkdir(parents=True, exist_ok=True)

            target_rows_path = target_dir / "remove_curve_sweep_rows.csv"
            target_runs_path = target_dir / "sweep_run_metadata.csv"
            target_rows.to_csv(target_rows_path, index=False)
            target_runs.to_csv(target_runs_path, index=False)

            target_summary = (
                target_rows.groupby(["n_permutations", cal_key, "method", "n_removed"], as_index=False)
                .agg(
                    corr_mean=("corr_eval", "mean"),
                    corr_std=("corr_eval", "std"),
                    mse_mean=("mse_adj", "mean"),
                    mse_std=("mse_adj", "std"),
                    n_rows=("corr_eval", "size"),
                )
            )
            target_summary_path = target_dir / "remove_curve_sweep_summary.csv"
            target_summary.to_csv(target_summary_path, index=False)

            shapley_target = target_summary[target_summary["method"] == "shapley_mean"].copy()
            idx_best_target = shapley_target.groupby(["n_permutations", cal_key])["corr_mean"].idxmax()
            best_target = shapley_target.loc[idx_best_target].copy()

            vfull_target = (
                target_runs.groupby(["n_permutations", cal_key], as_index=False)
                .agg(v_full_mean=("v_full", "mean"), n_runs=("v_full", "size"), n_source_islands=("n_source_islands", "mean"))
            )

            gain_target = best_target.merge(vfull_target, on=["n_permutations", cal_key], how="left")
            gain_target["best_n_removed"] = gain_target["n_removed"].astype(int)
            gain_target["best_n_islands"] = (
                gain_target["n_source_islands"].round().astype(int) - gain_target["best_n_removed"]
            ).clip(lower=1)
            gain_target["best_corr_mean"] = gain_target["corr_mean"]
            gain_target["avg_gain_over_v_full"] = gain_target["best_corr_mean"] - gain_target["v_full_mean"]
            gain_target = gain_target[
                [
                    "n_permutations",
                    cal_key,
                    "best_n_removed",
                    "best_n_islands",
                    "best_corr_mean",
                    "v_full_mean",
                    "avg_gain_over_v_full",
                    "n_runs",
                ]
            ].sort_values(["n_permutations", cal_key]).reset_index(drop=True)

            target_gain_path = target_dir / "gain_over_vfull_surface_data.csv"
            gain_target.to_csv(target_gain_path, index=False)

            target_grid_plot_path = target_dir / "remove_curve_grid.png"
            _plot_remove_curve_grid(
                target_summary,
                n_perm_grid=n_perm_grid,
                cal_values=[float(x) for x in cal_values],
                cal_key=cal_key,
                cal_label=cal_label,
                output_path=target_grid_plot_path,
            )

            target_surface_path = target_dir / "gain_over_vfull_surface_3d.png"
            _plot_gain_3d_surface(gain_target, output_path=target_surface_path, cal_key=cal_key, cal_label=cal_label)

            running_rank_outputs: Dict[str, Any] = {}
            if not rank_summary.empty:
                target_rank = rank_summary[rank_summary["target_island"] == target_code].copy()
                if not target_rank.empty:
                    for cal_val in sorted(target_rank[cal_key].unique().tolist()):
                        cal_rank = target_rank[np.isclose(target_rank[cal_key], float(cal_val))].copy()
                        if cal_rank.empty:
                            continue

                        token = _cal_value_token(cal_key, float(cal_val))
                        rank_csv_path = target_dir / f"running_ranking_{token}.csv"
                        rank_png_path = target_dir / f"running_ranking_{token}.png"

                        cal_rank.sort_values(["permutation_index", "rank_mean", "source_island"]).to_csv(
                            rank_csv_path, index=False
                        )
                        _plot_running_rank_heatmap(
                            cal_rank,
                            output_path=rank_png_path,
                            selected_permutations=n_perm_grid,
                            target_name=target_name,
                            cal_text=_cal_value_text(cal_key, float(cal_val)),
                        )

                        running_rank_outputs[token] = {
                            "csv": str(rank_csv_path),
                            "plot": str(rank_png_path),
                        }

            per_target_outputs[str(target_code)] = {
                "target_name": target_name,
                "remove_curve_sweep_rows": str(target_rows_path),
                "remove_curve_sweep_summary": str(target_summary_path),
                "sweep_run_metadata": str(target_runs_path),
                "gain_over_vfull_surface_data": str(target_gain_path),
                "remove_curve_grid_plot": str(target_grid_plot_path),
                "gain_surface_3d_plot": str(target_surface_path),
                "running_rankings": running_rank_outputs,
            }

        summary_payload["traits"][trait_name] = {
            "n_successful_runs": int(len(runs_df)),
            "outputs": {
                "remove_curve_sweep_rows": str(remove_all_path),
                "remove_curve_sweep_summary": str(remove_summary_path),
                "sweep_run_metadata": str(runs_path),
                "gain_over_vfull_surface_data": str(gain_path),
                "remove_curve_grid_plot": str(grid_plot_path),
                "gain_surface_3d_plot": str(surface_plot_path),
            },
            "per_target": per_target_outputs,
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

    with open(config_path, "r") as f:
        cfg = json.load(f)

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

    uncertainty_cfg = cfg.get("uncertainty", {})
    n_repeats = int(cfg.get("sweep", {}).get("n_target_split_repeats", uncertainty_cfg.get("n_target_split_repeats", 5)))
    if n_repeats < 1:
        raise ValueError("sweep.n_target_split_repeats must be >= 1")

    trait_specs = _build_trait_specs(cfg)
    n_traits = len(trait_specs)

    if num_shards <= n_repeats:
        # Repeat-only sharding: num_shards must divide n_repeats
        # (so repeats are evenly distributed)
        if (n_repeats % num_shards) != 0:
            logger.warning(
                "n_repeats (%d) is not divisible by num_shards (%d); "
                "some shards will do more work than others.",
                n_repeats, num_shards,
            )
    else:
        # Repeat+trait sharding
        if (num_shards % n_repeats) != 0:
            raise ValueError(
                f"num_shards ({num_shards}) must be divisible by n_repeats ({n_repeats}) "
                f"when num_shards > n_repeats"
            )
        trait_multiplier = num_shards // n_repeats
        if n_traits % trait_multiplier != 0:
            raise ValueError(
                f"num_shards/n_repeats = {trait_multiplier}, but the number of traits "
                f"({n_traits}) is not divisible by {trait_multiplier}. "
                f"Choose num_shards so that num_shards/n_repeats divides {n_traits}."
            )
        logger.info(
            "Trait-level sharding enabled: %d shards = %d repeats x %d trait groups "
            "(%d traits per group).",
            num_shards, n_repeats, trait_multiplier, n_traits // trait_multiplier,
        )

    run_worker(config_path, shard_index=shard_index, num_shards=num_shards, target_override=args.target_islands)


if __name__ == "__main__":
    main()
