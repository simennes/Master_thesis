#!/usr/bin/env python
"""
Run grid sweeps for island-level TMC-Shapley over:
- multiple n_permutations
- multiple cal_fraction values

Outputs:
- per-cell remove-curve summaries
- n_permutations x n_cal_fraction subplot figure of removal curves
- 3D surface for average gain over v_full using best-performing n_removed per cell
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
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


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def _make_repeat_seed(global_seed: int, target_code: int, cal_frac: float, repeat_idx: int) -> int:
    token = f"{int(global_seed)}|{int(target_code)}|{float(cal_frac):.12g}|{int(repeat_idx)}"
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    # Positive 31-bit int, suitable for RNG/filenames
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
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if gain_df.empty:
        return

    x_vals = sorted(gain_df["n_permutations"].unique().tolist())
    y_vals = sorted(gain_df[cal_key].unique().tolist())

    # Build dense matrix only when full grid available
    z_mat = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
    for yi, y in enumerate(y_vals):
        for xi, x in enumerate(x_vals):
            row = gain_df[(gain_df["n_permutations"] == x) & (np.isclose(gain_df[cal_key], y))]
            if not row.empty:
                z_mat[yi, xi] = float(row.iloc[0]["avg_gain_over_v_full"])

    X, Y = np.meshgrid(np.array(x_vals, dtype=float), np.array(y_vals, dtype=float))

    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    if np.isfinite(z_mat).all():
        surf = ax.plot_surface(X, Y, z_mat, cmap="viridis", edgecolor="none", alpha=0.9)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="avg gain over v_full")
    else:
        valid = gain_df[np.isfinite(gain_df["avg_gain_over_v_full"])].copy()
        ax.scatter(
            valid["n_permutations"],
            valid[cal_key],
            valid["avg_gain_over_v_full"],
            c=valid["avg_gain_over_v_full"],
            cmap="viridis",
            s=55,
        )

    ax.set_xlabel("n_permutations")
    ax.set_ylabel(cal_label)
    ax.set_zlabel("avg gain over v_full")
    ax.set_title("Best average gain over v_full by sweep cell")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TMC-Shapley sweep over n_permutations and cal_fraction")
    parser.add_argument("--config", required=True, help="Path to tmc_shapley_islands.json config file")
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

    paths = cfg["paths"]
    base_output = Path(paths.get("output_dir", "outputs/tmc_shapley"))
    sweep_output = Path(sweep_cfg.get("output_dir", str(base_output / "sweep")))
    save_cell_artifacts = bool(sweep_cfg.get("save_cell_artifacts", False))

    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    global_seed = int(cfg.get("seed", 42))
    set_seed(global_seed)

    trait_specs = _build_trait_specs(cfg)

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
    logger.info("Model config: type=%s alpha=%s", model_type, base_shapley_cfg.ridge_alpha)

    uncertainty_cfg = cfg.get("uncertainty", {})
    n_repeats = int(sweep_cfg.get("n_target_split_repeats", uncertainty_cfg.get("n_target_split_repeats", 5)))
    if n_repeats < 1:
        raise ValueError("sweep.n_target_split_repeats must be >= 1")

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        trait_output = sweep_output / trait_name if len(trait_specs) > 1 else sweep_output
        trait_output.mkdir(parents=True, exist_ok=True)

        logger.info("Loading data for trait '%s' from %s", trait_name, trait_spec["paths"].get("npz", "<unknown>"))
        X, y, ids, GRM_df, locality, code_to_label, y_eval = load_data(
            paths=trait_spec["paths"],
            target_column=trait_spec["target_column"],
            standardize_features=trait_spec["standardize_features"],
            return_locality=True,
            min_count=trait_spec["min_count"],
            return_eval=True,
            eval_target_column=trait_spec["eval_target_column"],
        )
        logger.info(
            "Trait '%s' loaded: %d samples, %d SNPs, %d islands",
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

        raw_targets = args.target_islands if args.target_islands else cfg.get("target_islands", [0])
        target_codes = [resolve_island_code(t, code_to_label, present_codes) for t in raw_targets]

        logger.info(
            "Sweep config trait=%s: n_permutations=%s, %s=%s, repeats=%d, targets=%s",
            trait_name,
            n_perm_grid,
            cal_label,
            cal_values,
            n_repeats,
            target_codes,
        )

        remove_rows: List[pd.DataFrame] = []
        run_rows: List[Dict[str, Any]] = []

        remove_all_path = trait_output / "remove_curve_sweep_rows.csv"
        runs_path = trait_output / "sweep_run_metadata.csv"
        split_plan_path = trait_output / "target_split_plan.csv"
        fixed_snp_path = trait_output / "fixed_snp_columns.npy"

        if remove_all_path.exists():
            remove_all_path.unlink()
        if runs_path.exists():
            runs_path.unlink()
        if split_plan_path.exists():
            split_plan_path.unlink()

        # Fixed SNP subset for entire trait sweep
        fixed_snp_cols: np.ndarray | None = None
        use_snp_selection = bool(cfg.get("use_snp_selection", False))
        num_snps = cfg.get("num_snps", None)
        snp_mode = str(cfg.get("snp_selection_mode", "random")).lower()
        if use_snp_selection and num_snps is not None and int(num_snps) < int(X.shape[1]):
            if snp_mode != "random":
                raise ValueError("Sweep currently supports fixed global SNP subset for snp_selection_mode='random'.")
            rng_snp = np.random.default_rng(global_seed)
            fixed_snp_cols = np.sort(rng_snp.choice(X.shape[1], size=int(num_snps), replace=False).astype(np.int64))
            np.save(fixed_snp_path, fixed_snp_cols)
            logger.info("Fixed SNP subset selected once for trait '%s': n=%d", trait_name, len(fixed_snp_cols))

        # Pre-draw and store target split plans for each (target, cal_setting, repeat)
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

                for repeat_idx in range(n_repeats):
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
                        "max_cal_fraction": float(max_cal_fraction_eff) if max_cal_fraction_eff is not None else None,
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
            pd.DataFrame(split_plan_rows).to_csv(split_plan_path, index=False)
            logger.info("Saved precomputed target split plan: %s", split_plan_path)

        total_jobs = len(target_codes) * len(n_perm_grid) * len(cal_values) * n_repeats
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
                    for repeat_idx in range(n_repeats):
                        split_key = (int(target_code), float(cal_val), int(repeat_idx))
                        split_plan = split_index_map[split_key]
                        repeat_seed = int(split_plan["repeat_seed"])
                        repeat_cfg = copy.deepcopy(base_shapley_cfg)
                        repeat_cfg.n_permutations = int(n_perm)
                        repeat_cfg.seed = int(repeat_seed)

                        if save_cell_artifacts:
                            run_output_dir = trait_output / f"target_{target_code}" / f"perm_{n_perm}" / f"cal_{split_plan['cal_fraction']:.3f}" / f"repeat_{repeat_idx}"
                        else:
                            run_output_dir = trait_output / "tmp"

                        logger.info(
                            "Sweep trait=%s %d/%d | target=%s perm=%d %s=%s repeat=%d/%d",
                            trait_name,
                            done_jobs + 1,
                            total_jobs,
                            target_code,
                            n_perm,
                            cal_label,
                            int(cal_val) if cal_mode == "samples" else f"{float(cal_val):.3f}",
                            repeat_idx + 1,
                            n_repeats,
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
                            max_cal_fraction=split_plan["max_cal_fraction"],
                            use_snp_selection=cfg.get("use_snp_selection", False),
                            num_snps=cfg.get("num_snps", None),
                            snp_selection_mode=cfg.get("snp_selection_mode", "random"),
                            cal_idx=np.asarray(split_plan["cal_idx"], dtype=np.int64),
                            test_idx=np.asarray(split_plan["test_idx"], dtype=np.int64),
                            snp_cols_override=fixed_snp_cols,
                            )
                        except Exception as e:
                            logger.exception(
                            "Failed sweep cell target=%s perm=%s cal=%s repeat=%s: %s",
                            target_code,
                            n_perm,
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
                        _append_csv(remove_df, remove_all_path)

                        run_row = {
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
                        run_rows.append(run_row)
                        _append_csv(pd.DataFrame([run_row]), runs_path)

                        done_jobs += 1

        if not remove_rows:
            raise RuntimeError(f"No successful sweep runs produced remove-curve outputs for trait {trait_name}.")

        remove_all_df = pd.concat(remove_rows, ignore_index=True)
        runs_df = pd.DataFrame(run_rows)

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

        summary_payload = {
            "trait": trait_name,
            "targets": [int(t) for t in target_codes],
            "n_permutations": [int(x) for x in n_perm_grid],
            "cal_mode": cal_mode,
            "cal_values": [float(x) for x in cal_values],
            "n_repeats": int(n_repeats),
            "n_successful_runs": int(len(runs_df)),
            "outputs": {
                "remove_curve_sweep_rows": str(remove_all_path),
                "remove_curve_sweep_summary": str(remove_summary_path),
                "sweep_run_metadata": str(runs_path),
                "gain_over_vfull_surface_data": str(gain_path),
                "remove_curve_grid_plot": str(grid_plot_path),
                "gain_surface_3d_plot": str(surface_plot_path),
            },
        }
        with open(trait_output / "sweep_summary.json", "w") as f:
            json.dump(summary_payload, f, indent=2)

        logger.info("Sweep complete for trait '%s'. Outputs written to: %s", trait_name, trait_output)


if __name__ == "__main__":
    main()
