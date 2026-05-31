#!/usr/bin/env python
"""Generate the per-trait, per-variant configs for E6 (PC ridge weighted)
and E8 (MLP, optionally PC-based, with weighting-mode knob).

Re-run this script if you tweak ranges or add traits.

Output filenames live under config/ as
    final_e6_pc_ridge_<weighting>_<trait>.json
    final_e8_<model>_<weighting>_<trait>.json

with <weighting> in {uniform, avggrm, importance} and <model> in {mlp, pc_mlp}.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG_ROOT = REPO / "config"
CFG_E6 = CFG_ROOT / "e6"
CFG_E8 = CFG_ROOT / "e8"
CFG_E6.mkdir(parents=True, exist_ok=True)
CFG_E8.mkdir(parents=True, exist_ok=True)


TRAITS = [
    {"name": "body_mass",  "npz": "Data/npz/snp_body_mass_ALL.npz",  "tuned_r": 2500, "tuned_alpha": 89093.20},
    {"name": "thr_tarsus", "npz": "Data/npz/snp_thr_tarsus_ALL.npz", "tuned_r": 3500, "tuned_alpha": 156026.34},
    {"name": "thr_wing",   "npz": "Data/npz/snp_thr_wing_ALL.npz",   "tuned_r": 4500, "tuned_alpha": 32010.23},
]


WEIGHT_SEARCH_AVGGRM = {
    "scheme_choices": ["uniform", "linear", "exponential"],
    "floor": 1e-6,
    "clip_max_choices": [None, 3.0, 5.0, 10.0, 20.0],
    "linear":      {"min_weight_range": [0.05, 1.0], "max_weight_range": [1.0, 5.0]},
    "exponential": {"beta_range": [0.05, 10.0], "beta_log": False},
}

WEIGHT_SEARCH_IMPORTANCE = {
    "method_choices": ["uniform", "pc_logistic"],
    "n_components_choices": [10, 20, 40, 60, 80, 100],
    "logistic_c_loguniform": [1e-7, 10.0],
    "pca_fit_choices": ["combined"],
    "solver_choices": ["lbfgs"],
    "standardize_with_source": True,
    "max_iter": 2000,
    "fit_intercept": True,
    "floor": 1e-6,
    "prob_clip_range": [0.0005, 0.05],
    "rho_choices": [0.3, 0.5, 0.7, 0.85, 1.0],
    "clip_max_choices": [1.5, 2.5, 4.0, 7.0, 12.0],
    "min_effective_sample_size_frac": 0.2,
}


# ----------------------------------------------------------------------------
# E6: PC ridge nested LOIO
# ----------------------------------------------------------------------------

def make_e6_config(trait: dict, weighting: str) -> dict:
    if weighting not in {"avggrm", "importance"}:
        raise ValueError(weighting)
    out_dir = f"outputs/final_results/e6_pc_ridge_{weighting}/{trait['name']}"
    out_name = f"e6_pc_ridge_{weighting}_{trait['name']}"

    base_paths = {
        "npz": trait["npz"],
        "output_dir": out_dir,
        "output_name": out_name,
    }
    if weighting == "avggrm":
        base_paths["grm_rds"] = "Data/GRM/GRM_vanraden.rds"

    cfg = {
        "base_train": {
            "seed": 14,
            "paths": base_paths,
            "target_column": "y_adjusted",
            "eval_target_column": "y_mean",
            "standardize_features": False,
            "min_count": 20,
        },
        "cv": {
            "strategy": "leave_island_out",
            "include_islands": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "selected_splits": None,
            "inner_top_k_related_islands": 10,
        },
        "model": {"type": "ridge"},
        "feature_pca": {
            "enabled": True,
            "n_pcs_fit": 5000,
            "n_pcs_range": [500, 5000],
            "n_pcs_step": 500,
            "standardize_for_pca": False,
            "pca_seed": 14,
        },
        "n_trials": 300,
        "enable_pruning": True,
        "show_progress_bar": False,
        "search_space": {
            "ridge": {"alpha_loguniform": [1e3, 1e7]},
            # When feature_pca is active the runner ignores this block, but we set
            # it to a safe no-op for clarity.
            "feature_selection": {"use_snp_selection_choices": [False]},
        },
    }
    if weighting == "avggrm":
        cfg["search_space"]["weighting"] = WEIGHT_SEARCH_AVGGRM
    else:
        cfg["search_space"]["importance_weighting"] = WEIGHT_SEARCH_IMPORTANCE

    return cfg


# ----------------------------------------------------------------------------
# E8: MLP nested LOIO
# ----------------------------------------------------------------------------

MLP_MODEL_SPACE = {
    "hidden_dims_choices": [
        [1024, 256], [512, 256], [512, 128], [512, 64],
        [256, 128], [256, 64], [128, 64],
    ],
    "dropout_range": [0.0, 0.6],
    "batch_norm_choices": [False],
}
MLP_TRAIN_SPACE = {
    "lr_loguniform": [1e-5, 5e-4],
    "wd_loguniform": [1e-8, 1e-3],
    "epochs_range": [50, 300],
    "loss_choices": ["mse", "mae"],
    "optimizer_choices": ["adam", "adamw"],
}


def make_e8_config(trait: dict, model: str, weighting: str) -> dict:
    if model not in {"mlp", "pc_mlp"}:
        raise ValueError(model)
    if weighting not in {"uniform", "avggrm", "importance"}:
        raise ValueError(weighting)
    out_dir = f"outputs/final_results/e8_{model}_{weighting}/{trait['name']}"
    out_name = f"e8_{model}_{weighting}_{trait['name']}"
    base_paths = {
        "npz": trait["npz"],
        "output_dir": out_dir,
        "output_name": out_name,
    }
    needs_grm = (weighting == "avggrm")
    # GRM is also helpful for inner_top_k_related_islands; we keep it loaded
    # whenever it would help.
    base_paths["grm_rds"] = "Data/GRM/GRM_vanraden.rds"

    cfg = {
        "base_train": {
            "seed": 14,
            "paths": base_paths,
            "target_column": "y_adjusted",
            "eval_target_column": "y_mean",
            "standardize_features": False,
            "min_count": 20,
        },
        "cv": {
            "strategy": "leave_island_out",
            "include_islands": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "selected_splits": None,
            "inner_top_k_related_islands": 10,
        },
        "model": {"type": "mlp"},
        "weighting_mode": weighting,
        # The uniform variants are the gating comparison (full-SNP vs PC); give
        # them more trials. Weighted variants piggyback on the architecture
        # decision.
        "n_trials": 150 if weighting == "uniform" else 80,
        "enable_pruning": True,
        "pruner_warmup_epochs": 5,
        "show_progress_bar": False,
        "parallel_outer_splits": {"enabled": True, "max_concurrent_gpus": 2},
        "search_space": {
            "model": MLP_MODEL_SPACE,
            "training": MLP_TRAIN_SPACE,
            "feature_selection": {"use_snp_selection_choices": [False]},
        },
    }

    if model == "pc_mlp":
        cfg["feature_pca"] = {
            "enabled": True,
            "n_pcs_fit": 5000,
            "n_pcs_range": [500, 5000],
            "n_pcs_step": 500,
            "standardize_for_pca": False,
            "pca_seed": 14,
        }

    if weighting == "avggrm":
        cfg["search_space"]["weighting"] = WEIGHT_SEARCH_AVGGRM
    elif weighting == "importance":
        cfg["search_space"]["importance_weighting"] = WEIGHT_SEARCH_IMPORTANCE
    # uniform: no weighting search block needed

    if not needs_grm and cfg["cv"]["inner_top_k_related_islands"] is None:
        base_paths.pop("grm_rds", None)

    return cfg


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    written: list[Path] = []
    for trait in TRAITS:
        for weighting in ("avggrm", "importance"):
            cfg = make_e6_config(trait, weighting)
            path = CFG_E6 / f"final_e6_pc_ridge_{weighting}_{trait['name']}_config.json"
            _write(path, cfg)
            written.append(path)
        for model in ("mlp", "pc_mlp"):
            for weighting in ("uniform", "avggrm", "importance"):
                cfg = make_e8_config(trait, model, weighting)
                path = CFG_E8 / f"final_e8_{model}_{weighting}_{trait['name']}_config.json"
                _write(path, cfg)
                written.append(path)

    print(f"Wrote {len(written)} configs:")
    for path in written:
        print(f"  {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
