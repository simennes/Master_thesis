#!/usr/bin/env python
"""Within-population nested k-fold PC-ridge runner.

This is the within-population counterpart to ``run_pc_ridge_nested_loio.py``.
It uses random k-fold outer splits, tunes the number of retained PCs and ridge
penalty inside each outer-training set, then refits on the full outer-training
fold and evaluates held-out individuals.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from src.cv_utils import island_label
from src.data import load_data
from src.utils import _pearson_corr, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "false", "0"}:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [int(v) for v in parsed]
    except Exception:
        pass
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _build_trait_specs(config: dict[str, Any], only_traits: set[str] | None) -> list[dict[str, Any]]:
    traits_cfg = config.get("traits")
    if not traits_cfg:
        paths = dict(config["paths"])
        traits_cfg = [{"name": "default", "npz": paths.get("npz", paths.get("npz_path"))}]

    specs: list[dict[str, Any]] = []
    for raw in traits_cfg:
        name = str(raw["name"])
        if only_traits is not None and name not in only_traits:
            continue
        if "npz" not in raw:
            raise ValueError(f"Trait entry missing 'npz': {raw}")
        specs.append(
            {
                "name": name,
                "paths": {"npz": str(raw["npz"])},
                "target_column": raw.get("target_column", config.get("target_column", "y_adjusted")),
                "eval_target_column": raw.get("eval_target_column", config.get("eval_target_column", "y_mean")),
                "standardize_features": bool(raw.get("standardize_features", config.get("standardize_features", False))),
                "min_count": int(raw.get("min_count", config.get("min_count", 20))),
            }
        )

    if not specs:
        requested = "<all>" if only_traits is None else ", ".join(sorted(only_traits))
        raise ValueError(f"No trait specs selected for: {requested}")
    return specs


def _resolve_pc_grid(search_cfg: dict[str, Any]) -> list[int]:
    grid = search_cfg.get("n_pcs_choices")
    if grid is None:
        range_cfg = search_cfg.get("n_pcs_range", [500, 5000])
        step = int(search_cfg.get("n_pcs_step", 500))
        lo, hi = int(range_cfg[0]), int(range_cfg[1])
        grid = list(range(lo, hi + 1, step))
    grid = sorted({int(v) for v in grid if int(v) > 0})
    if not grid:
        raise ValueError("PC grid must contain at least one positive integer.")
    return grid


def _resolve_alpha_range(search_cfg: dict[str, Any]) -> tuple[float, float, bool]:
    range_cfg = search_cfg.get("alpha_range", search_cfg.get("alpha_loguniform", [1e3, 1e6]))
    lo, hi = float(range_cfg[0]), float(range_cfg[1])
    if lo <= 0 or hi <= 0 or lo >= hi:
        raise ValueError(f"alpha_range must satisfy 0 < lo < hi; got {(lo, hi)}.")
    return lo, hi, bool(search_cfg.get("alpha_log", True))


def _filter_include_islands(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    locality: np.ndarray,
    y_eval: np.ndarray,
    include_islands: list[Any] | None,
    code_to_label: dict[int, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not include_islands:
        return X, y, ids, locality, y_eval

    label_to_code = {str(v): int(k) for k, v in (code_to_label or {}).items()}
    present_codes = {int(c) for c in np.unique(locality)}
    include_codes: set[int] = set()
    for value in include_islands:
        sval = str(value)
        if sval in label_to_code:
            include_codes.add(int(label_to_code[sval]))
            continue
        try:
            code = int(value)
        except Exception:
            continue
        if code in present_codes:
            include_codes.add(code)

    if not include_codes:
        available = [f"{c}:{(code_to_label or {}).get(int(c), '?')}" for c in sorted(present_codes)]
        raise ValueError(
            f"include_islands={include_islands} did not match any samples. "
            f"Available codes/labels: {available}"
        )

    mask = np.isin(locality.astype(int), np.fromiter(include_codes, dtype=int))
    if not np.any(mask):
        raise ValueError("include_islands filtered out all samples.")
    return X[mask], y[mask], ids[mask], locality[mask], y_eval[mask]


def _fit_dataset_pca(
    X: np.ndarray,
    max_r: int,
    seed: int,
    standardize: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    feasible_r = int(min(int(max_r), n_samples, n_features))
    if feasible_r < 1:
        raise ValueError(
            f"Cannot fit PCA: feasible_r={feasible_r} (n={n_samples}, p={n_features}, max_r={max_r})."
        )

    X_in = X.astype(np.float32, copy=False)
    if standardize:
        mean = X_in.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = X_in.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std == 0] = 1.0
        X_in = (X_in - mean) / std

    pca = PCA(n_components=feasible_r, svd_solver="randomized", random_state=int(seed))
    t0 = time.perf_counter()
    Z = pca.fit_transform(X_in)
    dt = float(time.perf_counter() - t0)
    return (
        Z.astype(np.float32, copy=False),
        dt,
        pca.explained_variance_ratio_.astype(np.float32, copy=False),
    )


def _fit_ridge_predict(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    r: int,
    alpha: float,
) -> tuple[np.ndarray, int]:
    r_eff = int(min(int(r), Z_train.shape[1]))
    model = Ridge(alpha=max(float(alpha), 1e-12))
    model.fit(Z_train[:, :r_eff], y_train)
    pred = model.predict(Z_test[:, :r_eff])
    return pred.astype(np.float64, copy=False), r_eff


def _fit_ridge_eval(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_eval_test: np.ndarray,
    r: int,
    alpha: float,
) -> float:
    pred, _ = _fit_ridge_predict(Z_train, y_train, Z_test, r, alpha)
    corr = float(_pearson_corr(y_eval_test, pred))
    return corr if np.isfinite(corr) else 0.0


def _make_cv_splits(
    indices: np.ndarray,
    locality: np.ndarray,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int,
    stratify_by_locality: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(indices) < 2:
        raise ValueError("Need at least two samples for k-fold CV.")

    effective_splits = int(min(int(n_splits), len(indices)))
    if effective_splits < 2:
        raise ValueError("n_splits must yield at least two folds.")
    if effective_splits < int(n_splits):
        logger.warning(
            "Requested %d splits but only %d samples are available; using %d splits.",
            int(n_splits),
            len(indices),
            effective_splits,
        )

    if stratify_by_locality and locality is not None:
        splitter = StratifiedKFold(
            n_splits=effective_splits,
            shuffle=shuffle,
            random_state=int(random_state) if shuffle else None,
        )
        split_iter = splitter.split(np.zeros(len(indices)), locality[indices].astype(int))
    else:
        splitter = KFold(
            n_splits=effective_splits,
            shuffle=shuffle,
            random_state=int(random_state) if shuffle else None,
        )
        split_iter = splitter.split(indices)

    return [(indices[tr], indices[te]) for tr, te in split_iter]


def _locality_summary(
    idx: np.ndarray,
    locality: np.ndarray,
    code_to_label: dict[int, Any] | None,
) -> tuple[str, str, str]:
    codes = sorted(int(v) for v in np.unique(locality[idx]))
    original = [
        str((code_to_label or {}).get(int(code), int(code)))
        for code in codes
    ]
    names = [island_label(int(code), code_to_label) for code in codes]
    return json.dumps(codes), json.dumps(original), json.dumps(names)


def _original_locality_label(code: int, code_to_label: dict[int, Any] | None) -> str | None:
    if code_to_label is None:
        return None
    if int(code) not in code_to_label:
        return None
    return str(code_to_label[int(code)])


def _write_trait_outputs(
    trait_dir: Path,
    file_stem: str,
    fold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    best_params_per_fold: list[dict[str, Any]],
    *,
    trait_name: str,
    pca_seconds: float,
) -> dict[str, Any] | None:
    trait_dir.mkdir(parents=True, exist_ok=True)
    if not fold_rows:
        return None

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(prediction_rows)

    fold_path = trait_dir / f"{file_stem}_per_fold_results.csv"
    pred_path = trait_dir / f"{file_stem}_predictions.csv"
    fold_df.to_csv(fold_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    summary = {
        "trait": trait_name,
        "pooled_pearson_r": float(_pearson_corr(pred_df["y_eval"].to_numpy(), pred_df["prediction"].to_numpy())),
        "fold_pearson_r_mean": float(fold_df["pearson_r"].mean()),
        "fold_pearson_r_std": float(fold_df["pearson_r"].std()),
        "n_folds": int(fold_df.shape[0]),
        "n_predictions": int(pred_df.shape[0]),
        "best_alpha_geomean": float(np.exp(np.log(fold_df["best_alpha"]).mean())),
        "best_n_pcs_median": float(fold_df["best_n_pcs"].median()),
        "best_n_pcs_mode": int(fold_df["best_n_pcs"].mode().iloc[0]),
        "pca_time_seconds": float(pca_seconds),
    }
    pd.DataFrame([summary]).to_csv(trait_dir / f"{file_stem}_summary.csv", index=False)

    with (trait_dir / f"{file_stem}_best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "trait": trait_name,
                "best_params_per_fold": best_params_per_fold,
                "summary": summary,
            },
            handle,
            indent=2,
        )

    logger.info("Saved per-fold results: %s", fold_path)
    logger.info("Saved predictions: %s", pred_path)
    return summary


def _run_trait(
    trait_spec: dict[str, Any],
    *,
    config: dict[str, Any],
    output_root: Path,
    file_stem: str,
    selected_folds: set[int] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    trait_name = str(trait_spec["name"])
    seed = int(config.get("seed", 42))
    cv_cfg = config.get("cv", {})
    search_cfg = config.get("search_space", {})

    n_trials = int(config.get("n_trials", 100))
    pc_grid = _resolve_pc_grid(search_cfg)
    alpha_lo, alpha_hi, alpha_log = _resolve_alpha_range(search_cfg)

    outer_splits = int(cv_cfg.get("n_splits", 10))
    inner_splits = int(cv_cfg.get("inner_splits", 5))
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = int(cv_cfg.get("random_state", seed))
    stratify_by_locality = bool(cv_cfg.get("stratify_by_locality", True))

    logger.info("==== Trait '%s' ====", trait_name)
    X, y, ids, _, locality, code_to_label, y_eval = load_data(
        paths=trait_spec["paths"],
        target_column=trait_spec["target_column"],
        standardize_features=False,
        return_locality=True,
        min_count=int(trait_spec["min_count"]),
        return_eval=True,
        eval_target_column=trait_spec["eval_target_column"],
    )
    if y_eval is None:
        y_eval = y.copy()

    X, y, ids, locality, y_eval = _filter_include_islands(
        X=X,
        y=y,
        ids=ids,
        locality=locality,
        y_eval=y_eval,
        include_islands=cv_cfg.get("include_islands"),
        code_to_label=code_to_label,
    )

    max_r = int(max(pc_grid))
    standardize = bool(trait_spec.get("standardize_features", False))
    Z_full, pca_seconds, var_ratio = _fit_dataset_pca(
        X=X,
        max_r=max_r,
        seed=seed,
        standardize=standardize,
    )
    logger.info(
        "Trait=%s PCA fit on full dataset (n=%d, p=%d, r=%d) in %.2fs (cumulative var=%.3f)",
        trait_name,
        int(X.shape[0]),
        int(X.shape[1]),
        int(Z_full.shape[1]),
        pca_seconds,
        float(np.sum(var_ratio)),
    )

    outer_plan = _make_cv_splits(
        indices=np.arange(len(y), dtype=int),
        locality=locality,
        n_splits=outer_splits,
        shuffle=shuffle,
        random_state=random_state,
        stratify_by_locality=stratify_by_locality,
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    best_params_per_fold: list[dict[str, Any]] = []

    for fold_idx, (idx_outer_train, idx_outer_test) in enumerate(outer_plan, start=1):
        if selected_folds is not None and fold_idx not in selected_folds:
            continue

        logger.info(
            "OUTER %d/%d within-pop fold (n_train=%d, n_test=%d)",
            fold_idx,
            len(outer_plan),
            int(idx_outer_train.size),
            int(idx_outer_test.size),
        )

        inner_plan = _make_cv_splits(
            indices=idx_outer_train,
            locality=locality,
            n_splits=inner_splits,
            shuffle=shuffle,
            random_state=random_state + fold_idx,
            stratify_by_locality=stratify_by_locality,
        )

        def objective(trial: optuna.Trial) -> float:
            r = int(trial.suggest_categorical("n_pcs", [int(v) for v in pc_grid]))
            alpha = float(trial.suggest_float("alpha", alpha_lo, alpha_hi, log=alpha_log))
            inner_scores: list[float] = []
            for step, (idx_inner_train, idx_inner_val) in enumerate(inner_plan, start=1):
                corr = _fit_ridge_eval(
                    Z_train=Z_full[idx_inner_train],
                    y_train=y[idx_inner_train],
                    Z_test=Z_full[idx_inner_val],
                    y_eval_test=y_eval[idx_inner_val],
                    r=r,
                    alpha=alpha,
                )
                inner_scores.append(corr)
                trial.report(float(np.mean(inner_scores)), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(inner_scores)) if inner_scores else 0.0

        sampler = optuna.samplers.TPESampler(seed=seed + fold_idx)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{trait_name}_within_pop_outer{fold_idx}",
            sampler=sampler,
            pruner=pruner,
        )
        t_study0 = time.perf_counter()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        study_seconds = float(time.perf_counter() - t_study0)

        best_trial = study.best_trial
        best_r = int(best_trial.params["n_pcs"])
        best_alpha = float(best_trial.params["alpha"])
        mean_inner_r = float(best_trial.value)

        t_fit0 = time.perf_counter()
        pred, best_r_fit = _fit_ridge_predict(
            Z_train=Z_full[idx_outer_train],
            y_train=y[idx_outer_train],
            Z_test=Z_full[idx_outer_test],
            r=best_r,
            alpha=best_alpha,
        )
        fit_seconds = float(time.perf_counter() - t_fit0)
        test_r = float(_pearson_corr(y_eval[idx_outer_test], pred))
        if not np.isfinite(test_r):
            test_r = 0.0

        test_codes, test_original, test_names = _locality_summary(idx_outer_test, locality, code_to_label)
        cum_var_at_best_r = float(np.sum(var_ratio[:best_r_fit]))
        fold_row = {
            "trait": trait_name,
            "fold": int(fold_idx),
            "cv_strategy": "within_population_kfold",
            "n_train": int(idx_outer_train.size),
            "n_test": int(idx_outer_test.size),
            "n_features_available": int(X.shape[1]),
            "n_pcs_available": int(Z_full.shape[1]),
            "best_n_pcs": int(best_r),
            "best_n_pcs_fit": int(best_r_fit),
            "best_alpha": float(best_alpha),
            "mean_inner_pearson_r": float(mean_inner_r),
            "pearson_r": float(test_r),
            "outer_pca_cumvar_at_best_r": float(cum_var_at_best_r),
            "fit_time_seconds": float(fit_seconds),
            "study_time_seconds": float(study_seconds),
            "n_trials_completed": int(len(study.trials)),
            "n_inner_folds": int(len(inner_plan)),
            "test_locality_codes": test_codes,
            "test_locality_original_labels": test_original,
            "test_localities": test_names,
        }
        fold_rows.append(fold_row)
        best_params_per_fold.append(
            {
                "fold": int(fold_idx),
                "best_params": {
                    "n_pcs": int(best_r),
                    "n_pcs_fit": int(best_r_fit),
                    "alpha": float(best_alpha),
                },
                "mean_inner_pearson_r": float(mean_inner_r),
                "pearson_r": float(test_r),
            }
        )

        for row_idx, sample_idx in enumerate(idx_outer_test):
            loc_code = int(locality[sample_idx])
            prediction_rows.append(
                {
                    "trait": trait_name,
                    "fold": int(fold_idx),
                    "ringnumber": str(ids[sample_idx]),
                    "locality_code": loc_code,
                    "locality_original_label": _original_locality_label(loc_code, code_to_label),
                    "locality_name": island_label(loc_code, code_to_label),
                    "y_train_target": float(y[sample_idx]),
                    "y_eval": float(y_eval[sample_idx]),
                    "prediction": float(pred[row_idx]),
                }
            )

        logger.info(
            "OUTER %d done in %.1fs: best n_pcs=%d alpha=%.3e inner_r=%.4f test_r=%.4f",
            fold_idx,
            study_seconds,
            best_r,
            best_alpha,
            mean_inner_r,
            test_r,
        )

    summary = _write_trait_outputs(
        output_root / trait_name,
        file_stem,
        fold_rows,
        prediction_rows,
        best_params_per_fold,
        trait_name=trait_name,
        pca_seconds=pca_seconds,
    )
    return fold_rows, prediction_rows, summary


def _write_combined_outputs(
    output_root: Path,
    file_stem: str,
    fold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(output_root / f"{file_stem}_per_fold_results.csv", index=False)
        logger.info("Saved combined per-fold results under %s", output_root)
    if prediction_rows:
        pd.DataFrame(prediction_rows).to_csv(output_root / f"{file_stem}_predictions.csv", index=False)
        logger.info("Saved combined predictions under %s", output_root)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_root / f"{file_stem}_summary.csv", index=False)
        logger.info("Saved combined summary under %s", output_root)


def merge_outputs(config: dict[str, Any], only_traits: set[str] | None = None) -> None:
    output_cfg = config.get("output", {})
    output_root = Path(output_cfg.get("root_dir", "outputs/final_results/within_pop_pc_ridge_10fold"))
    file_stem = str(output_cfg.get("file_stem", "within_pop_pc_ridge_10fold"))
    if selected_folds is not None:
        suffix = "_".join(str(i) for i in sorted(selected_folds))
        file_stem = f"{file_stem}_folds_{suffix}"
    output_root.mkdir(parents=True, exist_ok=True)

    trait_specs = _build_trait_specs(config, only_traits)
    fold_frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for spec in trait_specs:
        trait_dir = output_root / str(spec["name"])
        fold_path = trait_dir / f"{file_stem}_per_fold_results.csv"
        pred_path = trait_dir / f"{file_stem}_predictions.csv"
        summary_path = trait_dir / f"{file_stem}_summary.csv"
        if fold_path.exists():
            fold_frames.append(pd.read_csv(fold_path))
        else:
            logger.warning("Missing per-fold file for trait %s: %s", spec["name"], fold_path)
        if pred_path.exists():
            pred_frames.append(pd.read_csv(pred_path))
        else:
            logger.warning("Missing prediction file for trait %s: %s", spec["name"], pred_path)
        if summary_path.exists():
            summary_frames.append(pd.read_csv(summary_path))
        else:
            logger.warning("Missing summary file for trait %s: %s", spec["name"], summary_path)

    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(output_root / f"{file_stem}_per_fold_results.csv", index=False)
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(output_root / f"{file_stem}_predictions.csv", index=False)
    if summary_frames:
        pd.concat(summary_frames, ignore_index=True).to_csv(output_root / f"{file_stem}_summary.csv", index=False)
    logger.info("Merged available trait outputs under %s", output_root)


def run(
    config: dict[str, Any],
    *,
    only_traits: set[str] | None = None,
    selected_folds: set[int] | None = None,
) -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    cv_cfg = config.get("cv", {})
    strategy = str(cv_cfg.get("strategy", "kfold")).lower()
    if strategy not in {"kfold", "within_population_kfold"}:
        raise ValueError("This runner supports cv.strategy='kfold' or 'within_population_kfold'.")

    output_cfg = config.get("output", {})
    output_root = Path(output_cfg.get("root_dir", "outputs/final_results/within_pop_pc_ridge_10fold"))
    file_stem = str(output_cfg.get("file_stem", "within_pop_pc_ridge_10fold"))
    output_root.mkdir(parents=True, exist_ok=True)

    with (output_root / f"{file_stem}_config_used.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    trait_specs = _build_trait_specs(config, only_traits)
    all_fold_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []

    for trait_spec in trait_specs:
        fold_rows, prediction_rows, summary = _run_trait(
            trait_spec=trait_spec,
            config=config,
            output_root=output_root,
            file_stem=file_stem,
            selected_folds=selected_folds,
        )
        all_fold_rows.extend(fold_rows)
        all_prediction_rows.extend(prediction_rows)
        if summary is not None:
            all_summary_rows.append(summary)

    if only_traits is None and selected_folds is None:
        _write_combined_outputs(output_root, file_stem, all_fold_rows, all_prediction_rows, all_summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trait", action="append", default=None, help="Trait to run; can be repeated.")
    parser.add_argument("--selected_folds", default=None, help="Optional JSON list or comma-separated 1-based fold indices.")
    parser.add_argument("--n_trials", type=int, default=None, help="Override config n_trials for quick smoke runs.")
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if args.n_trials is not None:
        config["n_trials"] = int(args.n_trials)

    only_traits = set(_as_list(args.trait)) if args.trait else None
    selected = _parse_int_list(args.selected_folds)
    selected_folds = set(selected) if selected else None

    if args.mode == "merge":
        merge_outputs(config, only_traits=only_traits)
    else:
        run(config, only_traits=only_traits, selected_folds=selected_folds)


if __name__ == "__main__":
    main()
