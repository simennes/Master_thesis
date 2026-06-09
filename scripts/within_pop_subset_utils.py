"""Utilities shared by within-population subset-selection runners."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from src.cv_utils import island_label
from src.data import load_data
from src.utils import _pearson_corr


logger = logging.getLogger(__name__)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def parse_int_list(value: str | None) -> list[int] | None:
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


def build_trait_specs(
    config: dict[str, Any],
    only_traits: set[str] | None,
    *,
    require_grm: bool,
) -> list[dict[str, Any]]:
    traits_cfg = config.get("traits")
    if not traits_cfg:
        paths = dict(config["paths"])
        traits_cfg = [{"name": "default", "npz": paths.get("npz", paths.get("npz_path"))}]

    default_paths = dict(config.get("paths", {}))
    if "grm_rds" in config and "grm_rds" not in default_paths:
        default_paths["grm_rds"] = config["grm_rds"]

    specs: list[dict[str, Any]] = []
    for raw in traits_cfg:
        name = str(raw["name"])
        if only_traits is not None and name not in only_traits:
            continue
        if "npz" not in raw:
            raise ValueError(f"Trait entry missing 'npz': {raw}")

        paths = dict(default_paths)
        paths["npz"] = str(raw["npz"])
        if "grm_rds" in raw:
            paths["grm_rds"] = str(raw["grm_rds"])
        if require_grm and not paths.get("grm_rds"):
            raise ValueError("GRM-based subset selection requires paths.grm_rds or trait.grm_rds.")

        specs.append(
            {
                "name": name,
                "paths": paths,
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


def filter_include_islands(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    grm_df: pd.DataFrame | None,
    locality: np.ndarray,
    y_eval: np.ndarray,
    include_islands: list[Any] | None,
    code_to_label: dict[int, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame | None, np.ndarray, np.ndarray]:
    if not include_islands:
        return X, y, ids, grm_df, locality, y_eval

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
    idx = np.where(mask)[0]
    grm_out = grm_df.iloc[idx, idx] if grm_df is not None else None
    return X[mask], y[mask], ids[mask], grm_out, locality[mask], y_eval[mask]


def load_trait_data(
    spec: dict[str, Any],
    config: dict[str, Any],
    *,
    require_grm: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame | None, np.ndarray, dict[int, Any] | None, np.ndarray]:
    X, y, ids, grm_df, locality, code_to_label, y_eval = load_data(
        paths=spec["paths"],
        target_column=spec["target_column"],
        standardize_features=False,
        return_locality=True,
        min_count=int(spec["min_count"]),
        return_eval=True,
        eval_target_column=spec["eval_target_column"],
    )
    if y_eval is None:
        y_eval = y.copy()
    if require_grm and grm_df is None:
        raise ValueError("GRM was not loaded; check paths.grm_rds.")

    cv_cfg = config.get("cv", {})
    X, y, ids, grm_df, locality, y_eval = filter_include_islands(
        X=X,
        y=y,
        ids=ids.astype(str),
        grm_df=grm_df,
        locality=locality.astype(int),
        y_eval=y_eval,
        include_islands=cv_cfg.get("include_islands"),
        code_to_label=code_to_label,
    )
    return X, y, ids, grm_df, locality, code_to_label, y_eval


def make_cv_splits(
    indices: np.ndarray,
    locality: np.ndarray,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int,
    stratify_by_locality: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    effective_splits = int(min(int(n_splits), len(indices)))
    if effective_splits < 2:
        raise ValueError("n_splits must yield at least two folds.")

    if stratify_by_locality:
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


def fit_dataset_pca(
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


def load_fold_hyperparams(config: dict[str, Any], trait_name: str) -> dict[int, dict[str, float]]:
    fixed_cfg = config.get("fixed_params", {})
    root = Path(fixed_cfg.get("root_dir", "outputs/final_results/within_pop_pc_ridge_10fold"))
    stem = str(fixed_cfg.get("file_stem", "within_pop_pc_ridge_10fold"))
    path = fixed_cfg.get("per_fold_results_path")
    params_path = Path(path) if path else root / trait_name / f"{stem}_per_fold_results.csv"
    if not params_path.exists():
        raise FileNotFoundError(
            f"Missing fixed PC-ridge fold parameters for trait '{trait_name}': {params_path}. "
            "Run scripts.run_pc_ridge_nested_kfold first or update fixed_params."
        )

    df = pd.read_csv(params_path)
    required = {"fold", "best_n_pcs", "best_alpha"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{params_path} is missing columns: {sorted(missing)}")

    out: dict[int, dict[str, float]] = {}
    for row in df.itertuples(index=False):
        fold = int(getattr(row, "fold"))
        out[fold] = {
            "n_pcs": float(getattr(row, "best_n_pcs")),
            "alpha": float(getattr(row, "best_alpha")),
        }
    return out


def compute_step_counts(
    n_train_sizes_raw: list[Any] | None,
    n_available: int,
    *,
    include_full_train: bool,
    min_train_size: int = 2,
) -> np.ndarray:
    if n_train_sizes_raw is None:
        step_counts = np.linspace(0.1, 0.9, 9) * float(n_available)
        raw = [int(round(v)) for v in step_counts]
    else:
        raw = [int(v) for v in n_train_sizes_raw]

    values = [int(np.clip(v, min_train_size, n_available)) for v in raw]
    if include_full_train:
        values.append(int(n_available))
    values = sorted({v for v in values if min_train_size <= v <= n_available})
    return np.asarray(values, dtype=np.int64)


def evaluate_pc_ridge_subset(
    Z: np.ndarray,
    y: np.ndarray,
    y_eval: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    n_pcs: int,
    alpha: float,
) -> dict[str, Any]:
    if len(train_idx) < 2:
        return {
            "pearson_r": 0.0,
            "mse_adj": float("inf"),
            "n_pcs_fit": 0,
            "fit_time_seconds": 0.0,
        }

    r_eff = int(min(int(n_pcs), Z.shape[1]))
    t0 = time.perf_counter()
    model = Ridge(alpha=max(float(alpha), 1e-12))
    model.fit(Z[train_idx, :r_eff], y[train_idx])
    pred = model.predict(Z[test_idx, :r_eff])
    fit_time = float(time.perf_counter() - t0)

    corr = float(_pearson_corr(y_eval[test_idx], pred))
    if not np.isfinite(corr):
        corr = 0.0
    mse = float(np.mean((pred - y[test_idx]) ** 2))
    return {
        "pearson_r": corr,
        "mse_adj": mse,
        "n_pcs_fit": int(r_eff),
        "fit_time_seconds": fit_time,
    }


def avg_train_train_block(grm: np.ndarray, idx: np.ndarray, include_diagonal: bool) -> float:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return float("nan")
    block = np.asarray(grm[np.ix_(idx, idx)], dtype=float)
    if include_diagonal or idx.size <= 1:
        return float(np.mean(block))
    offdiag_sum = float(np.sum(block) - np.trace(block))
    return offdiag_sum / float(idx.size * (idx.size - 1))


def append_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    pd.DataFrame(rows).to_csv(path, mode="a", header=write_header, index=False)


def selected_rows(
    *,
    trait_name: str,
    fold: int,
    method: str,
    n_train: int,
    ids: np.ndarray,
    locality: np.ndarray,
    code_to_label: dict[int, Any] | None,
    selected_idx: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    extra = extra or {}
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(selected_idx, start=1):
        loc = int(locality[idx])
        row = {
            "trait": trait_name,
            "fold": int(fold),
            "method": method,
            "n_train_size": int(n_train),
            "rank": int(rank),
            "ringnumber": str(ids[idx]),
            "source_island": loc,
            "source_island_name": island_label(loc, code_to_label),
        }
        row.update(extra)
        rows.append(row)
    return rows


def write_summary(results_path: Path, summary_path: Path) -> None:
    if not results_path.exists():
        return
    df = pd.read_csv(results_path)
    if df.empty:
        return
    group_cols = [
        c
        for c in ["trait", "method", "selection_method", "lambda_div", "n_individuals"]
        if c in df.columns
    ]
    summary = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_std=("pearson_r", "std"),
            mse_adj_mean=("mse_adj", "mean"),
            n_rows=("pearson_r", "size"),
        )
        .sort_values(group_cols)
    )
    summary.to_csv(summary_path, index=False)
