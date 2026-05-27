#!/usr/bin/env python
"""Fit fixed PC ridge models on selected or random training subsets."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from src.cv_utils import island_label
from src.data import load_data
from src.utils import _pearson_corr, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _trait_specs(config: dict[str, Any], only_traits: set[str] | None) -> list[dict[str, Any]]:
    specs = []
    for raw in config.get("traits", []):
        name = str(raw["name"])
        if only_traits is not None and name not in only_traits:
            continue
        spec = {
            "name": name,
            "npz": str(raw["npz"]),
            "n_pcs": int(raw.get("n_pcs", config["pc_ridge"]["n_pcs"])),
            "alpha": float(raw.get("alpha", config["pc_ridge"]["alpha"])),
            "target_column": raw.get("target_column", config.get("target_column", "y_adjusted")),
            "eval_target_column": raw.get("eval_target_column", config.get("eval_target_column", "y_mean")),
            "standardize_features": bool(raw.get("standardize_features", config.get("standardize_features", False))),
            "min_count": int(raw.get("min_count", config.get("min_count", 20))),
        }
        specs.append(spec)
    if not specs:
        requested = "<all>" if only_traits is None else ", ".join(sorted(only_traits))
        raise ValueError(f"No trait specs selected for: {requested}")
    return specs


def _compute_pcs(
    X: np.ndarray,
    n_pcs: int,
    seed: int,
    standardize: bool,
) -> tuple[np.ndarray, float]:
    n_components = int(min(n_pcs, X.shape[0], X.shape[1]))
    if n_components < 1:
        raise ValueError("n_pcs must leave at least one component")

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed))
    if standardize:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std == 0] = 1.0
        X_in = (X - mean) / std
    else:
        X_in = X
    scores = pca.fit_transform(X_in).astype(np.float32, copy=False)
    return scores, float(np.sum(pca.explained_variance_ratio_))


def _load_trait_data(
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, Any], np.ndarray]:
    X, y, ids, _, locality, code_to_label, y_eval = load_data(
        paths={"npz": spec["npz"]},
        target_column=spec["target_column"],
        standardize_features=spec["standardize_features"],
        return_locality=True,
        min_count=spec["min_count"],
        return_eval=True,
        eval_target_column=spec["eval_target_column"],
    )
    if y_eval is None:
        y_eval = y.copy()
    return X, y, ids.astype(str), locality.astype(int), code_to_label, y_eval


def _selected_inputs_for_trait(config: dict[str, Any], trait_name: str) -> list[Path]:
    subset_cfg = config["subsets"]
    root = Path(subset_cfg["root_dir"]) / trait_name / subset_cfg.get("selected_dir_name", "selected_individuals")
    pattern = str(subset_cfg.get("file_pattern", "k_*.csv"))
    paths = sorted(root.glob(pattern), key=_k_from_path)
    if not paths:
        raise FileNotFoundError(f"No selected-subset files found under {root} matching {pattern}")
    return paths


def _k_from_path(path: Path) -> int:
    stem = path.stem
    if stem.startswith("k_"):
        return int(stem.split("_", 1)[1])
    return int(stem)


def _load_selected_jobs(paths: Iterable[Path]) -> Iterable[tuple[int, tuple[Any, ...], pd.DataFrame]]:
    for path in paths:
        k = _k_from_path(path)
        df = pd.read_csv(path)
        required = {"target_island", "target_island_name", "repeat", "repeat_seed", "ringnumber"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        group_cols = ["target_island", "target_island_name", "repeat", "repeat_seed"]
        for key, group in df.groupby(group_cols, sort=True, dropna=False):
            yield k, key, group


def _load_random_template(config: dict[str, Any], trait_name: str) -> pd.DataFrame:
    random_cfg = config["random"]
    if "template_results_root" in random_cfg:
        path = Path(random_cfg["template_results_root"]) / trait_name / random_cfg.get(
            "template_results_name", "pevmean_ga_results.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if "n_train_size" not in df.columns and "n_individuals" in df.columns:
            df = df.rename(columns={"n_individuals": "n_train_size"})
        needed = ["target_island", "target_island_name", "repeat", "repeat_seed", "n_train_size"]
        missing = set(needed).difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        template = df[needed].drop_duplicates().sort_values(needed).reset_index(drop=True)

        if "n_repeats" not in random_cfg:
            return template

        n_repeats = int(random_cfg["n_repeats"])
        seed = int(random_cfg.get("seed", config.get("seed", 14)))
        existing = {
            (
                int(row.target_island),
                int(row.n_train_size),
                int(row.repeat),
            ): int(row.repeat_seed)
            for row in template.itertuples(index=False)
        }
        target_grid = (
            template[["target_island", "target_island_name", "n_train_size"]]
            .drop_duplicates()
            .sort_values(["target_island", "n_train_size"])
        )
        rows = []
        for row in target_grid.itertuples(index=False):
            target = int(row.target_island)
            k = int(row.n_train_size)
            for repeat in range(n_repeats):
                repeat_seed = existing.get(
                    (target, k, repeat),
                    seed + target * 100000 + repeat,
                )
                rows.append(
                    {
                        "target_island": target,
                        "target_island_name": row.target_island_name,
                        "repeat": repeat,
                        "repeat_seed": repeat_seed,
                        "n_train_size": k,
                    }
                )
        return pd.DataFrame(rows)

    target_islands = [int(x) for x in random_cfg["target_islands"]]
    n_train_sizes = [int(x) for x in random_cfg["n_train_sizes"]]
    n_repeats = int(random_cfg.get("n_repeats", 5))
    seed = int(random_cfg.get("seed", config.get("seed", 14)))
    rows = []
    for target in target_islands:
        for repeat in range(n_repeats):
            for k in n_train_sizes:
                rows.append(
                    {
                        "target_island": target,
                        "target_island_name": "",
                        "repeat": repeat,
                        "repeat_seed": seed + target * 100000 + repeat,
                        "n_train_size": k,
                    }
                )
    return pd.DataFrame(rows)


def _fit_eval(
    X_pcs: np.ndarray,
    y: np.ndarray,
    y_eval: np.ndarray,
    locality: np.ndarray,
    train_idx: np.ndarray,
    target_island: int,
    alpha: float,
) -> dict[str, Any]:
    test_idx = np.flatnonzero(locality == int(target_island))
    if len(test_idx) == 0:
        raise ValueError(f"No test individuals found for target island {target_island}")
    if len(train_idx) == 0:
        raise ValueError("Training subset is empty")

    model = Ridge(alpha=max(float(alpha), 1e-12))
    t0 = time.perf_counter()
    model.fit(X_pcs[train_idx], y[train_idx])
    pred = model.predict(X_pcs[test_idx])
    fit_time = float(time.perf_counter() - t0)

    r = float(_pearson_corr(y_eval[test_idx], pred))
    if not np.isfinite(r):
        r = 0.0
    mse = float(np.mean((y_eval[test_idx] - pred) ** 2))
    return {
        "pearson_r": r,
        "mse_adj": mse,
        "fit_time_seconds": fit_time,
        "n_test": int(len(test_idx)),
    }


def _clean_target_name(value: Any, target_island: int, code_to_label: dict[int, Any]) -> str:
    if pd.isna(value):
        return island_label(target_island, code_to_label)
    text = str(value)
    if text == "":
        return island_label(target_island, code_to_label)
    return text


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


class SplitSelectedWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.handles: dict[int, Any] = {}
        self.writers: dict[int, csv.DictWriter] = {}
        self.fieldnames = [
            "trait",
            "target_island",
            "target_island_name",
            "repeat",
            "repeat_seed",
            "n_train_size",
            "method",
            "ringnumber",
            "source_island",
            "source_island_name",
        ]

    def writerow(self, k: int, row: dict[str, Any]) -> None:
        if k not in self.writers:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            handle = (self.output_dir / f"k_{int(k)}.csv").open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            self.handles[k] = handle
            self.writers[k] = writer
        self.writers[k].writerow(row)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def _write_summary(results_path: Path, summary_path: Path) -> None:
    if not results_path.exists():
        return
    df = pd.read_csv(results_path)
    if df.empty:
        return
    summary = (
        df.groupby(["trait", "method", "n_train_size"], as_index=False)
        .agg(
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_std=("pearson_r", "std"),
            mse_adj_mean=("mse_adj", "mean"),
            mse_adj_std=("mse_adj", "std"),
            n_evaluations=("pearson_r", "size"),
            n_pcs=("n_pcs", "first"),
            alpha=("alpha", "first"),
            fit_time_total_seconds=("fit_time_seconds", "sum"),
        )
        .sort_values(["trait", "method", "n_train_size"])
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)


def run_selected_mode(
    config: dict[str, Any],
    spec: dict[str, Any],
    X_pcs: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    locality: np.ndarray,
    code_to_label: dict[int, Any],
    y_eval: np.ndarray,
    results_path: Path,
) -> None:
    id_to_idx = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids)
    rows: list[dict[str, Any]] = []

    for k, key, group in _load_selected_jobs(_selected_inputs_for_trait(config, spec["name"])):
        target_island, target_name, repeat, repeat_seed = key
        selected_ids = group["ringnumber"].astype(str)
        missing = selected_ids[~selected_ids.isin(id_to_idx.index)]
        if len(missing) > 0:
            raise ValueError(f"{len(missing)} selected IDs are not present in {spec['name']} data")
        train_idx = id_to_idx.loc[selected_ids].to_numpy(dtype=np.int64)
        metrics = _fit_eval(
            X_pcs=X_pcs,
            y=y,
            y_eval=y_eval,
            locality=locality,
            train_idx=train_idx,
            target_island=int(target_island),
            alpha=float(spec["alpha"]),
        )
        rows.append(
            {
                "trait": spec["name"],
                "method": config["run"]["method"],
                "target_island": int(target_island),
                "target_island_name": str(target_name),
                "repeat": int(repeat),
                "repeat_seed": int(repeat_seed),
                "n_train_size": int(k),
                "n_train": int(len(train_idx)),
                "n_pcs": int(X_pcs.shape[1]),
                "n_pcs_requested": int(spec["n_pcs"]),
                "alpha": float(spec["alpha"]),
                **metrics,
            }
        )
        if len(rows) >= 100:
            _append_csv(results_path, rows)
            rows = []
    _append_csv(results_path, rows)


def run_random_mode(
    config: dict[str, Any],
    spec: dict[str, Any],
    X_pcs: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    locality: np.ndarray,
    code_to_label: dict[int, Any],
    y_eval: np.ndarray,
    results_path: Path,
    output_root: Path,
) -> None:
    random_cfg = config["random"]
    write_selected = bool(random_cfg.get("write_selected_individuals", True))
    selected_writer = (
        SplitSelectedWriter(output_root / spec["name"] / random_cfg.get("selected_dir_name", "selected_individuals"))
        if write_selected
        else None
    )
    rows: list[dict[str, Any]] = []

    try:
        template = _load_random_template(config, spec["name"])
        for _, job in template.iterrows():
            target_island = int(job["target_island"])
            k = int(job["n_train_size"])
            repeat = int(job["repeat"])
            repeat_seed = int(job["repeat_seed"])
            source_idx = np.flatnonzero(locality != target_island)
            if k > len(source_idx):
                raise ValueError(
                    f"Requested k={k} for trait {spec['name']} target {target_island}, "
                    f"but only {len(source_idx)} source individuals are available"
                )
            rng = np.random.default_rng(repeat_seed)
            train_idx = np.sort(rng.choice(source_idx, size=k, replace=False)).astype(np.int64)
            target_name = _clean_target_name(job["target_island_name"], target_island, code_to_label)

            metrics = _fit_eval(
                X_pcs=X_pcs,
                y=y,
                y_eval=y_eval,
                locality=locality,
                train_idx=train_idx,
                target_island=target_island,
                alpha=float(spec["alpha"]),
            )
            rows.append(
                {
                    "trait": spec["name"],
                    "method": config["run"]["method"],
                    "target_island": target_island,
                    "target_island_name": target_name,
                    "repeat": repeat,
                    "repeat_seed": repeat_seed,
                    "n_train_size": k,
                    "n_train": int(len(train_idx)),
                    "n_pcs": int(X_pcs.shape[1]),
                    "n_pcs_requested": int(spec["n_pcs"]),
                    "alpha": float(spec["alpha"]),
                    **metrics,
                }
            )

            if selected_writer is not None:
                for idx in train_idx:
                    source_island = int(locality[idx])
                    selected_writer.writerow(
                        k,
                        {
                            "trait": spec["name"],
                            "target_island": target_island,
                            "target_island_name": target_name,
                            "repeat": repeat,
                            "repeat_seed": repeat_seed,
                            "n_train_size": k,
                            "method": config["run"]["method"],
                            "ringnumber": ids[idx],
                            "source_island": source_island,
                            "source_island_name": island_label(source_island, code_to_label),
                        },
                    )

            if len(rows) >= 100:
                _append_csv(results_path, rows)
                rows = []
        _append_csv(results_path, rows)
    finally:
        if selected_writer is not None:
            selected_writer.close()


def run(config: dict[str, Any], only_traits: set[str] | None = None) -> None:
    seed = int(config.get("seed", 14))
    set_seed(seed)

    mode = str(config["run"]["mode"]).lower()
    if mode not in {"selected", "random"}:
        raise ValueError("run.mode must be 'selected' or 'random'")

    output_root = Path(config["output"]["root_dir"])
    file_stem = str(config["output"]["file_stem"])
    overwrite = bool(config["output"].get("overwrite", True))
    output_root.mkdir(parents=True, exist_ok=True)

    config_used = output_root / f"{file_stem}_config_used.json"
    with config_used.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    for spec in _trait_specs(config, only_traits):
        trait_output_root = output_root / spec["name"]
        trait_output_root.mkdir(parents=True, exist_ok=True)
        results_path = trait_output_root / f"{file_stem}_results.csv"
        summary_path = trait_output_root / f"{file_stem}_summary.csv"
        if overwrite:
            for path in (results_path, summary_path):
                if path.exists():
                    path.unlink()

        logger.info(
            "Running trait=%s mode=%s n_pcs=%d alpha=%.6g",
            spec["name"],
            mode,
            int(spec["n_pcs"]),
            float(spec["alpha"]),
        )
        X, y, ids, locality, code_to_label, y_eval = _load_trait_data(spec)
        t0 = time.perf_counter()
        X_pcs, var_explained = _compute_pcs(
            X=X,
            n_pcs=int(spec["n_pcs"]),
            seed=seed,
            standardize=bool(spec["standardize_features"]),
        )
        logger.info(
            "Trait=%s computed %d PCs in %.2fs; variance explained %.4f",
            spec["name"],
            int(X_pcs.shape[1]),
            float(time.perf_counter() - t0),
            var_explained,
        )

        if mode == "selected":
            run_selected_mode(config, spec, X_pcs, y, ids, locality, code_to_label, y_eval, results_path)
        else:
            run_random_mode(
                config,
                spec,
                X_pcs,
                y,
                ids,
                locality,
                code_to_label,
                y_eval,
                results_path,
                output_root,
            )

        _write_summary(results_path, summary_path)
        logger.info("Saved results to %s", results_path)
        logger.info("Saved summary to %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trait", action="append", default=None, help="Trait to run; can be repeated.")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    only_traits = set(_as_list(args.trait)) if args.trait else None
    run(config, only_traits=only_traits)


if __name__ == "__main__":
    main()
