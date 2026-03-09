#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from sklearn.linear_model import Ridge

from src.cv_utils import ISLAND_ID_TO_NAME, island_label, make_outer_splits
from src.data import load_data
from src.utils import _pearson_corr, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_island_code(value: Any, code_to_label: Dict[int, Any], present_codes: Set[int]) -> int:
    try:
        val = int(value)
    except (ValueError, TypeError):
        val = None

    if val is not None:
        if val in present_codes:
            return val
        for code, label in code_to_label.items():
            if int(label) == val:
                return int(code)

    if isinstance(value, str):
        target_lower = value.lower()
        for orig_label, name in ISLAND_ID_TO_NAME.items():
            if name.lower() == target_lower:
                for code, lbl in code_to_label.items():
                    if int(lbl) == int(orig_label):
                        return int(code)
        for code, label in code_to_label.items():
            if str(label).lower() == target_lower:
                return int(code)

    raise ValueError(f"Could not resolve island: {value!r}")


def _build_trait_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    traits_cfg = cfg.get("traits", None)
    if traits_cfg is None:
        paths = dict(cfg["paths"])
        return [{
            "name": "default",
            "paths": paths,
            "target_column": cfg.get("target_column", "y_adjusted"),
            "eval_target_column": cfg.get("eval_target_column", "y_mean"),
            "standardize_features": cfg.get("standardize_features", False),
            "min_count": int(cfg.get("min_count", 20)),
        }]

    specs: List[Dict[str, Any]] = []
    for t in traits_cfg:
        if "name" not in t or "npz" not in t:
            raise ValueError("Each trait must define 'name' and 'npz'.")
        specs.append({
            "name": str(t["name"]),
            "paths": {"npz": str(t["npz"])},
            "target_column": t.get("target_column", cfg.get("target_column", "y_adjusted")),
            "eval_target_column": t.get("eval_target_column", cfg.get("eval_target_column", "y_mean")),
            "standardize_features": bool(t.get("standardize_features", cfg.get("standardize_features", False))),
            "min_count": int(t.get("min_count", cfg.get("min_count", 20))),
        })
    return specs


def _filter_include_islands(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    locality: np.ndarray,
    y_eval: np.ndarray,
    include_islands: Optional[List[Any]],
    code_to_label: Dict[int, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not include_islands:
        return X, y, ids, locality, y_eval

    present_codes = set(int(c) for c in np.unique(locality))
    include_codes = {
        _resolve_island_code(v, code_to_label, present_codes)
        for v in include_islands
    }
    mask = np.isin(locality, list(include_codes))

    if not np.any(mask):
        raise ValueError("include_islands filtered out all samples.")

    return X[mask], y[mask], ids[mask], locality[mask], y_eval[mask]


def run_ridge_loio(config: Dict[str, Any], target_islands_override: Optional[List[str]] = None) -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    cv_cfg = config.get("cv", {})
    strategy = str(cv_cfg.get("strategy", "leave_island_out")).lower()
    if strategy != "leave_island_out":
        raise ValueError("This runner currently supports only cv.strategy='leave_island_out'.")

    include_islands = cv_cfg.get("include_islands", None)
    selected_test_islands = cv_cfg.get("selected_test_islands", None)

    model_cfg = config.get("ridge", {})
    alpha = float(model_cfg.get("alpha", 1e5))

    output_cfg = config.get("output", {})
    output_root = Path(output_cfg.get("root_dir", "outputs/nested_cv"))
    file_stem = str(output_cfg.get("file_stem", "ridge"))

    trait_specs = _build_trait_specs(config)

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        logger.info("Running trait '%s'", trait_name)

        X, y, ids, _, locality, code_to_label, y_eval = load_data(
            paths=trait_spec["paths"],
            target_column=trait_spec["target_column"],
            standardize_features=trait_spec["standardize_features"],
            return_locality=True,
            min_count=trait_spec["min_count"],
            return_eval=True,
            eval_target_column=trait_spec["eval_target_column"],
        )
        if y_eval is None:
            y_eval = y.copy()

        X, y, ids, locality, y_eval = _filter_include_islands(
            X, y, ids, locality, y_eval, include_islands, code_to_label,
        )

        present_codes = set(int(c) for c in np.unique(locality))
        selected_test_codes: Optional[Set[int]] = None

        if target_islands_override:
            selected_test_codes = {
                _resolve_island_code(v, code_to_label, present_codes)
                for v in target_islands_override
            }
        elif selected_test_islands:
            selected_test_codes = {
                _resolve_island_code(v, code_to_label, present_codes)
                for v in selected_test_islands
            }

        outer_results: List[float] = []
        per_fold: List[Dict[str, Any]] = []
        fold_i = 0

        for tr_idx, te_idx, test_island_code in make_outer_splits(
            strategy="leave_island_out",
            locality=locality,
            n_splits=len(np.unique(locality)),
            shuffle=False,
            random_state=seed,
            n=len(X),
        ):
            if test_island_code is None:
                continue
            test_island_code = int(test_island_code)
            if selected_test_codes is not None and test_island_code not in selected_test_codes:
                continue

            fold_i += 1
            model = Ridge(alpha=max(alpha, 1e-12))
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[te_idx])
            r = float(_pearson_corr(y_eval[te_idx], pred))
            if not np.isfinite(r):
                r = 0.0

            outer_results.append(r)
            per_fold.append({
                "fold": int(fold_i),
                "pearson_r": float(r),
                "test_island": island_label(test_island_code, code_to_label),
                "test_island_code": int(test_island_code),
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
            })
            logger.info(
                "Trait=%s fold=%d island=%s r=%.4f",
                trait_name,
                fold_i,
                island_label(test_island_code, code_to_label),
                r,
            )

        out_dir = output_root / trait_name / "loio"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "mode": "ridge",
            "cv_strategy": "leave_island_out",
            "per_fold": per_fold,
            "overall": {
                "pearson_r": float(np.mean(outer_results)) if outer_results else None,
            },
            "outer_test_corr": outer_results,
            "outer_test_corr_mean": float(np.mean(outer_results)) if outer_results else None,
            "outer_test_corr_std": float(np.std(outer_results)) if outer_results else None,
            "outer_splits": int(len(outer_results)),
            "inner_splits": None,
            "best_params_per_fold": [
                {
                    "fold": int(row["fold"]),
                    "best_params": {
                        "model_type": "ridge",
                        "alpha": float(alpha),
                    },
                    "mean_inner_r": None,
                }
                for row in per_fold
            ],
            "config_used": {
                "alpha": float(alpha),
                "include_islands": include_islands,
                "selected_test_islands": list(selected_test_codes) if selected_test_codes is not None else None,
                "target_column": trait_spec["target_column"],
                "eval_target_column": trait_spec["eval_target_column"],
            },
        }

        out_path = out_dir / f"{file_stem}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "Saved ridge LOIO results for trait '%s' to %s (mean r=%.4f)",
            trait_name,
            out_path,
            summary["outer_test_corr_mean"] if summary["outer_test_corr_mean"] is not None else float("nan"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ridge leave-one-island-out runner")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--target_islands",
        nargs="+",
        default=None,
        help="Optional override for cv.selected_test_islands",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    run_ridge_loio(cfg, target_islands_override=args.target_islands)


if __name__ == "__main__":
    main()
