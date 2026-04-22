from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from src.nested_cv_avggrm_weighted_unified import run_merge as run_unified_merge
from src.nested_cv_avggrm_weighted_unified import (
    run_nested_cv_avggrm_weighted_unified,
)


def _ensure_ridge_model_type(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(config)
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["type"] = "ridge"
    cfg["model"] = model_cfg
    return cfg


def run_merge(config: Dict[str, Any], config_path: Optional[Path] = None):
    return run_unified_merge(_ensure_ridge_model_type(config), config_path=config_path)


def run_nested_cv_avggrm_weighted_ridge(config: Dict[str, Any], config_path: Optional[Path] = None):
    return run_nested_cv_avggrm_weighted_unified(_ensure_ridge_model_type(config), config_path=config_path)


def main():
    parser = argparse.ArgumentParser(description="Nested CV (Ridge) with AvgGRM-weight hyperparameter tuning")
    parser.add_argument("--mode", choices=["worker", "merge"], default="worker")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--selected_splits",
        type=str,
        default=None,
        help="Optional: JSON list or comma-separated 1-based outer split indices to run (e.g., '[10,11]' or '10,11'). Use 'false' to disable.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.selected_splits is not None:
        s = args.selected_splits.strip()
        if s.lower() not in ("false", "none", "", "0"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    cfg.setdefault("cv", {})["selected_splits"] = [int(x) for x in parsed]
                else:
                    vals = [int(x) for x in s.split(",") if x.strip()]
                    cfg.setdefault("cv", {})["selected_splits"] = vals
            except Exception:
                try:
                    vals = [int(x) for x in s.split(",") if x.strip()]
                    cfg.setdefault("cv", {})["selected_splits"] = vals
                except Exception as exc:
                    raise ValueError(
                        "--selected_splits must be a JSON list or comma-separated integers, or 'false'."
                    ) from exc

    if args.mode == "merge":
        run_merge(cfg, config_path=config_path)
        return

    run_nested_cv_avggrm_weighted_ridge(cfg, config_path=config_path)


if __name__ == "__main__":
    main()
