"""Synthesize a 'source ridge results' JSON per trait whose per-fold weighting is
the m=top_m, lambda=lam re-selected importance-weighting config.

The tested ``nested_cv_importance_weighted_bpcrr_from_ridge`` runner reads a merged
ridge importance-weighted results JSON and, for each outer fold, copies that fold's
selected weighting hyperparameters to fit weighted BPCRR. By default it would copy
the run-time selection (all islands, lambda=0). This script instead writes a source
JSON whose ``best_params_per_fold[*].best_params.weighting`` is the trial chosen
under a given rule (default top_m=5, lambda=0, ESS-guarded), so weighted BPCRR
follows the m=5/lambda=0 importance config.

Run once (needs the GRM, so the master_thesis env / cluster):
    python -m scripts.make_reselected_ridge_source --top-m 5 --lambda 0
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.posthoc_reselect_importance import (
    fold_grm_ranking,
    load_trait_features,
    reselect_trial,
    score_trial,
)

TRAIT_ORDER = ["body_mass", "thr_tarsus", "thr_wing"]


def synthesize_source(
    trait: str,
    repo_root: Path,
    config_dir: Path,
    results_root: Path,
    result_subdir_suffix: str,
    top_m: int,
    lam: float,
) -> dict[str, Any]:
    cfg_path = config_dir / f"final_e6_pc_ridge_importance_{trait}_config.json"
    res_path = results_root / f"{trait}{result_subdir_suffix}" / f"e6_pc_ridge_importance_{trait}_results.json"
    if not cfg_path.exists() or not res_path.exists():
        raise FileNotFoundError(f"Missing config/results for {trait}: {cfg_path} / {res_path}")

    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload = json.loads(res_path.read_text(encoding="utf-8"))

    feats = load_trait_features(config, repo_root=repo_root, with_pca=False)  # GRM + locality only
    locality, code_to_label, grm_mat = feats["locality"], feats["code_to_label"], feats["grm_mat"]
    if grm_mat is None:
        raise RuntimeError(f"{trait}: GRM required to re-select top-{top_m} related islands.")

    best_params_per_fold: list[dict[str, Any]] = []
    per_fold_metrics: list[dict[str, Any]] = []

    for fh in payload.get("trial_history_per_fold", []) or []:
        fold = int(fh["fold"])
        isl = fh.get("test_island")
        isl_name = fh.get("test_island_name")
        idx_test = np.where(locality == isl)[0]
        idx_train = np.where(locality != isl)[0]
        grm_by_island = fold_grm_ranking(grm_mat, locality, idx_train, idx_test, code_to_label)

        chosen = reselect_trial(fh.get("trials") or [], top_m, lam, True, grm_by_island)
        if chosen is None:
            raise RuntimeError(f"{trait} fold {fold}: no trial selectable under top_m={top_m}, lambda={lam}.")
        weight_spec = chosen.get("weighting") or {"name": "uniform"}
        params = chosen.get("params") or {}
        inner_score = score_trial(chosen.get("inner_island_metrics"), top_m, lam, grm_by_island)

        best_params_per_fold.append({
            "fold": fold,
            "best_params": {
                "model_type": "pc_ridge",
                "alpha": params.get("alpha"),
                "use_snp_selection": False,  # the importance run used PC features, no SNP selection
                "num_snps": None,
                "n_pcs": params.get("n_pcs"),
                "weighting": weight_spec,
            },
            "mean_inner_r": None if inner_score is None else float(inner_score),
            "reselection_rule": {"top_m": int(top_m), "lambda": float(lam)},
            "source_trial_number": int(chosen["number"]),
        })
        per_fold_metrics.append({
            "fold": fold,
            "test_island": None if isl is None else int(isl),
            "test_island_name": isl_name,
            "test_corr": None,  # ridge outer r not refit here; weighted BPCRR is what we fit downstream
            "weighting": weight_spec,
        })

    del feats
    gc.collect()

    return {
        "mode": "ridge",
        "model_type": "ridge",
        "weighting_family": "importance_weighting",
        "cv_strategy": "leave_island_out",
        "source_results": str(res_path),
        "reselection_rule": {"top_m": int(top_m), "lambda": float(lam)},
        "importance_weighting_method_choices": payload.get("importance_weighting_method_choices", []),
        "best_params_per_fold": best_params_per_fold,
        "per_fold_metrics": per_fold_metrics,
    }


def main() -> None:
    from scripts.posthoc_reselect_importance import find_repo_root

    repo_root = find_repo_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traits", default=",".join(TRAIT_ORDER))
    parser.add_argument("--config-dir", default=str(repo_root / "config" / "e6"))
    parser.add_argument("--results-root", default=str(repo_root / "outputs" / "final_results" / "e6_pc_ridge_importance"))
    parser.add_argument("--result-subdir-suffix", default="_2")
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.0)
    parser.add_argument("--out-dir", default=str(repo_root / "outputs" / "nested_cv" / "bpcrr" / "weighted_from_m5l0"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for trait in [t.strip() for t in args.traits.split(",") if t.strip()]:
        summary = synthesize_source(
            trait=trait, repo_root=repo_root, config_dir=Path(args.config_dir),
            results_root=Path(args.results_root), result_subdir_suffix=args.result_subdir_suffix,
            top_m=args.top_m, lam=args.lam,
        )
        out_path = out_dir / f"{trait}_ridge_source.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        n = len(summary["best_params_per_fold"])
        non_uniform = sum(
            1 for b in summary["best_params_per_fold"]
            if str((b["best_params"]["weighting"] or {}).get("name", "uniform")).lower() != "uniform"
        )
        print(f"[{trait}] wrote {out_path} ({n} folds; {non_uniform} non-uniform weighting)")


if __name__ == "__main__":
    main()
