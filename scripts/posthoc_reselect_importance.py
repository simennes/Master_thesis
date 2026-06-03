"""Post-hoc re-selection of E6 importance-weighted PC-ridge configs.

The E6 ``<trait>_2`` runs record, in each trial's ``inner_island_metrics``, the
per-inner-island validation r for *all* training islands (no top-k restriction).
That lets us re-run *model selection* under any rule without re-tuning: rank the
inner islands by GRM relatedness to the held-out island, keep the ``top_m`` most
related, score each trial by ``mean - lambda * SE`` over those islands, take the
argmax trial, then **refit that config on the outer training set and evaluate it
on the held-out island**. The outer refit is necessary because the run only
stored the outer r of the config it actually selected.

The ``<trait>_2`` runs did not store ``avg_grm_to_outer_test`` (they ran without
``inner_top_k_related_islands``), so this script recomputes the GRM ranking from
the GRM matrix for each outer fold.

Outputs a long table (one row per trait x fold x rule) of outer-fold r.

Usage
-----
    python -m scripts.posthoc_reselect_importance \
        --top-m 3,5,8,10,15 --lambda 0,0.5,1,1.5 --plot

Run it where the SNP .npz files and the GRM .rds are available with enough RAM
for the PCA (recomputes the same PCA the runner used).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.avggrm_weighting import rank_inner_validation_islands_by_avg_grm
from src.data import load_data
from src.importance_weighting import compute_pc_logistic_importance_weights
from src.pc_feature import maybe_apply_pca, pca_active
from src.training_set_optimization.runner import _evaluate_ridge_subset
from src.utils import set_seed


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("posthoc_reselect")

TRAIT_LABELS = {"body_mass": "Body mass", "thr_tarsus": "Tarsus length", "thr_wing": "Wing length"}
TRAIT_ORDER = ["body_mass", "thr_tarsus", "thr_wing"]


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "Data").exists() and (path / "src").exists():
            return path
        candidate = path / "Master_thesis"
        if (candidate / "Data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not find the Master_thesis repo root.")


# --------------------------------------------------------------------------- #
# Feature reconstruction (mirrors the runner's load + filter + PCA)
# --------------------------------------------------------------------------- #
def _include_island_indices(
    locality: np.ndarray,
    include_islands: Any,
    code_to_label: Optional[dict[int, str]],
) -> Optional[np.ndarray]:
    """Indices to keep for ``cv.include_islands`` (None => keep all)."""
    if not include_islands:
        return None
    label_to_code = {str(v): int(k) for k, v in (code_to_label or {}).items()}
    present = set(np.unique(locality).astype(int).tolist())
    codes: set[int] = set()
    for val in include_islands:
        sval = str(val)
        if sval in label_to_code:
            codes.add(label_to_code[sval])
            continue
        try:
            ival = int(val)
            if ival in present:
                codes.add(ival)
        except (TypeError, ValueError):
            pass
    if not codes:
        raise ValueError(f"include_islands={include_islands} matched no islands; present codes={sorted(present)}")
    return np.where(np.isin(locality.astype(int), np.fromiter(codes, dtype=int)))[0]


def _resolve_paths(paths: dict[str, Any], repo_root: Optional[Path]) -> dict[str, Any]:
    """Make relative data paths absolute against repo_root (the configs store
    paths relative to the repo root; a notebook's cwd is usually elsewhere)."""
    if repo_root is None:
        return dict(paths)
    out = dict(paths)
    for key in ("npz", "npz_path", "grm_rds", "snp_feather", "phenotype_csv"):
        val = out.get(key)
        if isinstance(val, str) and val and not Path(val).is_absolute():
            out[key] = str(Path(repo_root) / val)
    return out


def _cap_pca_fit(config: dict[str, Any], max_pca_pcs: Optional[int]) -> dict[str, Any]:
    """Optionally cap the number of PCs the PCA fits (memory lever; APPROXIMATE —
    randomized SVD with fewer components yields slightly different PCs, and any
    config wanting more PCs than the cap is clamped to it)."""
    if not max_pca_pcs:
        return config
    config = json.loads(json.dumps(config))  # cheap deep copy
    fp = config.get("feature_pca")
    if not isinstance(fp, dict):
        fp = config.setdefault("base_train", {}).get("feature_pca")
    if isinstance(fp, dict):
        cur = int(fp.get("n_pcs_fit", max_pca_pcs))
        fp["n_pcs_fit"] = int(min(cur, int(max_pca_pcs)))
    return config


def load_trait_features(
    config: dict[str, Any],
    repo_root: Optional[Path] = None,
    max_pca_pcs: Optional[int] = None,
    with_pca: bool = True,
) -> dict[str, Any]:
    """Rebuild the labels and aligned GRM (and, if ``with_pca``, the PCA matrix).

    ``with_pca=False`` skips the expensive PCA step — useful when only the GRM
    rankings are needed (e.g. choosing a threshold)."""
    config = _cap_pca_fit(config, max_pca_pcs)
    base = config["base_train"]
    cv_cfg = config.get("cv", {})
    set_seed(int(base.get("seed", config.get("seed", 42))))

    X, y, ids, grm_df, locality, code_to_label, y_eval = load_data(
        _resolve_paths(base["paths"], repo_root),
        target_column=base.get("target_column", "y_adjusted"),
        standardize_features=base.get("standardize_features", False),
        return_locality=True,
        min_count=int(base.get("min_count", 20)),
        return_eval=True,
        eval_target_column=base.get("eval_target_column", "y_mean"),
    )
    if y_eval is None:
        y_eval = y.copy()

    keep = _include_island_indices(locality, cv_cfg.get("include_islands"), code_to_label)
    if keep is not None:
        X, y, y_eval, locality = X[keep], y[keep], y_eval[keep], locality[keep]
        if grm_df is not None:
            grm_df = grm_df.iloc[keep, keep]

    grm_mat = None if grm_df is None else grm_df.to_numpy(dtype=np.float64)

    if with_pca:
        pca_state = maybe_apply_pca(X, config)
        if pca_active(pca_state):
            X = pca_state["Z"]
    return {"X": X, "y": y, "y_eval": y_eval, "locality": locality, "code_to_label": code_to_label, "grm_mat": grm_mat}


def compute_fold_grm_rankings(config: dict[str, Any], repo_root: Optional[Path] = None) -> list[dict[int, float]]:
    """Per outer fold, the {inner island -> avg GRM to held-out island} map.
    Loads GRM + locality only (no PCA), so it is cheap enough to call for
    threshold selection."""
    feats = load_trait_features(config, repo_root=repo_root, with_pca=False)
    locality, code_to_label, grm_mat = feats["locality"], feats["code_to_label"], feats["grm_mat"]
    rankings: list[dict[int, float]] = []
    if grm_mat is not None:
        for isl in np.unique(locality):
            idx_test = np.where(locality == isl)[0]
            idx_train = np.where(locality != isl)[0]
            rankings.append(fold_grm_ranking(grm_mat, locality, idx_train, idx_test, code_to_label))
    del feats
    gc.collect()
    return rankings


def choose_grm_threshold(
    per_fold_rankings: list[dict[int, float]],
    target_count: float,
) -> tuple[float, float]:
    """Pick a single GRM threshold p so the mean number of inner islands with
    avg GRM > p (across folds) is as close as possible to ``target_count``."""
    values = sorted({g for r in per_fold_rankings for g in r.values()})
    if not values:
        raise ValueError("No GRM values available to choose a threshold from.")
    best_p, best_diff, best_mean = values[0], np.inf, float("nan")
    # Candidate thresholds: just below each observed value (so g>p keeps that value)
    # plus a tiny step below the minimum (keeps everything).
    candidates = [values[0] - 1e-9] + [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
    for p in candidates:
        counts = [sum(1 for g in r.values() if g > p) for r in per_fold_rankings]
        mean_count = float(np.mean(counts)) if counts else 0.0
        diff = abs(mean_count - float(target_count))
        if diff < best_diff:
            best_p, best_diff, best_mean = float(p), diff, mean_count
    return best_p, best_mean


def fold_grm_ranking(
    grm_mat: np.ndarray,
    locality: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    code_to_label: Optional[dict[int, str]],
) -> dict[int, float]:
    ranking = rank_inner_validation_islands_by_avg_grm(
        grm_mat=grm_mat, locality=locality, idx_outer_train=idx_train,
        idx_outer_test=idx_test, code_to_label=code_to_label,
    )
    return {int(r["island"]): float(r["avg_grm_to_outer_test"]) for r in ranking}


# --------------------------------------------------------------------------- #
# Re-selection rule
# --------------------------------------------------------------------------- #
def _grm_key(metric: dict[str, Any], grm_by_island: Optional[dict[int, float]]) -> float:
    if grm_by_island is not None:
        return grm_by_island.get(int(metric["inner_island"]), -np.inf)
    v = metric.get("avg_grm_to_outer_test")
    return -np.inf if v is None else float(v)


def select_inner_islands(
    metrics: list[dict[str, Any]],
    top_m: Optional[int],
    grm_by_island: Optional[dict[int, float]],
    grm_threshold: Optional[float],
) -> list[dict[str, Any]]:
    """Inner islands kept under the rule, ordered by GRM relatedness (desc).

    ``grm_threshold`` (keep islands with avg GRM > p) takes precedence over
    ``top_m`` (keep the m most related). The threshold rule never returns an
    empty set: if no island clears p, the single most-related island is kept.
    """
    ordered = sorted(metrics, key=lambda d: _grm_key(d, grm_by_island), reverse=True)
    if grm_threshold is not None:
        kept = [d for d in ordered if _grm_key(d, grm_by_island) > float(grm_threshold)]
        return kept if kept else ordered[:1]
    if top_m is not None:
        return ordered[: int(top_m)]
    return ordered


def score_trial(
    metrics: list[dict[str, Any]] | None,
    top_m: Optional[int],
    lam: float,
    grm_by_island: Optional[dict[int, float]] = None,
    grm_threshold: Optional[float] = None,
) -> Optional[float]:
    """Score a trial as ``mean - lam * SE`` over the selected inner islands."""
    if not metrics:
        return None
    sel = select_inner_islands(metrics, top_m, grm_by_island, grm_threshold)
    r = np.array([d["r"] for d in sel if d.get("r") is not None], dtype=float)
    if r.size == 0:
        return None
    mean = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(r.size)) if r.size > 1 else 0.0
    return mean - float(lam) * se


def reselect_trial(
    trials: list[dict[str, Any]],
    top_m: Optional[int],
    lam: float,
    respect_ess_guard: bool = True,
    grm_by_island: Optional[dict[int, float]] = None,
    grm_threshold: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    best, best_score = None, -np.inf
    for t in trials:
        if str(t.get("state")) != "COMPLETE":
            continue
        if respect_ess_guard and bool(t.get("effective_sample_size_rejected", False)):
            continue
        s = score_trial(t.get("inner_island_metrics"), top_m, lam, grm_by_island, grm_threshold)
        if s is None:
            continue
        if s > best_score:
            best_score, best = s, t
    return best


# --------------------------------------------------------------------------- #
# Outer refit
# --------------------------------------------------------------------------- #
def evaluate_config(
    feats: dict[str, Any],
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    alpha: float,
    n_pcs: int,
    weighting: Optional[dict[str, Any]],
) -> float:
    X, y, y_eval = feats["X"], feats["y"], feats["y_eval"]
    k = int(min(int(n_pcs), X.shape[1]))
    snp_cols = np.arange(k, dtype=np.int64)
    name = str((weighting or {}).get("name", "uniform")).lower()
    if name == "uniform":
        sample_weight = None
    else:
        wres = compute_pc_logistic_importance_weights(
            X=X, train_idx=idx_train, target_idx=idx_test,
            weight_cfg=weighting, feature_cols=snp_cols, precomputed_pcs=True,
        )
        sample_weight = wres["weights"]
    ev = _evaluate_ridge_subset(
        train_idx=idx_train, X_source=X, y_source=y,
        X_test=X[idx_test], y_test=y[idx_test], y_eval_test=y_eval[idx_test],
        alpha=float(alpha), snp_cols=snp_cols, sample_weight=sample_weight,
    )
    return float(ev["corr_eval"])


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _parse_top_m(text: str) -> list[Optional[int]]:
    out: list[Optional[int]] = []
    for tok in text.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        out.append(None if tok in ("all", "0", "none") else int(tok))
    return out


def _top_m_label(top_m: Optional[int]) -> str:
    return "all" if top_m is None else str(int(top_m))


def _config_path(config_dir: Path, trait: str) -> Path:
    return config_dir / f"final_e6_pc_ridge_importance_{trait}_config.json"


def _result_path(results_root: Path, trait: str, suffix: str) -> Path:
    return results_root / f"{trait}{suffix}" / f"e6_pc_ridge_importance_{trait}_results.json"


def run_reselection(
    repo_root: Path,
    traits: list[str],
    config_dir: Path,
    results_root: Path,
    top_m_values: list[Optional[int]],
    lambda_values: list[float],
    respect_ess_guard: bool = True,
    result_subdir_suffix: str = "_2",
    max_pca_pcs: Optional[int] = None,
    grm_thresholds: Optional[list[float]] = None,
    target_inner_count: Optional[float] = None,
) -> pd.DataFrame:
    config_dir, results_root = Path(config_dir), Path(results_root)
    grm_thresholds = list(grm_thresholds or [])
    rows: list[dict[str, Any]] = []

    # Phase 0: if a target island count was requested, pick a single global GRM
    # threshold (pooled over all traits/folds) that yields ~target islands/fold.
    if target_inner_count and not grm_thresholds:
        pooled: list[dict[int, float]] = []
        for trait in traits:
            cfg_path = _config_path(config_dir, trait)
            if not cfg_path.exists():
                continue
            logger.info("[%s] phase 0: GRM rankings for threshold selection ...", trait)
            pooled.extend(compute_fold_grm_rankings(json.loads(cfg_path.read_text(encoding="utf-8")), repo_root))
        p, achieved = choose_grm_threshold(pooled, float(target_inner_count))
        grm_thresholds = [p]
        logger.info("Chosen GRM threshold p=%.6f -> mean %.2f inner islands/fold (target %.1f)",
                    p, achieved, float(target_inner_count))

    # Unified rule list: ("top_m", m) and ("grm_threshold", p).
    rule_specs: list[tuple[str, Optional[float]]] = [("top_m", m) for m in top_m_values]
    rule_specs += [("grm_threshold", float(p)) for p in grm_thresholds]

    for trait in traits:
        cfg_path = _config_path(config_dir, trait)
        res_path = _result_path(results_root, trait, result_subdir_suffix)
        if not cfg_path.exists() or not res_path.exists():
            logger.warning("Skipping %s (missing config or results): %s / %s", trait, cfg_path, res_path)
            continue

        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        payload = json.loads(res_path.read_text(encoding="utf-8"))
        history = payload.get("trial_history_per_fold", []) or []
        if not any(t.get("inner_island_metrics") for fh in history for t in (fh.get("trials") or [])):
            logger.warning("%s: no per-island inner metrics in %s — skipping.", trait, res_path)
            continue

        logger.info("[%s] reconstructing PCA features + GRM ...", trait)
        feats = load_trait_features(config, repo_root=repo_root, max_pca_pcs=max_pca_pcs)
        locality, code_to_label, grm_mat = feats["locality"], feats["code_to_label"], feats["grm_mat"]
        if grm_mat is None:
            logger.warning("%s: no GRM available; cannot rank inner islands by relatedness — skipping.", trait)
            continue

        orig_outer = {int(m["fold"]): float(m["test_corr"]) for m in payload.get("per_fold_metrics", [])}
        cache: dict[tuple[int, int], float] = {}  # (fold, trial) -> outer r, evaluate each unique config once

        for fh in history:
            fold = int(fh["fold"])
            isl = fh.get("test_island")
            isl_name = fh.get("test_island_name")
            trials = fh.get("trials") or []
            idx_test = np.where(locality == isl)[0]
            idx_train = np.where(locality != isl)[0]
            grm_by_island = fold_grm_ranking(grm_mat, locality, idx_train, idx_test, code_to_label)
            rep_metrics = next((t.get("inner_island_metrics") for t in trials if t.get("inner_island_metrics")), None)

            def eval_chosen(chosen: dict[str, Any]) -> float:
                tnum = int(chosen["number"])
                key = (fold, tnum)
                if key not in cache:
                    p = chosen.get("params", {}) or {}
                    cache[key] = evaluate_config(
                        feats, idx_train, idx_test,
                        alpha=p.get("alpha"), n_pcs=p.get("n_pcs"), weighting=chosen.get("weighting"),
                    )
                return cache[key]

            for kind, param in rule_specs:
                top_m = int(param) if kind == "top_m" and param is not None else None
                thr = float(param) if kind == "grm_threshold" else None
                if kind == "top_m":
                    rule_label = _top_m_label(top_m)
                else:
                    rule_label = f"grm>{thr:.4f}"
                n_inner_used = (
                    len(select_inner_islands(rep_metrics, top_m, grm_by_island, thr)) if rep_metrics else None
                )
                for lam in lambda_values:
                    chosen = reselect_trial(trials, top_m, lam, respect_ess_guard, grm_by_island, thr)
                    if chosen is None:
                        continue
                    rows.append({
                        "trait": trait,
                        "trait_label": TRAIT_LABELS.get(trait, trait),
                        "fold": fold,
                        "test_island_code": isl,
                        "test_island": isl_name,
                        "rule_kind": kind,
                        "rule_label": rule_label,
                        "top_m": -1 if top_m is None else int(top_m),
                        "top_m_label": rule_label,  # back-compat: drives the existing figure's x-axis
                        "grm_threshold": thr,
                        "n_inner_used": n_inner_used,
                        "lambda": float(lam),
                        "selected_trial": int(chosen["number"]),
                        "weighting_method": str((chosen.get("weighting") or {}).get("name", "uniform")).lower(),
                        "n_pcs": (chosen.get("params") or {}).get("n_pcs"),
                        "alpha": (chosen.get("params") or {}).get("alpha"),
                        "outer_r": eval_chosen(chosen),
                    })

            # Sanity: 'all islands, lambda=0, ESS-guarded' reproduces the run-time selection.
            chk = reselect_trial(trials, None, 0.0, True, grm_by_island)
            if chk is not None and fold in orig_outer:
                diff = abs(eval_chosen(chk) - orig_outer[fold])
                if diff > 1e-4:
                    logger.warning(
                        "[%s] fold %d: refit r=%.4f vs recorded r=%.4f (|diff|=%.4f) — "
                        "feature reconstruction may differ from the run.",
                        trait, fold, cache[(fold, int(chk["number"]))], orig_outer[fold], diff,
                    )

        logger.info("[%s] done: %d unique outer refits.", trait, len(cache))
        del feats  # release the (large) PCA feature matrix before the next trait
        gc.collect()

    return pd.DataFrame(rows)


def plot_reselection_boxplots(df: pd.DataFrame, output_dir: Path, file_stem: str) -> Optional[tuple[Path, Path]]:
    if df.empty:
        logger.warning("No re-selection rows to plot.")
        return None
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600, "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    traits = [t for t in TRAIT_ORDER if t in set(df["trait"])]
    labels = list(df["top_m_label"].unique())
    ints = sorted([v for v in labels if str(v).isdigit()], key=int)
    others = [v for v in labels if not str(v).isdigit() and v != "all"]  # e.g. 'grm>0.0123'
    m_order = ints + (["all"] if "all" in labels else []) + others
    df = df.copy()
    df["top_m_label"] = pd.Categorical(df["top_m_label"], categories=m_order, ordered=True)
    df["lambda_str"] = df["lambda"].map(lambda x: f"{x:g}")
    lam_order = [f"{x:g}" for x in sorted(df["lambda"].unique())]

    fig, axes = plt.subplots(len(traits), 1, figsize=(9.2, 3.1 * len(traits)), constrained_layout=True, squeeze=False)
    for ax, trait in zip(axes[:, 0], traits):
        sub = df[df["trait"] == trait]
        sns.boxplot(
            data=sub, x="top_m_label", y="outer_r", hue="lambda_str",
            order=m_order, hue_order=lam_order, palette="viridis",
            width=0.74, fliersize=0, ax=ax, boxprops={"alpha": 0.78},
            medianprops={"color": "black", "linewidth": 1.1},
        )
        ax.axhline(0, color="0.6", linewidth=0.7, linestyle="--")
        ax.set_title(TRAIT_LABELS.get(trait, trait))
        ax.set_xlabel("Number of inner validation islands (top-m by GRM)")
        ax.set_ylabel("Outer-fold Pearson r")
        ax.legend(title=r"$\lambda$", frameon=False, ncol=len(lam_order), fontsize=8)
        sns.despine(ax=ax)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path, png_path = output_dir / f"{file_stem}.pdf", output_dir / f"{file_stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s and %s", pdf_path, png_path)
    return pdf_path, png_path


def main() -> None:
    repo_root = find_repo_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traits", default=",".join(TRAIT_ORDER))
    parser.add_argument("--config-dir", default=str(repo_root / "config" / "e6"))
    parser.add_argument("--results-root", default=str(repo_root / "outputs" / "final_results" / "e6_pc_ridge_importance"))
    parser.add_argument("--result-subdir-suffix", default="_2", help="Trait subdir suffix, e.g. '_2' for body_mass_2.")
    parser.add_argument("--top-m", default="3,5,8,10,15", help="Comma list of inner-island counts; 'all' = every island.")
    parser.add_argument("--lambda", dest="lambdas", default="0,0.5,1,1.5", help="Comma list of lambda for mean - lambda*SE.")
    parser.add_argument("--no-respect-ess", action="store_true", help="Ignore the ESS rejection flag during re-selection.")
    parser.add_argument("--max-pca-pcs", type=int, default=None,
                        help="Cap PCA components to save memory (APPROXIMATE; clamps configs wanting more PCs).")
    parser.add_argument("--grm-threshold", default=None,
                        help="Add a GRM-cutoff rule: keep inner islands with avg GRM > p. "
                             "Comma list of floats, or 'auto' to pick p from --target-inner-count.")
    parser.add_argument("--target-inner-count", type=float, default=None,
                        help="With --grm-threshold auto: pick p so folds average ~this many inner islands (e.g. 6.5).")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    results_root = Path(args.results_root)
    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else results_root / f"e6_importance_reselection_outer{args.result_subdir_suffix}.csv"
    )

    grm_thresholds = None
    target_inner_count = args.target_inner_count
    if args.grm_threshold is not None and args.grm_threshold.strip().lower() != "auto":
        grm_thresholds = [float(x) for x in args.grm_threshold.split(",") if x.strip()]
        target_inner_count = None  # explicit thresholds override auto

    df = run_reselection(
        repo_root=repo_root, traits=traits, config_dir=Path(args.config_dir), results_root=results_root,
        top_m_values=_parse_top_m(args.top_m), lambda_values=[float(x) for x in args.lambdas.split(",") if x.strip()],
        respect_ess_guard=not args.no_respect_ess, result_subdir_suffix=args.result_subdir_suffix,
        max_pca_pcs=args.max_pca_pcs, grm_thresholds=grm_thresholds, target_inner_count=target_inner_count,
    )
    if df.empty:
        logger.warning("No results produced.")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Saved %d rows to %s", len(df), out_csv)
    logger.info(
        "Mean outer r by rule:\n%s",
        df.groupby(["trait", "top_m_label", "lambda"], as_index=False)["outer_r"]
        .agg(mean_outer_r="mean", sd_outer_r="std", n_folds="size").to_string(index=False),
    )
    if args.plot:
        plot_reselection_boxplots(df, repo_root / "figures", "e6_importance_reselection_boxplots")


if __name__ == "__main__":
    main()
