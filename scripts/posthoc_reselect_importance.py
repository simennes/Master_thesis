"""Post-hoc re-selection of E6 importance-weighted PC-ridge configs.

For every outer fold we recorded, in each trial's ``inner_island_metrics``, the
per-inner-island validation r. This lets us re-run *model selection* under any
rule without re-tuning: pick the ``top_m`` most GRM-related inner islands, score
each trial by ``mean - lambda * SE`` over those islands, take the argmax trial,
then **refit that config on the outer training set and evaluate it on the
held-out island**. The outer refit is necessary because the run only stored the
outer r of the config it actually selected.

Outputs a long CSV (one row per trait x fold x rule) and boxplot figures of
outer-fold r across the (top_m, lambda) grid.

Requires E6 runs produced *after* per-island logging was added (the
``inner_island_metrics`` field). Older result files are skipped with a warning.

Usage
-----
    python -m scripts.posthoc_reselect_importance \
        --top-m 2,4,6,8,all --lambda 0,0.5,1,2 --plot

Run it where the SNP .npz files and a CPU with enough RAM for the PCA are
available (i.e. on the cluster), since it recomputes the same PCA the runner
used.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import load_data
from src.importance_weighting import compute_pc_logistic_importance_weights
from src.nested_cv_importance_weighted_ridge import _apply_include_islands_filter
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
# Re-selection rule
# --------------------------------------------------------------------------- #
def score_trial(metrics: list[dict[str, Any]] | None, top_m: Optional[int], lam: float) -> Optional[float]:
    """Score a trial as ``mean - lam * SE`` over the top_m most-related inner islands."""
    if not metrics:
        return None
    ordered = sorted(
        metrics,
        key=lambda d: (d.get("avg_grm_to_outer_test") if d.get("avg_grm_to_outer_test") is not None else -np.inf),
        reverse=True,
    )
    if top_m is not None:
        ordered = ordered[: int(top_m)]
    r = np.array([d["r"] for d in ordered if d.get("r") is not None], dtype=float)
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
) -> Optional[dict[str, Any]]:
    best, best_score = None, -np.inf
    for t in trials:
        if str(t.get("state")) != "COMPLETE":
            continue
        if respect_ess_guard and bool(t.get("effective_sample_size_rejected", False)):
            continue
        s = score_trial(t.get("inner_island_metrics"), top_m, lam)
        if s is None:
            continue
        if s > best_score:
            best_score, best = s, t
    return best


# --------------------------------------------------------------------------- #
# Feature reconstruction + outer evaluation
# --------------------------------------------------------------------------- #
def load_trait_features(config: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the exact (PCA) feature matrix and labels the runner used."""
    base = config["base_train"]
    cv_cfg = config.get("cv", {})
    seed = int(base.get("seed", config.get("seed", 42)))
    set_seed(seed)

    data_paths = dict(base["paths"])
    data_paths.pop("grm_rds", None)  # GRM not needed for refit; avg_grm comes from the JSON

    X, y, ids, grm_df, locality, code_to_label, y_eval = load_data(
        data_paths,
        target_column=base.get("target_column", "y_adjusted"),
        standardize_features=base.get("standardize_features", False),
        return_locality=True,
        min_count=int(base.get("min_count", 20)),
        return_eval=True,
        eval_target_column=base.get("eval_target_column", "y_mean"),
    )
    if y_eval is None:
        y_eval = y.copy()

    X, y, y_eval, ids, locality, grm_df = _apply_include_islands_filter(
        X=X, y=y, y_eval=y_eval, ids=ids, locality=locality,
        code_to_label=code_to_label, grm_df=grm_df, include_islands=cv_cfg.get("include_islands"),
    )

    pca_state = maybe_apply_pca(X, config)
    if pca_active(pca_state):
        X = pca_state["Z"]
    return {"X": X, "y": y, "y_eval": y_eval, "locality": locality, "code_to_label": code_to_label}


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


def run_reselection(
    repo_root: Path,
    traits: list[str],
    config_dir: Path,
    results_root: Path,
    top_m_values: list[Optional[int]],
    lambda_values: list[float],
    respect_ess_guard: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for trait in traits:
        cfg_path = config_dir / f"final_e6_pc_ridge_importance_{trait}_config.json"
        res_path = results_root / trait / f"e6_pc_ridge_importance_{trait}_results.json"
        if not cfg_path.exists() or not res_path.exists():
            logger.warning("Skipping %s (missing config or results): %s / %s", trait, cfg_path, res_path)
            continue

        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        payload = json.loads(res_path.read_text(encoding="utf-8"))
        history = payload.get("trial_history_per_fold", []) or []
        has_island = any(
            t.get("inner_island_metrics") for fh in history for t in (fh.get("trials") or [])
        )
        if not has_island:
            logger.warning(
                "%s: result file has no per-island inner metrics. Re-run E6 after the "
                "per-island logging change to enable post-hoc re-selection.", trait,
            )
            continue

        logger.info("[%s] reconstructing PCA feature matrix ...", trait)
        feats = load_trait_features(config)
        locality = feats["locality"]
        code_to_label = feats["code_to_label"]

        orig_outer = {int(m["fold"]): float(m["test_corr"]) for m in payload.get("per_fold_metrics", [])}

        # Unique (fold, trial) configs needed across the whole rule grid -> evaluate once.
        cache: dict[tuple[int, int], float] = {}

        for fh in history:
            fold = int(fh["fold"])
            isl = fh.get("test_island")
            isl_name = fh.get("test_island_name")
            trials = fh.get("trials") or []
            idx_test = np.where(locality == isl)[0]
            idx_train = np.where(locality != isl)[0]

            for top_m in top_m_values:
                for lam in lambda_values:
                    chosen = reselect_trial(trials, top_m, lam, respect_ess_guard)
                    if chosen is None:
                        continue
                    tnum = int(chosen["number"])
                    key = (fold, tnum)
                    if key not in cache:
                        params = chosen.get("params", {}) or {}
                        cache[key] = evaluate_config(
                            feats, idx_train, idx_test,
                            alpha=params.get("alpha"), n_pcs=params.get("n_pcs"),
                            weighting=chosen.get("weighting"),
                        )
                    rows.append({
                        "trait": trait,
                        "trait_label": TRAIT_LABELS.get(trait, trait),
                        "fold": fold,
                        "test_island_code": isl,
                        "test_island": isl_name,
                        "top_m": -1 if top_m is None else int(top_m),
                        "top_m_label": _top_m_label(top_m),
                        "lambda": float(lam),
                        "selected_trial": tnum,
                        "weighting_method": str((chosen.get("weighting") or {}).get("name", "uniform")).lower(),
                        "n_pcs": (chosen.get("params") or {}).get("n_pcs"),
                        "alpha": (chosen.get("params") or {}).get("alpha"),
                        "outer_r": cache[key],
                    })

            # Sanity check: the "all islands, lambda=0, ESS-guarded" rule should reproduce
            # the config the run actually selected (and thus its recorded outer r).
            if orig_outer and None in top_m_values and 0.0 in [float(l) for l in lambda_values]:
                chk = reselect_trial(trials, None, 0.0, True)
                if chk is not None:
                    refit = cache.get((fold, int(chk["number"])))
                    if refit is not None and fold in orig_outer:
                        diff = abs(refit - orig_outer[fold])
                        if diff > 1e-4:
                            logger.warning(
                                "[%s] fold %d: refit r=%.4f vs recorded r=%.4f (|diff|=%.4f) "
                                "— reconstruction may differ from the run.",
                                trait, fold, refit, orig_outer[fold], diff,
                            )

        logger.info("[%s] done: %d unique outer refits.", trait, len(cache))

    return pd.DataFrame(rows)


def plot_reselection_boxplots(df: pd.DataFrame, output_dir: Path, file_stem: str) -> Optional[tuple[Path, Path]]:
    if df.empty:
        logger.warning("No re-selection rows to plot.")
        return None

    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.titlesize": 11, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    traits = [t for t in TRAIT_ORDER if t in set(df["trait"])]
    m_order = sorted([v for v in df["top_m_label"].unique() if v != "all"], key=int) + (
        ["all"] if "all" in set(df["top_m_label"]) else []
    )
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
            width=0.74, fliersize=0, ax=ax,
            boxprops={"alpha": 0.75}, medianprops={"color": "black", "linewidth": 1.1},
        )
        ax.axhline(0, color="0.6", linewidth=0.7, linestyle="--")
        ax.set_title(TRAIT_LABELS.get(trait, trait))
        ax.set_xlabel("Number of inner validation islands (top-m by GRM)")
        ax.set_ylabel("Outer-fold Pearson r")
        ax.legend(title=r"$\lambda$", frameon=False, ncol=len(lam_order), loc="lower center", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
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
    parser.add_argument("--traits", default=",".join(TRAIT_ORDER), help="Comma-separated trait keys.")
    parser.add_argument("--config-dir", default=str(repo_root / "config" / "e6"))
    parser.add_argument("--results-root", default=str(repo_root / "outputs" / "final_results" / "e6_pc_ridge_importance"))
    parser.add_argument("--top-m", default="2,4,6,8,all", help="Comma list of inner-island counts; 'all' = every recorded island.")
    parser.add_argument("--lambda", dest="lambdas", default="0,0.5,1,2", help="Comma list of lambda for mean - lambda*SE.")
    parser.add_argument("--no-respect-ess", action="store_true", help="Ignore the ESS rejection flag during re-selection.")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    top_m_values = _parse_top_m(args.top_m)
    lambda_values = [float(x) for x in args.lambdas.split(",") if x.strip()]
    results_root = Path(args.results_root)
    out_csv = Path(args.out_csv) if args.out_csv else results_root / "e6_importance_reselection_outer.csv"

    df = run_reselection(
        repo_root=repo_root, traits=traits, config_dir=Path(args.config_dir), results_root=results_root,
        top_m_values=top_m_values, lambda_values=lambda_values, respect_ess_guard=not args.no_respect_ess,
    )
    if df.empty:
        logger.warning("No results produced. Did you re-run E6 with per-island logging?")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Saved %d rows to %s", len(df), out_csv)

    summary = (
        df.groupby(["trait", "top_m_label", "lambda"], as_index=False)["outer_r"]
        .agg(mean_outer_r="mean", sd_outer_r="std", n_folds="size")
    )
    logger.info("Mean outer r by rule:\n%s", summary.to_string(index=False))

    if args.plot:
        plot_reselection_boxplots(df, repo_root / "figures", "e6_importance_reselection_boxplots")


if __name__ == "__main__":
    main()
