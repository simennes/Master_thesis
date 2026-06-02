"""Plot thesis figures for E6 importance-weighted PC ridge diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TRAIT_LABELS = {
    "body_mass": "Body mass",
    "thr_tarsus": "Tarsus length",
    "thr_wing": "Wing length",
}
TRAIT_ORDER = ["body_mass", "thr_tarsus", "thr_wing"]

METHOD_COLORS = {
    "Uniform": "#4C78A8",
    "PC logistic": "#E15759",
}


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "Data").exists() and (path / "src").exists():
            return path
        candidate = path / "Master_thesis"
        if (candidate / "Data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not find the Master_thesis repo root.")


def configure_plot_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="colorblind",
        font="Times New Roman",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.55,
        },
    )
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.22, linewidth=0.55)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def save_figure(fig: plt.Figure, path: Path, repo_root: Path, **kwargs: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, **kwargs)
        return path
    except PermissionError as exc:
        fallback = repo_root / "figures" / path.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fallback, **kwargs)
        print(f"Could not write {path} ({exc}). Wrote {fallback} instead.")
        return fallback


def save_csv(df: pd.DataFrame, path: Path, repo_root: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError as exc:
        fallback = repo_root / "figures" / path.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_csv(fallback, index=False)
            print(f"Could not write {path} ({exc}). Wrote {fallback} instead.")
            return fallback
        except PermissionError as fallback_exc:
            print(f"Could not write {path} ({exc}) or {fallback} ({fallback_exc}). CSV was not updated.")
            return path


def _method_label(method: Any) -> str:
    method = str(method or "uniform").lower()
    if method == "pc_logistic":
        return "PC logistic"
    return method.replace("_", " ").title()


def _safe_log10(value: Any) -> float:
    value = float(value) if value is not None else np.nan
    return float(np.log10(value)) if np.isfinite(value) and value > 0 else np.nan


def _safe_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_e6_importance_diagnostics(
    results_root: Path,
    trait_order: list[str] | None = None,
) -> pd.DataFrame:
    trait_order = trait_order or TRAIT_ORDER
    rows: list[dict[str, Any]] = []

    for trait in trait_order:
        result_path = results_root / trait / f"e6_pc_ridge_importance_{trait}_results.json"
        if not result_path.exists():
            continue

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        best_by_fold = {
            int(item.get("fold")): item
            for item in payload.get("best_params_per_fold", [])
            if item.get("fold") is not None
        }
        total_n = int(sum(int(metric.get("test_size", 0) or 0) for metric in payload.get("per_fold_metrics", [])))

        for metric in payload.get("per_fold_metrics", []):
            fold = int(metric["fold"])
            best_entry = best_by_fold.get(fold, {})
            best_params = best_entry.get("best_params", {})
            weighting = metric.get("weighting", {}) or best_params.get("weighting", {}) or {"name": "uniform"}
            method = str(weighting.get("name", "uniform")).lower()
            test_size = int(metric.get("test_size", 0) or 0)
            n_train = total_n - test_size if total_n > test_size else np.nan
            ess = _safe_float(metric.get("effective_sample_size"))
            pre_ess = _safe_float(metric.get("pre_shrink_effective_sample_size"))
            mean_inner_r = best_entry.get("mean_inner_r")
            pearson_r = metric.get("test_corr")
            inner_islands = metric.get("inner_validation_islands") or best_entry.get("inner_validation_islands") or []

            rows.append(
                {
                    "trait": trait,
                    "trait_label": TRAIT_LABELS.get(trait, trait),
                    "fold": fold,
                    "test_island_code": metric.get("test_island"),
                    "test_island": metric.get("test_island_name"),
                    "test_size": test_size,
                    "n_train": n_train,
                    "pearson_r": pearson_r,
                    "mean_inner_r": mean_inner_r,
                    "outer_minus_inner_r": None
                    if pearson_r is None or mean_inner_r is None
                    else float(pearson_r) - float(mean_inner_r),
                    "alpha": metric.get("alpha", best_params.get("alpha")),
                    "log10_alpha": _safe_log10(metric.get("alpha", best_params.get("alpha"))),
                    "n_pcs": metric.get("n_pcs", best_params.get("n_pcs")),
                    "num_snps": metric.get("num_snps", best_params.get("num_snps")),
                    "weighting_method": method,
                    "weighting_label": _method_label(method),
                    "clip_max": weighting.get("clip_max"),
                    "prob_clip": weighting.get("prob_clip"),
                    "n_components": weighting.get("n_components", metric.get("n_components_used")),
                    "n_components_used": metric.get("n_components_used", weighting.get("n_components")),
                    "logistic_c": weighting.get("logistic_c"),
                    "rho": weighting.get("rho"),
                    "pca_fit": weighting.get("pca_fit"),
                    "effective_sample_size": ess,
                    "pre_shrink_effective_sample_size": pre_ess,
                    "ess_ratio": ess / float(n_train) if np.isfinite(n_train) and n_train > 0 else np.nan,
                    "pre_shrink_ess_ratio": pre_ess / float(n_train) if np.isfinite(n_train) and n_train > 0 else np.nan,
                    "inner_validation_top_k_requested": metric.get("inner_validation_top_k_related_islands_requested"),
                    "inner_validation_top_k_used": metric.get("inner_validation_top_k_related_islands_used"),
                    "inner_validation_n_samples": int(sum(float(item.get("n_samples", 0)) for item in inner_islands)),
                    "top_inner_avg_grm": inner_islands[0].get("avg_grm_to_outer_test") if inner_islands else np.nan,
                    "source_file": str(result_path),
                }
            )

    return pd.DataFrame(rows)


def load_e6_importance_inner_island_trials(
    results_root: Path,
    trait_order: list[str] | None = None,
) -> pd.DataFrame:
    """Long table with one row per (trait, fold, trial, inner validation island).

    Requires runs produced after per-island inner r logging was added
    (``inner_island_metrics`` in each trial of ``trial_history_per_fold``).
    Older result files simply contribute no rows.

    This is the table to drive post-hoc model-selection experiments: for any
    subset of inner islands (e.g. top-m by ``avg_grm_to_outer_test``) recompute
    a per-trial score, apply a rule such as ``mean - lambda * SE`` or an ESS
    floor, take the argmax over trials, and you have the config that *would*
    have been selected under that rule. Note the outer test r is only available
    for the config actually selected at run time (see
    ``load_e6_importance_diagnostics``); scoring a re-selected config on the
    held-out island requires re-running the outer fit for that config.
    """
    trait_order = trait_order or TRAIT_ORDER
    rows: list[dict[str, Any]] = []

    for trait in trait_order:
        result_path = results_root / trait / f"e6_pc_ridge_importance_{trait}_results.json"
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))

        for fold_history in payload.get("trial_history_per_fold", []) or []:
            fold = int(fold_history["fold"])
            test_island = fold_history.get("test_island")
            test_island_name = fold_history.get("test_island_name")
            for trial in fold_history.get("trials", []) or []:
                params = trial.get("params", {}) or {}
                weighting = trial.get("weighting", {}) or {"name": params.get("iw_method", "uniform")}
                method = str(weighting.get("name", params.get("iw_method", "uniform"))).lower()
                for island in trial.get("inner_island_metrics", []) or []:
                    rows.append(
                        {
                            "trait": trait,
                            "trait_label": TRAIT_LABELS.get(trait, trait),
                            "fold": fold,
                            "test_island_code": test_island,
                            "test_island": test_island_name,
                            "trial_number": trial.get("number"),
                            "state": trial.get("state"),
                            "weighting_method": method,
                            "weighting_label": _method_label(method),
                            "alpha": params.get("alpha"),
                            "log10_alpha": _safe_log10(params.get("alpha")),
                            "n_pcs": params.get("n_pcs"),
                            "trial_mean_inner_r": trial.get("value"),
                            "trial_inner_r_se": trial.get("inner_r_se"),
                            "n_inner_islands_scored": trial.get("n_inner_islands_scored"),
                            "effective_sample_size_rejected": bool(
                                trial.get("effective_sample_size_rejected", False)
                            ),
                            "inner_island": island.get("inner_island"),
                            "inner_island_name": island.get("inner_island_name"),
                            "avg_grm_to_outer_test": island.get("avg_grm_to_outer_test"),
                            "inner_r": island.get("r"),
                            "inner_n_train": island.get("n_train"),
                            "inner_n_val": island.get("n_val"),
                            "inner_effective_sample_size": island.get("effective_sample_size"),
                            "inner_pre_shrink_effective_sample_size": island.get(
                                "pre_shrink_effective_sample_size"
                            ),
                            "inner_ess_threshold": island.get("ess_threshold"),
                            "source_file": str(result_path),
                        }
                    )

    return pd.DataFrame(rows)


def load_e6_importance_trials(
    results_root: Path,
    trait_order: list[str] | None = None,
) -> pd.DataFrame:
    trait_order = trait_order or TRAIT_ORDER
    rows: list[dict[str, Any]] = []

    for trait in trait_order:
        result_path = results_root / trait / f"e6_pc_ridge_importance_{trait}_results.json"
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))

        for fold_history in payload.get("trial_history_per_fold", []) or []:
            fold = int(fold_history["fold"])
            test_island = fold_history.get("test_island")
            test_island_name = fold_history.get("test_island_name")
            for trial in fold_history.get("trials", []) or []:
                params = trial.get("params", {}) or {}
                weighting = trial.get("weighting", {}) or {"name": params.get("iw_method", "uniform")}
                method = str(weighting.get("name", params.get("iw_method", "uniform"))).lower()
                ess = trial.get("mean_inner_effective_sample_size")
                threshold = trial.get("mean_inner_effective_sample_size_threshold")
                rows.append(
                    {
                        "trait": trait,
                        "trait_label": TRAIT_LABELS.get(trait, trait),
                        "fold": fold,
                        "test_island_code": test_island,
                        "test_island": test_island_name,
                        "trial_number": trial.get("number"),
                        "state": trial.get("state"),
                        "inner_r": trial.get("value"),
                        "alpha": params.get("alpha"),
                        "log10_alpha": _safe_log10(params.get("alpha")),
                        "n_pcs": params.get("n_pcs"),
                        "weighting_method": method,
                        "weighting_label": _method_label(method),
                        "clip_max": weighting.get("clip_max"),
                        "prob_clip": weighting.get("prob_clip"),
                        "n_components": weighting.get("n_components"),
                        "logistic_c": weighting.get("logistic_c"),
                        "rho": weighting.get("rho"),
                        "pca_fit": weighting.get("pca_fit"),
                        "mean_inner_effective_sample_size": ess,
                        "mean_inner_effective_sample_size_threshold": threshold,
                        "ess_ratio_to_threshold": None
                        if ess is None or threshold in (None, 0)
                        else float(ess) / float(threshold),
                        "effective_sample_size_rejected": bool(trial.get("effective_sample_size_rejected", False)),
                        "source_file": str(result_path),
                    }
                )

    return pd.DataFrame(rows)


def summarize_e6_importance_diagnostics(diagnostics: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics

    def mode_value(values: pd.Series) -> Any:
        counts = values.dropna().value_counts()
        return counts.index[0] if not counts.empty else np.nan

    summary = (
        diagnostics.groupby(["trait", "trait_label"], as_index=False)
        .agg(
            mean_r=("pearson_r", "mean"),
            sd_r=("pearson_r", "std"),
            median_inner_r=("mean_inner_r", "median"),
            median_outer_minus_inner_r=("outer_minus_inner_r", "median"),
            median_n_pcs=("n_pcs", "median"),
            median_alpha=("alpha", "median"),
            modal_weighting=("weighting_label", mode_value),
            median_clip_max=("clip_max", "median"),
            median_prob_clip=("prob_clip", "median"),
            median_n_components=("n_components_used", "median"),
            median_logistic_c=("logistic_c", "median"),
            median_rho=("rho", "median"),
            median_ess=("effective_sample_size", "median"),
            median_ess_ratio=("ess_ratio", "median"),
            median_pre_shrink_ess_ratio=("pre_shrink_ess_ratio", "median"),
        )
        .sort_values("trait")
        .reset_index(drop=True)
    )

    if not trials.empty:
        trial_summary = (
            trials.groupby("trait", as_index=False)
            .agg(
                n_trials=("trial_number", "size"),
                rejected_trials=("effective_sample_size_rejected", "sum"),
            )
        )
        trial_summary["rejected_trial_fraction"] = (
            trial_summary["rejected_trials"] / trial_summary["n_trials"].clip(lower=1)
        )
        summary = summary.merge(trial_summary, on="trait", how="left")

    return summary


def plot_e6_importance_dashboard(
    diagnostics: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
    file_stem: str = "e6_importance_diagnostic_dashboard",
) -> tuple[Path, Path] | None:
    if diagnostics.empty:
        print("No E6 importance diagnostics found.")
        return None

    trait_order = [trait for trait in TRAIT_ORDER if trait in set(diagnostics["trait"])]
    trait_labels = [TRAIT_LABELS.get(trait, trait) for trait in trait_order]
    trait_palette = {"Body mass": "#4C78A8", "Tarsus length": "#59A14F", "Wing length": "#F28E2B"}

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6), constrained_layout=True)

    ax = axes[0, 0]
    sns.boxplot(
        data=diagnostics,
        x="trait_label",
        y="n_pcs",
        order=trait_labels,
        hue="trait_label",
        hue_order=trait_labels,
        palette=trait_palette,
        width=0.55,
        dodge=False,
        fliersize=0,
        legend=False,
        ax=ax,
        boxprops={"alpha": 0.55},
        medianprops={"color": "black", "linewidth": 1.2},
    )
    sns.stripplot(
        data=diagnostics,
        x="trait_label",
        y="n_pcs",
        order=trait_labels,
        hue="trait_label",
        hue_order=trait_labels,
        palette=trait_palette,
        jitter=0.14,
        size=4.1,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.4,
        legend=False,
        ax=ax,
    )
    ax.set_title("Selected number of PCs")
    ax.set_xlabel("")
    ax.set_ylabel("PCs")
    ax.tick_params(axis="x", rotation=20)
    style_axes(ax)

    ax = axes[0, 1]
    sns.boxplot(
        data=diagnostics,
        x="trait_label",
        y="alpha",
        order=trait_labels,
        hue="trait_label",
        hue_order=trait_labels,
        palette=trait_palette,
        width=0.55,
        dodge=False,
        fliersize=0,
        legend=False,
        ax=ax,
        boxprops={"alpha": 0.55},
        medianprops={"color": "black", "linewidth": 1.2},
    )
    sns.stripplot(
        data=diagnostics,
        x="trait_label",
        y="alpha",
        order=trait_labels,
        hue="trait_label",
        hue_order=trait_labels,
        palette=trait_palette,
        jitter=0.14,
        size=4.1,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.4,
        legend=False,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("Selected ridge penalty")
    ax.set_xlabel("")
    ax.set_ylabel(r"$\alpha$")
    ax.tick_params(axis="x", rotation=20)
    style_axes(ax)

    ax = axes[1, 0]
    counts = (
        diagnostics.groupby(["trait_label", "weighting_label"], as_index=False)
        .size()
        .pivot(index="trait_label", columns="weighting_label", values="size")
        .reindex(trait_labels)
        .fillna(0)
    )
    x = np.arange(len(counts))
    bottom = np.zeros(len(counts), dtype=float)
    for label in ["Uniform", "PC logistic"]:
        values = counts[label].to_numpy(dtype=float) if label in counts.columns else np.zeros(len(counts))
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=METHOD_COLORS.get(label, "0.5"),
            label=label,
            width=0.62,
            edgecolor="white",
            linewidth=0.6,
        )
        bottom += values
    ax.set_title("Selected weighting model")
    ax.set_xlabel("")
    ax.set_ylabel("Outer folds")
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=20, ha="right")
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)

    ax = axes[1, 1]
    ess_long = diagnostics.melt(
        id_vars=["trait", "trait_label"],
        value_vars=["pre_shrink_ess_ratio", "ess_ratio"],
        var_name="ess_type",
        value_name="ess_ratio_value",
    ).dropna(subset=["ess_ratio_value"])
    ess_long["ess_type"] = ess_long["ess_type"].map(
        {
            "pre_shrink_ess_ratio": "Raw ratio",
            "ess_ratio": "After shrinkage",
        }
    )
    sns.boxplot(
        data=ess_long,
        x="trait_label",
        y="ess_ratio_value",
        hue="ess_type",
        order=trait_labels,
        hue_order=["Raw ratio", "After shrinkage"],
        palette={"Raw ratio": "#BAB0AC", "After shrinkage": "#E15759"},
        width=0.62,
        fliersize=0,
        ax=ax,
        boxprops={"alpha": 0.62},
        medianprops={"color": "black", "linewidth": 1.1},
    )
    ax.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.set_title("Kish effective sample size")
    ax.set_xlabel("")
    ax.set_ylabel("ESS / training N")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)

    output_dir = Path(output_dir)
    pdf_path = save_figure(fig, output_dir / f"{file_stem}.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / f"{file_stem}.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def plot_e6_importance_island_diagnostics(
    diagnostics: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
    file_stem: str = "e6_importance_island_diagnostics",
) -> tuple[Path, Path] | None:
    if diagnostics.empty:
        print("No E6 importance diagnostics found.")
        return None

    island_order = (
        diagnostics[["test_island_code", "test_island"]]
        .drop_duplicates()
        .sort_values("test_island_code")
    )
    island_labels = island_order["test_island"].astype(str).tolist()
    trait_labels = [TRAIT_LABELS.get(trait, trait) for trait in TRAIT_ORDER if trait in set(diagnostics["trait"])]

    corr = (
        diagnostics.pivot_table(
            index="trait_label",
            columns="test_island",
            values="pearson_r",
            aggfunc="mean",
        )
        .reindex(index=trait_labels, columns=island_labels)
    )
    ess = (
        diagnostics.pivot_table(
            index="trait_label",
            columns="test_island",
            values="ess_ratio",
            aggfunc="mean",
        )
        .reindex(index=trait_labels, columns=island_labels)
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 4.9),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-0.10,
        vmax=0.45,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.86},
        ax=axes[0],
    )
    axes[0].set_title("E6 importance prediction by held-out island")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="x", labelrotation=35)
    axes[0].tick_params(axis="y", labelrotation=0)

    sns.heatmap(
        ess,
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "ESS / training N", "shrink": 0.86},
        ax=axes[1],
    )
    axes[1].set_title("Weight concentration by held-out island")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", labelrotation=35)
    axes[1].tick_params(axis="y", labelrotation=0)

    output_dir = Path(output_dir)
    pdf_path = save_figure(fig, output_dir / f"{file_stem}.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / f"{file_stem}.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def plot_e6_importance_trial_diagnostics(
    diagnostics: pd.DataFrame,
    trials: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
    file_stem: str = "e6_importance_trial_diagnostics",
) -> tuple[Path, Path] | None:
    if diagnostics.empty or trials.empty:
        print("No E6 importance trial diagnostics found.")
        return None

    trait_order = [trait for trait in TRAIT_ORDER if trait in set(diagnostics["trait"])]
    trait_labels = [TRAIT_LABELS.get(trait, trait) for trait in trait_order]
    trait_palette = {"Body mass": "#4C78A8", "Tarsus length": "#59A14F", "Wing length": "#F28E2B"}

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.5), constrained_layout=True)

    ax = axes[0, 0]
    reject = (
        trials.groupby(["trait", "trait_label"], as_index=False)
        .agg(
            n_trials=("trial_number", "size"),
            rejected_trials=("effective_sample_size_rejected", "sum"),
        )
    )
    reject["rejected_fraction"] = reject["rejected_trials"] / reject["n_trials"].clip(lower=1)
    sns.barplot(
        data=reject,
        x="trait_label",
        y="rejected_fraction",
        order=trait_labels,
        hue="trait_label",
        hue_order=trait_labels,
        palette=trait_palette,
        legend=False,
        ax=ax,
    )
    ax.set_ylim(0, max(0.05, float(reject["rejected_fraction"].max()) * 1.25))
    ax.set_title("Rejected candidate trials")
    ax.set_xlabel("")
    ax.set_ylabel("Fraction rejected")
    ax.tick_params(axis="x", rotation=20)
    style_axes(ax)

    ax = axes[0, 1]
    valid_trials = trials[
        trials["inner_r"].notna()
        & np.isfinite(trials["inner_r"].astype(float))
        & (trials["inner_r"].astype(float) > -0.999)
    ].copy()
    sns.boxplot(
        data=valid_trials,
        x="weighting_label",
        y="inner_r",
        hue="weighting_label",
        palette=METHOD_COLORS,
        width=0.55,
        dodge=False,
        fliersize=0,
        legend=False,
        ax=ax,
        boxprops={"alpha": 0.60},
        medianprops={"color": "black", "linewidth": 1.1},
    )
    ax.set_title("Candidate inner-CV score")
    ax.set_xlabel("")
    ax.set_ylabel("Mean inner Pearson r")
    style_axes(ax)

    ax = axes[1, 0]
    ess_trials = trials[trials["ess_ratio_to_threshold"].notna()].copy()
    sns.boxplot(
        data=ess_trials,
        x="weighting_label",
        y="ess_ratio_to_threshold",
        hue="weighting_label",
        palette=METHOD_COLORS,
        width=0.55,
        dodge=False,
        fliersize=0,
        legend=False,
        ax=ax,
        boxprops={"alpha": 0.60},
        medianprops={"color": "black", "linewidth": 1.1},
    )
    ax.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_title("Candidate ESS margin")
    ax.set_xlabel("")
    ax.set_ylabel("ESS / threshold")
    style_axes(ax)

    ax = axes[1, 1]
    sns.scatterplot(
        data=diagnostics,
        x="mean_inner_r",
        y="pearson_r",
        hue="trait_label",
        hue_order=trait_labels,
        palette=trait_palette,
        s=38,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    finite = diagnostics[["mean_inner_r", "pearson_r"]].dropna().to_numpy(dtype=float)
    if finite.size:
        lo = float(np.nanmin(finite)) - 0.02
        hi = float(np.nanmax(finite)) + 0.02
        ax.plot([lo, hi], [lo, hi], color="0.35", linewidth=0.9, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_title("Selected inner vs outer score")
    ax.set_xlabel("Mean inner Pearson r")
    ax.set_ylabel("Outer Pearson r")
    ax.legend(frameon=False, loc="best")
    style_axes(ax)

    output_dir = Path(output_dir)
    pdf_path = save_figure(fig, output_dir / f"{file_stem}.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / f"{file_stem}.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def make_e6_importance_figures(
    repo_root: Path | None = None,
    output_dir: Path | None = None,
    results_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = find_repo_root() if repo_root is None else Path(repo_root)
    output_dir = repo_root / "figures" if output_dir is None else Path(output_dir)
    results_root = (
        repo_root / "outputs" / "final_results" / "e6_pc_ridge_importance"
        if results_root is None
        else Path(results_root)
    )

    configure_plot_style()
    diagnostics = load_e6_importance_diagnostics(results_root)
    trials = load_e6_importance_trials(results_root)
    summary = summarize_e6_importance_diagnostics(diagnostics, trials)

    diagnostics_csv = save_csv(
        diagnostics,
        results_root / "e6_importance_fold_diagnostics.csv",
        repo_root,
    )
    trials_csv = save_csv(
        trials,
        results_root / "e6_importance_trial_diagnostics.csv",
        repo_root,
    )
    summary_csv = save_csv(
        summary,
        results_root / "e6_importance_diagnostic_summary.csv",
        repo_root,
    )

    return {
        "diagnostics": diagnostics,
        "trials": trials,
        "summary": summary,
        "diagnostics_csv": diagnostics_csv,
        "trials_csv": trials_csv,
        "summary_csv": summary_csv,
        "dashboard": plot_e6_importance_dashboard(diagnostics, output_dir, repo_root),
        "island_diagnostics": plot_e6_importance_island_diagnostics(diagnostics, output_dir, repo_root),
        "trial_diagnostics": plot_e6_importance_trial_diagnostics(diagnostics, trials, output_dir, repo_root),
    }


if __name__ == "__main__":
    paths = make_e6_importance_figures()
    for key, value in paths.items():
        if key in {"diagnostics", "trials", "summary"}:
            print(key, value.shape)
        else:
            print(key, value)
