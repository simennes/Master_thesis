"""Plot thesis diagnostics for regular tuned PC-ridge nested LOIO."""

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


def _safe_log10(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return float(np.log10(value)) if np.isfinite(value) and value > 0 else np.nan


def _mode_value(values: pd.Series) -> Any:
    counts = values.dropna().value_counts()
    return counts.index[0] if not counts.empty else np.nan


def _geomean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[arr > 0]
    return float(np.exp(np.mean(np.log(arr)))) if arr.size else np.nan


def load_e1_pc_ridge_diagnostics(
    results_root: Path,
    file_stem_prefix: str = "e1_pc_ridge_nested_loio_10",
    trait_order: list[str] | None = None,
) -> pd.DataFrame:
    trait_order = trait_order or TRAIT_ORDER
    rows: list[dict[str, Any]] = []

    for trait in trait_order:
        trait_dir = results_root / trait
        per_fold_path = trait_dir / f"{file_stem_prefix}_per_fold_results.csv"
        best_params_path = trait_dir / f"{file_stem_prefix}_best_params.json"
        if not per_fold_path.exists():
            continue

        per_fold = pd.read_csv(per_fold_path)
        if "trait" not in per_fold.columns:
            per_fold["trait"] = trait

        best_by_fold: dict[int, dict[str, Any]] = {}
        if best_params_path.exists():
            payload = json.loads(best_params_path.read_text(encoding="utf-8"))
            best_by_fold = {
                int(item.get("fold")): item
                for item in payload.get("best_params_per_fold", [])
                if item.get("fold") is not None
            }

        for row in per_fold.itertuples(index=False):
            fold = int(row.fold)
            best_entry = best_by_fold.get(fold, {})
            inner_islands = best_entry.get("inner_validation_islands") or []
            mean_inner_r = getattr(row, "mean_inner_pearson_r", np.nan)
            pearson_r = getattr(row, "pearson_r", np.nan)
            best_alpha = getattr(row, "best_alpha", np.nan)
            best_n_pcs = getattr(row, "best_n_pcs", np.nan)

            rows.append(
                {
                    "trait": trait,
                    "trait_label": TRAIT_LABELS.get(trait, trait),
                    "fold": fold,
                    "test_island_code": getattr(row, "test_island_code", np.nan),
                    "test_island": getattr(row, "test_island", None),
                    "n_train": getattr(row, "n_train", np.nan),
                    "n_test": getattr(row, "n_test", np.nan),
                    "n_features_available": getattr(row, "n_features_available", np.nan),
                    "best_n_pcs": best_n_pcs,
                    "best_alpha": best_alpha,
                    "log10_alpha": _safe_log10(best_alpha),
                    "mean_inner_r": mean_inner_r,
                    "pearson_r": pearson_r,
                    "outer_minus_inner_r": (
                        np.nan
                        if pd.isna(mean_inner_r) or pd.isna(pearson_r)
                        else float(pearson_r) - float(mean_inner_r)
                    ),
                    "outer_pca_cumvar_at_best_r": getattr(row, "outer_pca_cumvar_at_best_r", np.nan),
                    "fit_time_seconds": getattr(row, "fit_time_seconds", np.nan),
                    "study_time_seconds": getattr(row, "study_time_seconds", np.nan),
                    "n_trials_completed": getattr(row, "n_trials_completed", np.nan),
                    "inner_top_k_requested": getattr(row, "inner_top_k_requested", np.nan),
                    "inner_top_k_used": getattr(row, "inner_top_k_used", np.nan),
                    "n_inner_folds": getattr(row, "n_inner_folds", np.nan),
                    "inner_validation_n_samples": int(sum(float(item.get("n_samples", 0)) for item in inner_islands)),
                    "top_inner_avg_grm": inner_islands[0].get("avg_grm_to_outer_test") if inner_islands else np.nan,
                    "source_file": str(per_fold_path),
                }
            )

    return pd.DataFrame(rows)


def summarize_e1_pc_ridge_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics

    return (
        diagnostics.groupby(["trait", "trait_label"], as_index=False)
        .agg(
            mean_r=("pearson_r", "mean"),
            sd_r=("pearson_r", "std"),
            median_inner_r=("mean_inner_r", "median"),
            median_outer_minus_inner_r=("outer_minus_inner_r", "median"),
            modal_n_pcs=("best_n_pcs", _mode_value),
            median_n_pcs=("best_n_pcs", "median"),
            alpha_geomean=("best_alpha", _geomean),
            median_alpha=("best_alpha", "median"),
            median_pca_cumvar=("outer_pca_cumvar_at_best_r", "median"),
            median_study_time_seconds=("study_time_seconds", "median"),
            median_n_trials=("n_trials_completed", "median"),
            median_inner_top_k_used=("inner_top_k_used", "median"),
            median_inner_validation_n_samples=("inner_validation_n_samples", "median"),
        )
        .sort_values("trait")
        .reset_index(drop=True)
    )


def plot_e1_pc_ridge_diagnostic_dashboard(
    diagnostics: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
    file_stem: str = "e1_pc_ridge_diagnostic_dashboard",
) -> tuple[Path, Path] | None:
    if diagnostics.empty:
        print("No E1 PC-ridge diagnostics found.")
        return None

    trait_order = [trait for trait in TRAIT_ORDER if trait in set(diagnostics["trait"])]
    trait_labels = [TRAIT_LABELS.get(trait, trait) for trait in trait_order]
    palette = {"Body mass": "#4C78A8", "Tarsus length": "#59A14F", "Wing length": "#F28E2B"}

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.5), constrained_layout=True)

    ax = axes[0, 0]
    sns.boxplot(
        data=diagnostics,
        x="trait_label",
        y="best_n_pcs",
        order=trait_labels,
        hue="trait_label",
        hue_order=trait_labels,
        palette=palette,
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
        y="best_n_pcs",
        order=trait_labels,
        hue="trait_label",
        hue_order=trait_labels,
        palette=palette,
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
        y="best_alpha",
        order=trait_labels,
        hue="trait_label",
        hue_order=trait_labels,
        palette=palette,
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
        y="best_alpha",
        order=trait_labels,
        hue="trait_label",
        hue_order=trait_labels,
        palette=palette,
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
    sns.boxplot(
        data=diagnostics,
        x="trait_label",
        y="mean_inner_r",
        order=trait_labels,
        hue="trait_label",
        hue_order=trait_labels,
        palette=palette,
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
        y="mean_inner_r",
        order=trait_labels,
        hue="trait_label",
        hue_order=trait_labels,
        palette=palette,
        jitter=0.14,
        size=4.1,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.4,
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_title("Selected inner validation score")
    ax.set_xlabel("")
    ax.set_ylabel("Mean inner Pearson r")
    ax.tick_params(axis="x", rotation=20)
    style_axes(ax)

    ax = axes[1, 1]
    sns.scatterplot(
        data=diagnostics,
        x="mean_inner_r",
        y="pearson_r",
        hue="trait_label",
        hue_order=trait_labels,
        palette=palette,
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


def plot_e1_pc_ridge_island_diagnostics(
    diagnostics: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
    file_stem: str = "e1_pc_ridge_island_diagnostics",
) -> tuple[Path, Path] | None:
    if diagnostics.empty:
        print("No E1 PC-ridge diagnostics found.")
        return None

    island_order = (
        diagnostics[["test_island_code", "test_island"]]
        .drop_duplicates()
        .sort_values("test_island_code")
    )
    island_labels = island_order["test_island"].astype(str).tolist()
    trait_labels = [TRAIT_LABELS.get(trait, trait) for trait in TRAIT_ORDER if trait in set(diagnostics["trait"])]

    outer = (
        diagnostics.pivot_table(index="trait_label", columns="test_island", values="pearson_r", aggfunc="mean")
        .reindex(index=trait_labels, columns=island_labels)
    )
    inner = (
        diagnostics.pivot_table(index="trait_label", columns="test_island", values="mean_inner_r", aggfunc="mean")
        .reindex(index=trait_labels, columns=island_labels)
    )
    gap = (
        diagnostics.pivot_table(
            index="trait_label",
            columns="test_island",
            values="outer_minus_inner_r",
            aggfunc="mean",
        )
        .reindex(index=trait_labels, columns=island_labels)
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.4, 6.9),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]},
    )
    heatmaps = [
        (outer, "Outer prediction by held-out island", "Pearson r", "vlag", 0.0, -0.10, 0.45),
        (inner, "Selected inner validation score", "Mean inner Pearson r", "vlag", 0.0, 0.05, 0.35),
        (gap, "Outer minus selected inner score", "Outer - inner r", "vlag", 0.0, -0.25, 0.25),
    ]
    for ax, (table, title, cbar_label, cmap, center, vmin, vmax) in zip(axes, heatmaps):
        sns.heatmap(
            table,
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": cbar_label, "shrink": 0.82},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=35)
        ax.tick_params(axis="y", labelrotation=0)

    output_dir = Path(output_dir)
    pdf_path = save_figure(fig, output_dir / f"{file_stem}.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / f"{file_stem}.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def make_e1_pc_ridge_diagnostic_figures(
    repo_root: Path | None = None,
    output_dir: Path | None = None,
    results_root: Path | None = None,
    file_stem_prefix: str = "e1_pc_ridge_nested_loio_10",
) -> dict[str, Any]:
    repo_root = find_repo_root() if repo_root is None else Path(repo_root)
    output_dir = repo_root / "figures" if output_dir is None else Path(output_dir)
    results_root = (
        repo_root / "outputs" / "final_results" / "e1_pc_ridge_nested_loio_10"
        if results_root is None
        else Path(results_root)
    )

    configure_plot_style()
    diagnostics = load_e1_pc_ridge_diagnostics(results_root, file_stem_prefix=file_stem_prefix)
    summary = summarize_e1_pc_ridge_diagnostics(diagnostics)

    diagnostics_csv = save_csv(
        diagnostics,
        results_root / "e1_pc_ridge_fold_diagnostics.csv",
        repo_root,
    )
    summary_csv = save_csv(
        summary,
        results_root / "e1_pc_ridge_diagnostic_summary.csv",
        repo_root,
    )

    return {
        "diagnostics": diagnostics,
        "summary": summary,
        "diagnostics_csv": diagnostics_csv,
        "summary_csv": summary_csv,
        "dashboard": plot_e1_pc_ridge_diagnostic_dashboard(diagnostics, output_dir, repo_root),
        "island_diagnostics": plot_e1_pc_ridge_island_diagnostics(diagnostics, output_dir, repo_root),
    }


if __name__ == "__main__":
    paths = make_e1_pc_ridge_diagnostic_figures()
    for key, value in paths.items():
        if key in {"diagnostics", "summary"}:
            print(key, value.shape)
        else:
            print(key, value)
