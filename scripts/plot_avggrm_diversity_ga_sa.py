"""Plot the AvgGRM-diversity greedy versus GA+SA comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from thesis_style import (
        FULL_WIDTH,
        PALETTE,
        SEMANTIC_COLORS,
        configure_thesis_style,
        style_axes,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.thesis_style import (
        FULL_WIDTH,
        PALETTE,
        SEMANTIC_COLORS,
        configure_thesis_style,
        style_axes,
    )


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "Data").exists() and (path / "src").exists():
            return path
        candidate = path / "Master_thesis"
        if (candidate / "Data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not find the Master_thesis repo root.")


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


def summarize_avggrm_diversity_ga(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    group_cols = ["lambda_div", "k"] if "k" in df.columns else ["lambda_div"]
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            n=("repeat", "count"),
            q_greedy_mean=("q_greedy", "mean"),
            q_ga_mean=("q_ga", "mean"),
            ga_gain_mean=("q_ga_minus_greedy", "mean"),
            ga_gain_sd=("q_ga_minus_greedy", "std"),
            ga_relative_gain_pct_mean=("q_relative_gain", lambda x: 100.0 * x.mean()),
            ga_jaccard_greedy_mean=("ga_jaccard_greedy", "mean"),
            ga_jaccard_greedy_sd=("ga_jaccard_greedy", "std"),
            greedy_time_ms_mean=("greedy_time_sec", lambda x: 1000.0 * x.mean()),
            ga_time_sec_mean=("ga_time_sec", "mean"),
        )
    )


def plot_avggrm_diversity_ga_comparison(
    results: pd.DataFrame,
    output_dir: Path,
    stem: str = "avggrm_diversity_greedy_vs_ga_sa",
    repo_root: Path | None = None,
) -> tuple[Path, Path]:
    """Save the 1x2 greedy versus GA+SA appendix figure."""
    if results.empty:
        raise ValueError("Cannot plot an empty AvgGRM-diversity result table.")

    required = {
        "lambda_div",
        "q_greedy",
        "q_ga",
        "ga_jaccard_greedy",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    repo_root = find_repo_root() if repo_root is None else Path(repo_root)
    output_dir = Path(output_dir)
    df = results.copy()
    df["lambda_label"] = df["lambda_div"].map(lambda value: f"{float(value):g}")

    configure_thesis_style()
    palette = sns.color_palette(PALETTE, n_colors=df["lambda_label"].nunique())
    lambda_order = [f"{float(value):g}" for value in sorted(df["lambda_div"].unique())]
    lambda_palette = dict(zip(lambda_order, palette))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_WIDTH, 0.56 * FULL_WIDTH),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(h_pad=0.14, hspace=0.16)
    fig.suptitle("AvgGRM-diversity optimization for Sleneset", fontsize=13.5)

    ax = axes[0]
    for lam_label in lambda_order:
        sub = df[df["lambda_label"].eq(lam_label)]
        ax.scatter(
            sub["q_greedy"],
            sub["q_ga"],
            s=18,
            color=lambda_palette[lam_label],
            alpha=0.9,
            label=fr"$\lambda_{{\mathrm{{div}}}}={lam_label}$",
        )

    all_q = pd.concat([df["q_greedy"], df["q_ga"]])
    pad = 0.04 * float(all_q.max() - all_q.min() or 1.0)
    lims = [float(all_q.min() - pad), float(all_q.max() + pad)]
    ax.plot(lims, lims, color=SEMANTIC_COLORS["reference"], linewidth=1.0, linestyle="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Greedy objective $Q$")
    ax.set_ylabel("GA+SA objective $Q$")
    ax.set_title("Objective versus greedy")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)

    ax = axes[1]
    sns.boxplot(
        data=df,
        x="lambda_label",
        y="ga_jaccard_greedy",
        hue="lambda_label",
        order=lambda_order,
        hue_order=lambda_order,
        palette=lambda_palette,
        width=0.58,
        linewidth=1.0,
        fliersize=0,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="lambda_label",
        y="ga_jaccard_greedy",
        order=lambda_order,
        color=SEMANTIC_COLORS["reference"],
        size=3.0,
        jitter=0.14,
        alpha=0.65,
        ax=ax,
    )
    ax.set_xlabel(r"$\lambda_{\mathrm{div}}$")
    ax.set_ylabel("Jaccard overlap with greedy")
    ax.set_ylim(0.65, 1.02)
    ax.set_title("Selected-set overlap")
    style_axes(ax)

    pdf_path = save_figure(fig, output_dir / f"{stem}.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / f"{stem}.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def make_avggrm_diversity_ga_figure(
    repo_root: Path | None = None,
    results_path: Path | None = None,
    output_dir: Path | None = None,
    stem: str = "avggrm_diversity_greedy_vs_ga_sa",
) -> tuple[Path, Path]:
    repo_root = find_repo_root() if repo_root is None else Path(repo_root)
    if results_path is None:
        results_path = (
            repo_root
            / "outputs"
            / "final_results"
            / "avggrm_diversity_ga_sa"
            / f"{stem}_results.csv"
        )
    if output_dir is None:
        output_dir = repo_root / "figures"

    results = pd.read_csv(results_path)
    return plot_avggrm_diversity_ga_comparison(
        results,
        output_dir=Path(output_dir),
        stem=stem,
        repo_root=repo_root,
    )


if __name__ == "__main__":
    root = find_repo_root()
    print(make_avggrm_diversity_ga_figure(repo_root=root))
