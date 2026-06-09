"""Across-population diagnostic figure for the thesis Results section."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


TRAIT_ORDER = ["body_mass", "thr_tarsus", "thr_wing"]
TRAIT_LABELS = {
    "body_mass": "Body mass",
    "thr_tarsus": "Tarsus length",
    "thr_wing": "Wing length",
}

METHOD_FILES = {
    "PC Ridge": "outputs/final_results/e1_pc_ridge_nested_loio_10/e1_pc_ridge_fold_diagnostics.csv",
    "PC Ridge + AvgGRM": "outputs/final_results/e6_pc_ridge_avggrm/e6_avggrm_fold_diagnostics.csv",
    "PC Ridge + importance": "outputs/final_results/e6_pc_ridge_importance/e6_importance_fold_diagnostics.csv",
    "PC MLP, uniform": "outputs/final_results/e8_pc_mlp_uniform/e8_pc_mlp_uniform_fold_diagnostics.csv",
}

METHOD_COLORS = {
    "PC Ridge": "#6EA6D7",
    "PC Ridge + AvgGRM": "#3F7FB4",
    "PC Ridge + importance": "#1F4E79",
    "PC MLP, uniform": "#F4C06A",
}

TRAIT_MARKERS = {
    "body_mass": "o",
    "thr_tarsus": "^",
    "thr_wing": "s",
}


def load_diagnostics(repo_root: Path | str = Path(".")) -> pd.DataFrame:
    """Load fold-level diagnostics for completed across-population runs."""
    repo_root = Path(repo_root)
    frames: list[pd.DataFrame] = []

    for method, rel_path in METHOD_FILES.items():
        path = repo_root / rel_path
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["method"] = method
        if "outer_minus_inner_r" not in df.columns:
            df["outer_minus_inner_r"] = df["pearson_r"] - df["mean_inner_r"]
        if "trait_label" not in df.columns:
            df["trait_label"] = df["trait"].map(TRAIT_LABELS).fillna(df["trait"])
        if "ess_ratio" not in df.columns:
            df["ess_ratio"] = np.nan
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    diagnostics = pd.concat(frames, ignore_index=True, sort=False)
    diagnostics["trait"] = pd.Categorical(diagnostics["trait"], TRAIT_ORDER, ordered=True)
    diagnostics["trait_label"] = diagnostics["trait"].astype(str).map(TRAIT_LABELS)
    return diagnostics


def summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarise inner--outer transfer and weighting strength by method and trait."""
    if diagnostics.empty:
        return pd.DataFrame()

    rows = []
    for (method, trait), sub in diagnostics.groupby(["method", "trait"], observed=True):
        rows.append(
            {
                "method": method,
                "trait": trait,
                "trait_label": TRAIT_LABELS.get(str(trait), str(trait)),
                "n_folds": int(len(sub)),
                "mean_outer_r": float(sub["pearson_r"].mean()),
                "median_inner_r": float(sub["mean_inner_r"].median()),
                "median_outer_minus_inner_r": float(sub["outer_minus_inner_r"].median()),
                "corr_inner_outer": float(sub["mean_inner_r"].corr(sub["pearson_r"])),
                "median_ess_ratio": float(sub["ess_ratio"].median()) if sub["ess_ratio"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=9)


def plot_across_pop_diagnostics(
    diagnostics: pd.DataFrame,
    output_dir: Path | str = Path("figures"),
    file_stem: str = "across_pop_model_selection_diagnostics",
) -> tuple[Path, Path]:
    """Plot inner--outer transfer and Kish effective sample-size diagnostics."""
    if diagnostics.empty:
        raise ValueError("No diagnostics available for across-population diagnostic plot.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)

    scatter_ax, ess_ax = axes
    for method, method_df in diagnostics.groupby("method", sort=False):
        for trait, sub in method_df.groupby("trait", observed=True, sort=False):
            if sub.empty:
                continue
            scatter_ax.scatter(
                sub["mean_inner_r"],
                sub["pearson_r"],
                s=34,
                marker=TRAIT_MARKERS.get(str(trait), "o"),
                color=METHOD_COLORS.get(method, "0.4"),
                edgecolor="white",
                linewidth=0.5,
                alpha=0.86,
            )

    finite = diagnostics[["mean_inner_r", "pearson_r"]].dropna()
    lim_min = float(np.floor((finite.min().min() - 0.025) * 20) / 20)
    lim_max = float(np.ceil((finite.max().max() + 0.025) * 20) / 20)
    scatter_ax.plot([lim_min, lim_max], [lim_min, lim_max], color="0.35", lw=1.0, ls="--")
    scatter_ax.set_xlim(lim_min, lim_max)
    scatter_ax.set_ylim(lim_min, lim_max)
    scatter_ax.set_xlabel("Inner validation Pearson $r$")
    scatter_ax.set_ylabel("Outer LOIO Pearson $r$")
    scatter_ax.set_title("Inner validation transfer")
    _style_axes(scatter_ax)

    method_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markeredgecolor="white", label=method, markersize=7)
        for method, color in METHOD_COLORS.items()
        if method in set(diagnostics["method"])
    ]
    trait_handles = [
        Line2D([0], [0], marker=marker, linestyle="", color="0.35", label=TRAIT_LABELS[trait], markersize=7)
        for trait, marker in TRAIT_MARKERS.items()
        if trait in set(diagnostics["trait"].astype(str))
    ]
    leg1 = scatter_ax.legend(handles=method_handles, title="Model", frameon=False, fontsize=8, title_fontsize=8, loc="upper left")
    scatter_ax.add_artist(leg1)
    scatter_ax.legend(handles=trait_handles, title="Trait", frameon=False, fontsize=8, title_fontsize=8, loc="lower right")

    weighted = diagnostics[diagnostics["method"].isin(["PC Ridge + AvgGRM", "PC Ridge + importance"])].copy()
    weighted = weighted[weighted["ess_ratio"].notna()]
    weighted["model_display"] = weighted["method"].map(
        {
            "PC Ridge + AvgGRM": "AvgGRM\nPC-ridge",
            "PC Ridge + importance": "Importance\nPC-ridge",
        }
    )
    palette = {TRAIT_LABELS[trait]: color for trait, color in zip(TRAIT_ORDER, ["#4C78A8", "#59A14F", "#E15759"])}
    sns.boxplot(
        data=weighted,
        x="model_display",
        y="ess_ratio",
        hue="trait_label",
        hue_order=[TRAIT_LABELS[t] for t in TRAIT_ORDER],
        palette=palette,
        width=0.68,
        linewidth=0.9,
        fliersize=0,
        ax=ess_ax,
    )
    sns.stripplot(
        data=weighted,
        x="model_display",
        y="ess_ratio",
        hue="trait_label",
        hue_order=[TRAIT_LABELS[t] for t in TRAIT_ORDER],
        palette=palette,
        dodge=True,
        size=3.0,
        alpha=0.55,
        linewidth=0.3,
        edgecolor="white",
        ax=ess_ax,
    )
    handles, labels = ess_ax.get_legend_handles_labels()
    ess_ax.legend(handles[:3], labels[:3], title="Trait", frameon=False, fontsize=8, title_fontsize=8, loc="lower left")
    ess_ax.axhline(1.0, color="0.35", lw=1.0, ls="--")
    ess_ax.set_ylim(0.2, 1.05)
    ess_ax.set_xlabel("")
    ess_ax.set_ylabel("Kish ESS / source size")
    ess_ax.set_title("Weight concentration")
    _style_axes(ess_ax)

    pdf_path = output_dir / f"{file_stem}.pdf"
    png_path = output_dir / f"{file_stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def make_across_pop_diagnostic_figure(
    repo_root: Path | str = Path("."),
    output_dir: Path | str = Path("figures"),
) -> dict[str, object]:
    diagnostics = load_diagnostics(repo_root)
    summary = summarize_diagnostics(diagnostics)
    pdf_path, png_path = plot_across_pop_diagnostics(diagnostics, output_dir=output_dir)
    return {"diagnostics": diagnostics, "summary": summary, "figure": (pdf_path, png_path)}


if __name__ == "__main__":
    paths = make_across_pop_diagnostic_figure()
    print(paths["figure"])
