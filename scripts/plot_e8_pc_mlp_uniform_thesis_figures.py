"""Plot thesis figures for E8 uniform PC-MLP nested-LOIO diagnostics."""

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


def _architecture_label(hidden_dims: Any) -> str:
    if isinstance(hidden_dims, str):
        try:
            hidden_dims = json.loads(hidden_dims)
        except json.JSONDecodeError:
            return hidden_dims
    if isinstance(hidden_dims, (list, tuple)):
        return "-".join(str(int(x)) for x in hidden_dims)
    return str(hidden_dims)


def _trial_history_summary(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_history in payload.get("trial_history_per_fold", []) or []:
        values = [
            float(trial["value"])
            for trial in fold_history.get("trials", [])
            if trial.get("state") == "COMPLETE" and trial.get("value") is not None
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "fold": int(fold_history["fold"]),
                "n_complete_trials": int(arr.size),
                "best_inner_r_from_history": float(np.max(arr)),
                "median_trial_inner_r": float(np.median(arr)),
                "p90_trial_inner_r": float(np.quantile(arr, 0.90)),
                "trial_inner_r_iqr": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
            }
        )
    return pd.DataFrame(rows)


def load_e8_pc_mlp_uniform_results(
    results_root: Path,
    trait_order: list[str] | None = None,
) -> pd.DataFrame:
    trait_order = trait_order or TRAIT_ORDER
    rows: list[dict[str, Any]] = []

    for trait in trait_order:
        result_path = results_root / trait / f"e8_pc_mlp_uniform_{trait}_results.json"
        if not result_path.exists():
            continue

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        best_by_fold = {
            int(item.get("fold")): item
            for item in payload.get("best_params_per_fold", [])
            if item.get("fold") is not None
        }
        trial_summary = _trial_history_summary(payload)

        for metric in payload.get("per_fold_metrics", []):
            fold = int(metric["fold"])
            best_entry = best_by_fold.get(fold, {})
            best_params = best_entry.get("best_params", {})
            inner_islands = metric.get("inner_validation_islands") or best_entry.get("inner_validation_islands") or []

            hidden_dims = metric.get("hidden_dims", best_params.get("hidden_dims"))
            n_train = float(metric.get("effective_sample_size", np.nan))
            effective_sample_size = float(metric.get("effective_sample_size", np.nan))
            mean_inner_r = best_entry.get("mean_inner_r")
            pearson_r = metric.get("test_corr")

            rows.append(
                {
                    "trait": trait,
                    "trait_label": TRAIT_LABELS.get(trait, trait),
                    "fold": fold,
                    "test_island_code": metric.get("test_island"),
                    "test_island": metric.get("test_island_name"),
                    "pearson_r": pearson_r,
                    "mean_inner_r": mean_inner_r,
                    "outer_minus_inner_r": None
                    if pearson_r is None or mean_inner_r is None
                    else float(pearson_r) - float(mean_inner_r),
                    "model_type": metric.get("model_type", best_params.get("model_type")),
                    "weighting_mode": metric.get("weighting_mode", "uniform"),
                    "n_pcs": metric.get("n_pcs", best_params.get("n_pcs")),
                    "num_snps": metric.get("num_snps"),
                    "hidden_dims": hidden_dims,
                    "architecture": _architecture_label(hidden_dims),
                    "dropout": metric.get("dropout", best_params.get("dropout")),
                    "batch_norm": metric.get("batch_norm", best_params.get("batch_norm")),
                    "lr": metric.get("lr", best_params.get("lr")),
                    "log10_lr": np.log10(metric.get("lr", best_params.get("lr"))),
                    "weight_decay": metric.get("weight_decay", best_params.get("weight_decay")),
                    "log10_weight_decay": np.log10(metric.get("weight_decay", best_params.get("weight_decay"))),
                    "epochs": metric.get("epochs", best_params.get("epochs")),
                    "loss": metric.get("loss", best_params.get("loss")),
                    "optimizer": metric.get("optimizer", best_params.get("optimizer")),
                    "test_size": metric.get("test_size"),
                    "n_train": n_train,
                    "effective_sample_size": effective_sample_size,
                    "ess_ratio": effective_sample_size / n_train if n_train > 0 else np.nan,
                    "inner_validation_top_k_used": metric.get("inner_validation_top_k_related_islands_used"),
                    "inner_validation_n_samples": int(sum(float(item.get("n_samples", 0)) for item in inner_islands)),
                    "top_inner_avg_grm": inner_islands[0].get("avg_grm_to_outer_test") if inner_islands else np.nan,
                    "source_file": str(result_path),
                }
            )

        if rows and not trial_summary.empty:
            pass

        if not trial_summary.empty:
            for row in rows:
                if row["trait"] != trait:
                    continue
                match = trial_summary[trial_summary["fold"].eq(row["fold"])]
                if match.empty:
                    continue
                for key, value in match.iloc[0].to_dict().items():
                    if key != "fold":
                        row[key] = value

    return pd.DataFrame(rows)


def summarize_e8_pc_mlp_uniform(diagnostics: pd.DataFrame) -> pd.DataFrame:
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
            modal_architecture=("architecture", mode_value),
            median_dropout=("dropout", "median"),
            median_lr=("lr", "median"),
            median_weight_decay=("weight_decay", "median"),
            median_epochs=("epochs", "median"),
            modal_loss=("loss", mode_value),
            modal_optimizer=("optimizer", mode_value),
            median_ess=("effective_sample_size", "median"),
            median_ess_ratio=("ess_ratio", "median"),
        )
        .sort_values("trait")
        .reset_index(drop=True)
    )
    return summary


def _trait_palette(labels: list[str]) -> dict[str, str]:
    base = {"Body mass": "#4C78A8", "Tarsus length": "#59A14F", "Wing length": "#F28E2B"}
    return {label: base.get(label, "#888888") for label in labels}


def _box_strip(
    ax: plt.Axes,
    diagnostics: pd.DataFrame,
    trait_labels: list[str],
    y: str,
    title: str,
    ylabel: str,
    palette: dict[str, str],
    yscale: str | None = None,
) -> None:
    sns.boxplot(
        data=diagnostics,
        x="trait_label",
        y=y,
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
        y=y,
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
    if yscale:
        ax.set_yscale(yscale)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    style_axes(ax)


def plot_e8_pc_mlp_uniform_diagnostic_dashboard(
    diagnostics: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
    file_stem: str = "e8_pc_mlp_uniform_diagnostic_dashboard",
) -> tuple[Path, Path] | None:
    if diagnostics.empty:
        print("No E8 PC-MLP uniform diagnostics found.")
        return None

    trait_order = [trait for trait in TRAIT_ORDER if trait in set(diagnostics["trait"])]
    trait_labels = [TRAIT_LABELS.get(trait, trait) for trait in trait_order]
    palette = _trait_palette(trait_labels)

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.6), constrained_layout=True)
    _box_strip(axes[0, 0], diagnostics, trait_labels, "n_pcs", "Selected PCs", "PCs", palette)
    _box_strip(axes[0, 1], diagnostics, trait_labels, "lr", "Learning rate", "LR", palette, yscale="log")
    _box_strip(
        axes[0, 2],
        diagnostics,
        trait_labels,
        "weight_decay",
        "Weight decay",
        "Weight decay",
        palette,
        yscale="log",
    )
    _box_strip(axes[1, 0], diagnostics, trait_labels, "dropout", "Dropout", "Dropout", palette)
    axes[1, 0].set_ylim(-0.02, 0.62)
    _box_strip(axes[1, 1], diagnostics, trait_labels, "epochs", "Training epochs", "Epochs", palette)

    ax = axes[1, 2]
    architectures = (
        diagnostics.groupby(["trait_label", "architecture"], as_index=False)
        .size()
        .sort_values(["trait_label", "architecture"])
    )
    architecture_order = (
        architectures.groupby("architecture")["size"].sum().sort_values(ascending=False).index.tolist()
    )
    pivot = (
        architectures.pivot(index="trait_label", columns="architecture", values="size")
        .reindex(index=trait_labels, columns=architecture_order)
        .fillna(0)
    )
    colors = sns.color_palette("Set2", n_colors=max(1, len(architecture_order)))
    bottom = np.zeros(len(pivot), dtype=float)
    x = np.arange(len(pivot))
    for color, architecture in zip(colors, architecture_order):
        values = pivot[architecture].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            width=0.62,
            label=architecture,
        )
        bottom += values
    ax.set_title("Selected architecture")
    ax.set_xlabel("")
    ax.set_ylabel("Outer folds")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.legend(frameon=False, loc="upper left", ncol=2, title="Hidden layers")
    style_axes(ax)

    output_dir = Path(output_dir)
    pdf_path = save_figure(fig, output_dir / f"{file_stem}.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / f"{file_stem}.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def plot_e8_pc_mlp_uniform_search_diagnostics(
    diagnostics: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
    file_stem: str = "e8_pc_mlp_uniform_search_diagnostics",
) -> tuple[Path, Path] | None:
    if diagnostics.empty:
        print("No E8 PC-MLP uniform diagnostics found.")
        return None

    diagnostics = diagnostics.copy()
    trait_labels = [TRAIT_LABELS.get(trait, trait) for trait in TRAIT_ORDER if trait in set(diagnostics["trait"])]
    palette = _trait_palette(trait_labels)

    island_order = (
        diagnostics[["test_island_code", "test_island"]]
        .drop_duplicates()
        .sort_values("test_island_code")
    )
    island_labels = island_order["test_island"].astype(str).tolist()
    outer = (
        diagnostics.pivot_table(index="trait_label", columns="test_island", values="pearson_r", aggfunc="mean")
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

    fig = plt.figure(figsize=(10.0, 7.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.18])
    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_outer = fig.add_subplot(gs[0, 1])
    ax_gap = fig.add_subplot(gs[1, 1])

    for trait_label in trait_labels:
        sub = diagnostics[diagnostics["trait_label"].eq(trait_label)]
        ax_scatter.scatter(
            sub["mean_inner_r"],
            sub["pearson_r"],
            s=36,
            color=palette.get(trait_label, "#888888"),
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            label=trait_label,
        )
    lim_min = float(np.nanmin([diagnostics["mean_inner_r"].min(), diagnostics["pearson_r"].min(), -0.05]))
    lim_max = float(np.nanmax([diagnostics["mean_inner_r"].max(), diagnostics["pearson_r"].max(), 0.45]))
    ax_scatter.plot([lim_min, lim_max], [lim_min, lim_max], color="0.35", linestyle="--", linewidth=0.9)
    ax_scatter.axhline(0, color="0.55", linestyle=":", linewidth=0.8)
    ax_scatter.set_xlim(lim_min - 0.02, lim_max + 0.02)
    ax_scatter.set_ylim(lim_min - 0.02, lim_max + 0.02)
    ax_scatter.set_title("Inner validation versus outer test")
    ax_scatter.set_xlabel("Best mean inner Pearson $r$")
    ax_scatter.set_ylabel("Outer Pearson $r$")
    ax_scatter.legend(frameon=False, loc="upper left")
    style_axes(ax_scatter)

    sns.heatmap(
        outer,
        cmap="vlag",
        center=0,
        vmin=-0.10,
        vmax=0.45,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Outer Pearson r", "shrink": 0.82},
        ax=ax_outer,
    )
    ax_outer.set_title("Outer prediction by held-out island")
    ax_outer.set_xlabel("")
    ax_outer.set_ylabel("")
    ax_outer.tick_params(axis="x", labelrotation=35)
    ax_outer.tick_params(axis="y", labelrotation=0)

    sns.heatmap(
        gap,
        cmap="vlag",
        center=0,
        vmin=-0.28,
        vmax=0.28,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Outer - inner r", "shrink": 0.82},
        ax=ax_gap,
    )
    ax_gap.set_title("Validation optimism by held-out island")
    ax_gap.set_xlabel("")
    ax_gap.set_ylabel("")
    ax_gap.tick_params(axis="x", labelrotation=35)
    ax_gap.tick_params(axis="y", labelrotation=0)

    output_dir = Path(output_dir)
    pdf_path = save_figure(fig, output_dir / f"{file_stem}.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / f"{file_stem}.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def make_e8_pc_mlp_uniform_figures(
    repo_root: Path | None = None,
    output_dir: Path | None = None,
    results_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = find_repo_root() if repo_root is None else Path(repo_root)
    output_dir = repo_root / "figures" if output_dir is None else Path(output_dir)
    results_root = (
        repo_root / "outputs" / "final_results" / "e8_pc_mlp_uniform"
        if results_root is None
        else Path(results_root)
    )

    configure_plot_style()
    diagnostics = load_e8_pc_mlp_uniform_results(results_root)
    summary = summarize_e8_pc_mlp_uniform(diagnostics)

    diagnostics_csv = save_csv(
        diagnostics,
        repo_root / "outputs" / "final_results" / "e8_pc_mlp_uniform" / "e8_pc_mlp_uniform_fold_diagnostics.csv",
        repo_root,
    )
    summary_csv = save_csv(
        summary,
        repo_root / "outputs" / "final_results" / "e8_pc_mlp_uniform" / "e8_pc_mlp_uniform_diagnostic_summary.csv",
        repo_root,
    )

    paths = {
        "diagnostics": diagnostics,
        "summary": summary,
        "diagnostics_csv": diagnostics_csv,
        "summary_csv": summary_csv,
        "dashboard": plot_e8_pc_mlp_uniform_diagnostic_dashboard(diagnostics, output_dir, repo_root),
        "search_diagnostics": plot_e8_pc_mlp_uniform_search_diagnostics(diagnostics, output_dir, repo_root),
    }
    return paths

