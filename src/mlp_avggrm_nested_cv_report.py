from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")

BOXPLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titleweight": "normal",
    "axes.titlepad": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}

BASE_MODEL_COLORS = {
    "MLP avgGRM weighted": "#4C78A8",
    "Ridge": "#8C564B",
    "BPCRR | unweighted": "#BAB0AC",
}

SCHEME_COLORS = {
    "uniform": "#4C78A8",
    "linear": "#72B7B2",
    "minmax": "#54A24B",
    "exponential": "#F58518",
    "top-heavy": "#E45756",
}


def _project_root(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).resolve()
    for candidate in [root, *root.parents]:
        if (candidate / "outputs").exists() and (candidate / "src").exists():
            return candidate
    return root


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prettify_bpcrr_method(method: str) -> str:
    if method == "full_source_unweighted":
        return "unweighted"
    if method.startswith("bpcrr_weight_"):
        return method.removeprefix("bpcrr_weight_")
    return method


def _build_island_order(*dfs: pd.DataFrame) -> dict[str, int]:
    order: dict[str, int] = {}
    for df in dfs:
        if df.empty or "target_island_name" not in df.columns or "target_island" not in df.columns:
            continue
        island_rows = (
            df[["target_island_name", "target_island"]]
            .dropna()
            .drop_duplicates()
            .sort_values("target_island")
        )
        for name, idx in island_rows.itertuples(index=False):
            order.setdefault(str(name), int(idx))
    return order


def _sort_islands(names, order_map: dict[str, int]) -> list[str]:
    return sorted(names, key=lambda name: (order_map.get(str(name), 10**9), str(name)))


def _load_mlp_results(path: Path) -> tuple[dict, pd.DataFrame]:
    payload = _load_json(path)
    best_by_fold = {
        int(item.get("fold")): item
        for item in payload.get("best_params_per_fold", [])
        if item.get("fold") is not None
    }

    rows = []
    for fold_row in payload.get("per_fold_metrics", []):
        fold = int(fold_row["fold"])
        best_row = best_by_fold.get(fold, {})
        params = dict(best_row.get("best_params", {}))
        weighting = dict(params.get("weighting", fold_row.get("weighting", {})) or {})
        hidden_dims = params.get("hidden_dims", [])

        rows.append(
            {
                "model_label": "MLP avgGRM weighted",
                "fold": fold,
                "target_island": int(fold_row["test_island"]),
                "target_island_name": str(fold_row["test_island_name"]),
                "corr": float(fold_row["test_corr"]),
                "test_size": int(fold_row["test_size"]),
                "mean_inner_r": float(best_row["mean_inner_r"]) if best_row.get("mean_inner_r") is not None else np.nan,
                "weight_scheme": str(weighting.get("name", params.get("weight_scheme", "unknown"))),
                "weight_floor": float(weighting["floor"]) if weighting.get("floor") is not None else np.nan,
                "weight_clip_max": float(weighting["clip_max"]) if weighting.get("clip_max") is not None else np.nan,
                "weight_beta": float(weighting["beta"]) if weighting.get("beta") is not None else np.nan,
                "weight_eps": float(weighting["eps"]) if weighting.get("eps") is not None else np.nan,
                "weight_top_frac": float(weighting["top_frac"]) if weighting.get("top_frac") is not None else np.nan,
                "weight_low": float(weighting["low"]) if weighting.get("low") is not None else np.nan,
                "weight_high": float(weighting["high"]) if weighting.get("high") is not None else np.nan,
                "weight_linear_min": float(weighting["min_weight"]) if weighting.get("min_weight") is not None else np.nan,
                "weight_linear_max": float(weighting["max_weight"]) if weighting.get("max_weight") is not None else np.nan,
                "optimizer": params.get("optimizer"),
                "loss": params.get("loss"),
                "epochs": int(params["epochs"]) if params.get("epochs") is not None else np.nan,
                "lr": float(params["lr"]) if params.get("lr") is not None else np.nan,
                "weight_decay": float(params["weight_decay"]) if params.get("weight_decay") is not None else np.nan,
                "dropout": float(params["dropout"]) if params.get("dropout") is not None else np.nan,
                "use_snp_selection": bool(params.get("use_snp_selection", False)),
                "num_snps": int(params["num_snps"]) if params.get("num_snps") is not None else np.nan,
                "hidden_dims": "x".join(str(int(x)) for x in hidden_dims) if hidden_dims else "NA",
                "batch_norm": bool(params.get("batch_norm", False)),
            }
        )

    df = pd.DataFrame(rows).sort_values(["target_island", "fold"]).reset_index(drop=True)
    df["use_snp_selection_label"] = np.where(df["use_snp_selection"], "True", "False")
    df["batch_norm_label"] = np.where(df["batch_norm"], "True", "False")
    df["weight_clip_max_label"] = df["weight_clip_max"].map(lambda x: "none" if pd.isna(x) else f"{x:g}")
    return payload, df


def _load_ridge_results(path: Path) -> tuple[dict, pd.DataFrame]:
    payload = _load_json(path)
    rows = []
    for fold_row in payload.get("per_fold", []):
        rows.append(
            {
                "model_label": "Ridge",
                "fold": int(fold_row["fold"]),
                "target_island": int(fold_row["test_island_code"]),
                "target_island_name": str(fold_row["test_island"]),
                "corr": float(fold_row["pearson_r"]),
                "test_size": int(fold_row["n_test"]),
            }
        )
    df = pd.DataFrame(rows).sort_values(["target_island", "fold"]).reset_index(drop=True)
    return payload, df


def _load_bpcrr_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[df["trait"] == "body_mass"].copy()
    df["target_island"] = pd.to_numeric(df["target_island"], errors="coerce")
    df["corr"] = pd.to_numeric(df["corr_eval"], errors="coerce")
    df["n_individuals"] = pd.to_numeric(df["n_individuals"], errors="coerce")
    df = (
        df.groupby(["method", "weight_scheme", "target_island", "target_island_name"], as_index=False)["corr"]
        .mean()
        .sort_values(["method", "target_island"])
        .reset_index(drop=True)
    )
    return df


def bpcrr_method_summary(df: pd.DataFrame, islands: list[str] | None = None) -> pd.DataFrame:
    subset = df.copy()
    if islands is not None:
        subset = subset.loc[subset["target_island_name"].isin(islands)].copy()

    summary = (
        subset.groupby(["method", "weight_scheme"], as_index=False)
        .agg(
            n_islands=("corr", "size"),
            mean_corr=("corr", "mean"),
            median_corr=("corr", "median"),
            std_corr=("corr", "std"),
        )
        .sort_values(["mean_corr", "median_corr"], ascending=False)
        .reset_index(drop=True)
    )
    summary["display"] = summary["method"].map(prettify_bpcrr_method)
    return summary


def pick_best_bpcrr_method(
    df: pd.DataFrame,
    islands: list[str] | None = None,
    weighted_only: bool = False,
) -> pd.Series:
    summary = bpcrr_method_summary(df, islands)
    if weighted_only:
        summary = summary.loc[summary["method"] != "full_source_unweighted"].copy()
    if summary.empty:
        raise ValueError("No BPCRR methods available for the requested subset.")
    return summary.iloc[0]


def _model_subset(df: pd.DataFrame, label: str, islands: list[str] | None = None) -> pd.DataFrame:
    subset = df.copy()
    if islands is not None:
        subset = subset.loc[subset["target_island_name"].isin(islands)].copy()
    subset = subset[["target_island_name", "target_island", "corr"]].copy()
    subset["model_label"] = label
    return subset


def _bpcrr_subset(df: pd.DataFrame, method: str, label: str, islands: list[str] | None = None) -> pd.DataFrame:
    subset = df.loc[df["method"] == method].copy()
    if islands is not None:
        subset = subset.loc[subset["target_island_name"].isin(islands)].copy()
    subset = subset[["target_island_name", "target_island", "corr"]].copy()
    subset["model_label"] = label
    return subset


def _add_mean_markers(ax, data: pd.DataFrame, order: list[str]) -> None:
    mean_map = data.groupby("model_label")["corr"].mean()
    xpos = [mean_map[label] for label in order]
    ypos = list(range(len(order)))
    ax.scatter(
        xpos,
        ypos,
        marker="D",
        s=70,
        color="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=6,
        label="Mean",
    )


def _annotate_sample_sizes(ax, data: pd.DataFrame, order: list[str]) -> None:
    xmin, xmax = ax.get_xlim()
    pad = 0.03 * (xmax - xmin)
    for y, label in enumerate(order):
        count = int((data["model_label"] == label).sum())
        ax.text(xmax + pad, y, f"n={count}", va="center", ha="left", fontsize=11)
    ax.set_xlim(xmin, xmax + 0.18 * (xmax - xmin))


def _model_colors(report: dict) -> dict[str, str]:
    colors = dict(BASE_MODEL_COLORS)
    colors[report["best_bpcrr_common_label"]] = "#E45756"
    colors[report["best_bpcrr_shared_label"]] = "#E45756"
    return colors


def prepare_report_data(root: Path | None = None) -> dict:
    project_root = _project_root(root)
    mlp_path = project_root / "outputs" / "nested_cv" / "mlp_avggrm_weighted_results.json"
    ridge_path = project_root / "outputs" / "nested_cv" / "body_mass" / "loio" / "ridge_results.json"
    bpcrr_path = project_root / "outputs" / "bpcrr_inla_rank_weight" / "bpcrr_inla_rank_weight_results.csv"

    mlp_payload, mlp_df = _load_mlp_results(mlp_path)
    ridge_payload, ridge_df = _load_ridge_results(ridge_path)
    bpcrr_df = _load_bpcrr_results(bpcrr_path)

    island_order = _build_island_order(mlp_df, ridge_df, bpcrr_df)
    common_all_models = _sort_islands(
        set(mlp_df["target_island_name"]) & set(ridge_df["target_island_name"]) & set(bpcrr_df["target_island_name"]),
        island_order,
    )
    shared_mlp_bpcrr = _sort_islands(
        set(mlp_df["target_island_name"]) & set(bpcrr_df["target_island_name"]),
        island_order,
    )
    shared_mlp_bpcrr_baselines = _sort_islands(
        set(mlp_df["target_island_name"])
        & set(bpcrr_df.loc[bpcrr_df["method"] == "full_source_unweighted", "target_island_name"]),
        island_order,
    )

    bpcrr_common_summary = bpcrr_method_summary(bpcrr_df, common_all_models)
    bpcrr_shared_summary = bpcrr_method_summary(bpcrr_df, shared_mlp_bpcrr_baselines)
    best_bpcrr_common = pick_best_bpcrr_method(bpcrr_df, common_all_models, weighted_only=True)
    best_bpcrr_shared = pick_best_bpcrr_method(bpcrr_df, shared_mlp_bpcrr_baselines, weighted_only=True)

    best_bpcrr_common_label = (
        f"BPCRR | best fixed weighted ({prettify_bpcrr_method(best_bpcrr_common['method'])})"
    )
    best_bpcrr_shared_label = (
        f"BPCRR | best fixed weighted ({prettify_bpcrr_method(best_bpcrr_shared['method'])})"
    )

    return {
        "project_root": project_root,
        "mlp_payload": mlp_payload,
        "mlp_df": mlp_df,
        "ridge_payload": ridge_payload,
        "ridge_df": ridge_df,
        "bpcrr_df": bpcrr_df,
        "island_order": island_order,
        "common_all_models": common_all_models,
        "shared_mlp_bpcrr": shared_mlp_bpcrr,
        "shared_mlp_bpcrr_baselines": shared_mlp_bpcrr_baselines,
        "bpcrr_common_summary": bpcrr_common_summary,
        "bpcrr_shared_summary": bpcrr_shared_summary,
        "best_bpcrr_common": best_bpcrr_common,
        "best_bpcrr_shared": best_bpcrr_shared,
        "best_bpcrr_common_label": best_bpcrr_common_label,
        "best_bpcrr_shared_label": best_bpcrr_shared_label,
    }


def _mean_on_islands(df: pd.DataFrame, islands: list[str]) -> float:
    subset = df.loc[df["target_island_name"].isin(islands)]
    return float(subset["corr"].mean()) if not subset.empty else np.nan


def build_model_summary(report: dict) -> pd.DataFrame:
    mlp_df = report["mlp_df"]
    ridge_df = report["ridge_df"]
    bpcrr_df = report["bpcrr_df"]
    common_all_models = report["common_all_models"]
    shared_mlp_bpcrr_baselines = report["shared_mlp_bpcrr_baselines"]
    best_bpcrr_common = report["best_bpcrr_common"]

    return pd.DataFrame(
        [
            {
                "Model": "MLP avgGRM weighted",
                "Available islands": mlp_df["target_island_name"].nunique(),
                "Mean r (available)": mlp_df["corr"].mean(),
                "Mean r (common 11)": _mean_on_islands(mlp_df, common_all_models),
                "Mean r (shared 15 with BPCRR unweighted)": _mean_on_islands(mlp_df, shared_mlp_bpcrr_baselines),
            },
            {
                "Model": "Ridge",
                "Available islands": ridge_df["target_island_name"].nunique(),
                "Mean r (available)": ridge_df["corr"].mean(),
                "Mean r (common 11)": _mean_on_islands(ridge_df, common_all_models),
                "Mean r (shared 15 with BPCRR unweighted)": np.nan,
            },
            {
                "Model": "BPCRR | unweighted",
                "Available islands": bpcrr_df.loc[bpcrr_df["method"] == "full_source_unweighted", "target_island_name"].nunique(),
                "Mean r (available)": bpcrr_df.loc[bpcrr_df["method"] == "full_source_unweighted", "corr"].mean(),
                "Mean r (common 11)": _mean_on_islands(
                    bpcrr_df.loc[bpcrr_df["method"] == "full_source_unweighted"],
                    common_all_models,
                ),
                "Mean r (shared 15 with BPCRR unweighted)": _mean_on_islands(
                    bpcrr_df.loc[bpcrr_df["method"] == "full_source_unweighted"],
                    shared_mlp_bpcrr_baselines,
                ),
            },
            {
                "Model": report["best_bpcrr_common_label"],
                "Available islands": int(best_bpcrr_common["n_islands"]),
                "Mean r (available)": float(best_bpcrr_common["mean_corr"]),
                "Mean r (common 11)": float(best_bpcrr_common["mean_corr"]),
                "Mean r (shared 15 with BPCRR unweighted)": _mean_on_islands(
                    bpcrr_df.loc[bpcrr_df["method"] == best_bpcrr_common["method"]],
                    shared_mlp_bpcrr_baselines,
                ),
            },
        ]
    )


def build_comparison_common(report: dict) -> pd.DataFrame:
    return pd.concat(
        [
            _model_subset(report["mlp_df"], "MLP avgGRM weighted", report["common_all_models"]),
            _model_subset(report["ridge_df"], "Ridge", report["common_all_models"]),
            _bpcrr_subset(report["bpcrr_df"], "full_source_unweighted", "BPCRR | unweighted", report["common_all_models"]),
            _bpcrr_subset(
                report["bpcrr_df"],
                report["best_bpcrr_common"]["method"],
                report["best_bpcrr_common_label"],
                report["common_all_models"],
            ),
        ],
        ignore_index=True,
    )


def build_comparison_mlp_bpcrr(report: dict) -> pd.DataFrame:
    return pd.concat(
        [
            _model_subset(report["mlp_df"], "MLP avgGRM weighted", report["shared_mlp_bpcrr_baselines"]),
            _bpcrr_subset(
                report["bpcrr_df"],
                "full_source_unweighted",
                "BPCRR | unweighted",
                report["shared_mlp_bpcrr_baselines"],
            ),
            _bpcrr_subset(
                report["bpcrr_df"],
                report["best_bpcrr_shared"]["method"],
                report["best_bpcrr_shared_label"],
                report["shared_mlp_bpcrr_baselines"],
            ),
        ],
        ignore_index=True,
    )


def build_heatmap_df(report: dict) -> pd.DataFrame:
    return (
        build_comparison_common(report)
        .pivot(index="target_island_name", columns="model_label", values="corr")
        .reindex(report["common_all_models"])
    )


def build_winner_counts(report: dict) -> pd.DataFrame:
    heatmap_df = build_heatmap_df(report)
    order = [
        "MLP avgGRM weighted",
        "Ridge",
        "BPCRR | unweighted",
        report["best_bpcrr_common_label"],
    ]
    return (
        heatmap_df.idxmax(axis=1)
        .value_counts()
        .reindex(order, fill_value=0)
        .rename_axis("model_label")
        .reset_index(name="wins")
    )


def build_scheme_summary(report: dict) -> pd.DataFrame:
    mlp_df = report["mlp_df"]
    return (
        mlp_df.groupby("weight_scheme", as_index=False)
        .agg(count=("corr", "size"), mean_outer_r=("corr", "mean"), median_outer_r=("corr", "median"))
        .sort_values(["count", "mean_outer_r"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_preference_counts(report: dict, column: str) -> pd.DataFrame:
    mlp_df = report["mlp_df"]
    return (
        mlp_df[column]
        .astype(str)
        .value_counts()
        .rename_axis("choice")
        .reset_index(name="count")
    )


def _weighting_detail(row: pd.Series) -> str:
    if row["weight_scheme"] == "exponential":
        return f"beta={row['weight_beta']:.2f}"
    if row["weight_scheme"] == "minmax":
        return f"eps={row['weight_eps']:.3f}, clip={row['weight_clip_max_label']}"
    if row["weight_scheme"] == "top-heavy":
        return (
            f"top_frac={row['weight_top_frac']:.2f}, "
            f"low={row['weight_low']:.2f}, high={row['weight_high']:.2f}"
        )
    if row["weight_scheme"] == "linear":
        return f"min={row['weight_linear_min']:.2f}, max={row['weight_linear_max']:.2f}"
    return "uniform"


def build_winner_table(report: dict) -> pd.DataFrame:
    mlp_df = report["mlp_df"]
    winner_table = mlp_df[
        [
            "fold",
            "target_island_name",
            "corr",
            "mean_inner_r",
            "weight_scheme",
            "optimizer",
            "loss",
            "use_snp_selection_label",
            "hidden_dims",
            "epochs",
            "num_snps",
        ]
    ].copy()
    winner_table["weighting_detail"] = mlp_df.apply(_weighting_detail, axis=1)
    return winner_table[
        [
            "fold",
            "target_island_name",
            "corr",
            "mean_inner_r",
            "weight_scheme",
            "weighting_detail",
            "optimizer",
            "loss",
            "use_snp_selection_label",
            "hidden_dims",
            "epochs",
            "num_snps",
        ]
    ]


def plot_common_model_comparison(report: dict):
    comparison_common = build_comparison_common(report)
    order = [
        "MLP avgGRM weighted",
        "Ridge",
        "BPCRR | unweighted",
        report["best_bpcrr_common_label"],
    ]
    palette = {label: _model_colors(report).get(label, "#B279A2") for label in order}

    with plt.rc_context(BOXPLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12.5, 5.8), constrained_layout=True)
        sns.boxplot(
            data=comparison_common,
            x="corr",
            y="model_label",
            order=order,
            hue="model_label",
            hue_order=order,
            palette=palette,
            dodge=False,
            linewidth=1.1,
            fliersize=0,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        sns.stripplot(
            data=comparison_common,
            x="corr",
            y="model_label",
            order=order,
            color="black",
            alpha=0.72,
            size=5.5,
            ax=ax,
        )
        _add_mean_markers(ax, comparison_common, order)
        _annotate_sample_sizes(ax, comparison_common, order)
        ax.set_title("Model comparison on common islands (MLP, Ridge, BPCRR)", fontsize=17)
        ax.set_xlabel("Outer-fold Pearson r")
        ax.set_ylabel("")
        ax.legend(loc="lower right", frameon=True)
        plt.show()
    return fig, ax


def plot_mlp_bpcrr_comparison(report: dict):
    comparison = build_comparison_mlp_bpcrr(report)
    order = [
        "MLP avgGRM weighted",
        "BPCRR | unweighted",
        report["best_bpcrr_shared_label"],
    ]
    palette = {label: _model_colors(report).get(label, "#B279A2") for label in order}

    with plt.rc_context(BOXPLOT_STYLE):
        fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
        sns.boxplot(
            data=comparison,
            x="corr",
            y="model_label",
            order=order,
            hue="model_label",
            hue_order=order,
            palette=palette,
            dodge=False,
            linewidth=1.1,
            fliersize=0,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        sns.stripplot(
            data=comparison,
            x="corr",
            y="model_label",
            order=order,
            color="black",
            alpha=0.72,
            size=5.5,
            ax=ax,
        )
        _add_mean_markers(ax, comparison, order)
        _annotate_sample_sizes(ax, comparison, order)
        ax.set_title("MLP vs BPCRR on the shared 15-island subset", fontsize=17)
        ax.set_xlabel("Outer-fold Pearson r")
        ax.set_ylabel("")
        ax.legend(loc="lower right", frameon=True)
        plt.show()
    return fig, ax


def plot_island_heatmap_and_winners(report: dict):
    heatmap_df = build_heatmap_df(report)
    winner_counts = build_winner_counts(report)

    with plt.rc_context(BOXPLOT_STYLE):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(15.5, 6.4),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [2.7, 1.0]},
        )

        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            linewidths=0.5,
            cbar_kws={"label": "Pearson r"},
            ax=axes[0],
        )
        axes[0].set_title("Per-island outer-fold Pearson r on the common 11 islands", fontsize=16)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("")

        palette = {label: _model_colors(report).get(label, "#B279A2") for label in winner_counts["model_label"]}
        sns.barplot(
            data=winner_counts,
            x="wins",
            y="model_label",
            hue="model_label",
            palette=palette,
            dodge=False,
            ax=axes[1],
        )
        if axes[1].legend_ is not None:
            axes[1].legend_.remove()
        axes[1].set_title("Island winners", fontsize=16)
        axes[1].set_xlabel("Number of islands won")
        axes[1].set_ylabel("")
        plt.show()
    return fig, axes


def plot_mlp_selection_summary(report: dict):
    mlp_df = report["mlp_df"]
    scheme_summary = build_scheme_summary(report)
    scheme_order = scheme_summary["weight_scheme"].tolist()
    palette = {name: SCHEME_COLORS.get(name, "#808080") for name in scheme_order}

    with plt.rc_context(BOXPLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.3), constrained_layout=True)

        sns.barplot(
            data=scheme_summary,
            x="count",
            y="weight_scheme",
            hue="weight_scheme",
            hue_order=scheme_order,
            palette=palette,
            dodge=False,
            ax=axes[0],
        )
        if axes[0].legend_ is not None:
            axes[0].legend_.remove()
        axes[0].set_title("How often each weighting scheme was selected", fontsize=16)
        axes[0].set_xlabel("Winning outer folds")
        axes[0].set_ylabel("Weighting scheme")

        sns.boxplot(
            data=mlp_df,
            x="corr",
            y="weight_scheme",
            order=scheme_order,
            hue="weight_scheme",
            hue_order=scheme_order,
            palette=palette,
            dodge=False,
            linewidth=1.1,
            fliersize=0,
            ax=axes[1],
        )
        if axes[1].legend_ is not None:
            axes[1].legend_.remove()
        sns.stripplot(
            data=mlp_df,
            x="corr",
            y="weight_scheme",
            order=scheme_order,
            color="black",
            alpha=0.72,
            size=5.5,
            ax=axes[1],
        )
        axes[1].set_title("Outer-fold Pearson r by selected weighting scheme", fontsize=16)
        axes[1].set_xlabel("Outer-fold Pearson r")
        axes[1].set_ylabel("Weighting scheme")
        plt.show()
    return fig, axes


def plot_mlp_preference_grid(report: dict):
    mlp_df = report["mlp_df"]
    specs = [
        ("optimizer", "Optimizer"),
        ("loss", "Loss"),
        ("use_snp_selection_label", "Use SNP selection"),
        ("weight_clip_max_label", "Weight clip max"),
        ("hidden_dims", "Hidden dims"),
    ]

    with plt.rc_context(BOXPLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)
        axes = axes.ravel()

        for ax, (column, title) in zip(axes[:5], specs):
            counts = build_preference_counts(report, column)
            sns.barplot(data=counts, x="count", y="choice", color="#4C78A8", ax=ax)
            ax.set_title(title, fontsize=15)
            ax.set_xlabel("Winning outer folds")
            ax.set_ylabel("")

        axes[5].axis("off")
        axes[5].text(
            0.0,
            0.80,
            f"Batch norm selected in {int(mlp_df['batch_norm'].sum())} / {len(mlp_df)} winning folds.",
            fontsize=12,
        )
        axes[5].text(
            0.0,
            0.58,
            f"Mean inner r across winning folds: {mlp_df['mean_inner_r'].mean():.3f}",
            fontsize=12,
        )
        axes[5].text(
            0.0,
            0.36,
            f"Mean outer r across winning folds: {mlp_df['corr'].mean():.3f}",
            fontsize=12,
        )
        axes[5].text(
            0.0,
            0.14,
            "This is a winner-only summary; non-winning Optuna trials are not stored in the JSON.",
            fontsize=11,
        )
        plt.show()
    return fig, axes
