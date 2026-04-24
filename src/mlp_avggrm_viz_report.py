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

MODEL_COLORS = {
    "MLP avgGRM weighted": "#4C78A8",
    "MLP avgGRM weighted (smart 5 inner islands)": "#9C755F",
    "MLP avgGRM weighted (smart 3 inner islands)": "#BAB0AC",
    "Ridge (avgGRM nested CV)": "#2E8B57",
    "Ridge (density-ratio nested CV)": "#A05195",
    "BPCRR | full_source_unweighted": "#BAB0AC",
    "Ridge (PCA) | full_source_unweighted": "#8C564B",
    "Ridge (avgGRM) | full_source_unweighted": "#B279A2",
    "BPCRR | bpcrr_topk_avggrm (k=1500)": "#E45756",
    "BPCRR | bpcrr_topk_pc_distance (k=1500)": "#72B7B2",
    "Ridge (PCA) | pca_source_topk (k=1500)": "#F58518",
    "Ridge (avgGRM) | avggrm_topk (k=1500)": "#54A24B",
    "Ridge (PEV) | pevmean_ga (k=1500)": "#FF9DA6",
}

SCHEME_COLORS = {
    "uniform": "#4C78A8",
    "linear": "#72B7B2",
    "minmax": "#54A24B",
    "exponential": "#F58518",
    "top-heavy": "#E45756",
    "pc_logistic": "#A05195",
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


def _load_mlp_results(path: Path, model_label: str) -> tuple[dict, pd.DataFrame]:
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
                "model_label": model_label,
                "fold": fold,
                "target_island": int(fold_row["test_island"]),
                "target_island_name": str(fold_row["test_island_name"]),
                "corr": float(fold_row["test_corr"]),
                "test_size": int(fold_row["test_size"]),
                "mean_inner_r": float(best_row["mean_inner_r"]) if best_row.get("mean_inner_r") is not None else np.nan,
                "inner_top_k_related_islands": payload.get("inner_top_k_related_islands"),
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


def _load_nested_ridge_results(path: Path, model_label: str) -> tuple[dict, pd.DataFrame]:
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
        mean_inner_ess = params.get("mean_inner_effective_sample_size")

        rows.append(
            {
                "model_label": model_label,
                "fold": fold,
                "target_island": int(fold_row["test_island"]),
                "target_island_name": str(fold_row["test_island_name"]),
                "corr": float(fold_row["test_corr"]),
                "test_size": int(fold_row["test_size"]),
                "mean_inner_r": float(best_row["mean_inner_r"]) if best_row.get("mean_inner_r") is not None else np.nan,
                "inner_top_k_related_islands": payload.get("inner_top_k_related_islands"),
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
                "weight_prob_clip": float(weighting["prob_clip"]) if weighting.get("prob_clip") is not None else np.nan,
                "weight_n_components": float(weighting["n_components"]) if weighting.get("n_components") is not None else np.nan,
                "weight_logistic_c": float(weighting["logistic_c"]) if weighting.get("logistic_c") is not None else np.nan,
                "weight_pca_fit": weighting.get("pca_fit"),
                "alpha": float(params["alpha"]) if params.get("alpha") is not None else np.nan,
                "use_snp_selection": bool(params.get("use_snp_selection", False)),
                "num_snps": int(params["num_snps"]) if params.get("num_snps") is not None else np.nan,
                "mean_inner_effective_sample_size": float(mean_inner_ess) if mean_inner_ess is not None else np.nan,
                "effective_sample_size": float(fold_row["effective_sample_size"]) if fold_row.get("effective_sample_size") is not None else np.nan,
                "n_components_used": int(fold_row["n_components_used"]) if fold_row.get("n_components_used") is not None else np.nan,
            }
        )

    df = pd.DataFrame(rows).sort_values(["target_island", "fold"]).reset_index(drop=True)
    df["use_snp_selection_label"] = np.where(df["use_snp_selection"], "True", "False")
    df["weight_clip_max_label"] = df["weight_clip_max"].map(lambda x: "none" if pd.isna(x) else f"{x:g}")
    if not df.empty:
        df["alpha_log10"] = np.log10(df["alpha"].clip(lower=1e-12))
    else:
        df["alpha_log10"] = pd.Series(dtype=float)
    return payload, df


def _load_reference_results(path: Path, model_key: str, model_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map = {}
    if "selection_method" in df.columns and "method" not in df.columns:
        rename_map["selection_method"] = "method"
    if "n_train_size" in df.columns and "n_individuals" not in df.columns:
        rename_map["n_train_size"] = "n_individuals"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["target_island", "method", "corr_eval"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{model_key}: missing required columns {missing} in {path}")

    df["target_island"] = pd.to_numeric(df["target_island"], errors="coerce")
    if "n_individuals" in df.columns:
        df["n_individuals"] = pd.to_numeric(df["n_individuals"], errors="coerce")
    else:
        df["n_individuals"] = np.nan
    df["corr"] = pd.to_numeric(df["corr_eval"], errors="coerce")
    df = df.dropna(subset=["target_island", "corr"]).copy()
    df["target_island"] = df["target_island"].astype(int)

    if "target_island_name" not in df.columns:
        df["target_island_name"] = df["target_island"].astype(str)
    df["target_island_name"] = df["target_island_name"].astype(str)

    if "analysis" not in df.columns:
        df["analysis"] = np.where(df["method"].astype(str).eq("full_source_unweighted"), "full_baseline", "ranked_subset")

    df["model_key"] = model_key
    df["model_label"] = model_label
    return df


def _build_island_name_map(*dfs: pd.DataFrame) -> dict[int, str]:
    out: dict[int, str] = {}
    for df in dfs:
        if df.empty or "target_island" not in df.columns or "target_island_name" not in df.columns:
            continue
        sub = (
            df[["target_island", "target_island_name"]]
            .dropna()
            .drop_duplicates()
            .sort_values("target_island")
        )
        for island_id, island_name in sub.itertuples(index=False):
            out.setdefault(int(island_id), str(island_name))
    return out


def _available_islands_for_spec(report: dict, spec: dict) -> set[int]:
    if spec["source"] == "mlp":
        return set(report["mlp_variants_df"]["target_island"].astype(int).tolist())
    if spec["source"] == "ridge_nested":
        df = report[spec.get("df_key", "ridge_nested_df")]
        return set(df["target_island"].astype(int).tolist())

    reference_df = report["reference_df"]
    sub = reference_df[
        (reference_df["model_key"] == spec["model_key"])
        & (reference_df["method"].astype(str) == str(spec["method"]))
    ].copy()
    if spec.get("n_individuals") is not None:
        sub = sub[sub["n_individuals"].eq(spec["n_individuals"])]
    return set(sub["target_island"].astype(int).tolist())


def _common_islands_for_specs(report: dict, specs: list[dict]) -> list[int]:
    island_sets = [_available_islands_for_spec(report, spec) for spec in specs]
    common = set.intersection(*island_sets) if island_sets else set()
    return sorted(common)


def _mlp_subset(df: pd.DataFrame, label: str, island_ids: list[int] | None = None) -> pd.DataFrame:
    subset = df.copy()
    if island_ids is not None:
        subset = subset.loc[subset["target_island"].isin(island_ids)].copy()
    subset = subset[["target_island", "target_island_name", "corr"]].copy()
    subset["model_label"] = label
    return subset


def _reference_subset(
    reference_df: pd.DataFrame,
    model_key: str,
    method: str,
    label: str,
    island_ids: list[int] | None = None,
    n_individuals: int | None = None,
) -> pd.DataFrame:
    subset = reference_df[
        (reference_df["model_key"] == model_key)
        & (reference_df["method"].astype(str) == str(method))
    ].copy()
    if n_individuals is not None:
        subset = subset[subset["n_individuals"].eq(n_individuals)].copy()
    if island_ids is not None:
        subset = subset[subset["target_island"].isin(island_ids)].copy()
    subset = (
        subset.groupby(["target_island", "target_island_name"], as_index=False)["corr"]
        .mean()
        .sort_values("target_island")
        .reset_index(drop=True)
    )
    subset["model_label"] = label
    return subset


def _collect_comparison_points(report: dict, specs: list[dict], island_ids: list[int]) -> pd.DataFrame:
    frames = []
    for spec in specs:
        if spec["source"] == "mlp":
            variant_df = report["mlp_variants_df"].loc[report["mlp_variants_df"]["model_label"] == spec["label"]].copy()
            frames.append(_mlp_subset(variant_df, spec["label"], island_ids))
        elif spec["source"] == "ridge_nested":
            frames.append(_mlp_subset(report[spec.get("df_key", "ridge_nested_df")], spec["label"], island_ids))
        else:
            frames.append(
                _reference_subset(
                    report["reference_df"],
                    spec["model_key"],
                    spec["method"],
                    spec["label"],
                    island_ids,
                    spec.get("n_individuals"),
                )
            )
    return pd.concat(frames, ignore_index=True)


def _summarize_comparison(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("model_label", as_index=False)
        .agg(mean_corr=("corr", "mean"), median_corr=("corr", "median"), std_corr=("corr", "std"), n=("corr", "size"))
        .sort_values("mean_corr", ascending=False)
        .reset_index(drop=True)
    )


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


def _plot_comparison_boxplot(data: pd.DataFrame, order: list[str], title: str, figsize: tuple[float, float]):
    palette = {label: MODEL_COLORS.get(label, "#808080") for label in order}
    with plt.rc_context(BOXPLOT_STYLE):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        sns.boxplot(
            data=data,
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
            data=data,
            x="corr",
            y="model_label",
            order=order,
            color="black",
            alpha=0.72,
            size=5.5,
            ax=ax,
        )
        _add_mean_markers(ax, data, order)
        _annotate_sample_sizes(ax, data, order)
        ax.set_title(title, fontsize=17)
        ax.set_xlabel("Pearson r")
        ax.set_ylabel("")
        ax.legend(loc="lower right", frameon=True)
        plt.show()
    return fig, ax


def prepare_report_data(root: Path | None = None, topk_value: int = 1500) -> dict:
    project_root = _project_root(root)
    ridge_nested_label = "Ridge (avgGRM nested CV)"
    ridge_importance_weighted_label = "Ridge (density-ratio nested CV)"

    mlp_paths = {
        "MLP avgGRM weighted": project_root / "outputs" / "nested_cv" / "mlp_avggrm_weighted_results.json",
        "MLP avgGRM weighted (smart 5 inner islands)": project_root / "outputs" / "nested_cv" / "mlp_avggrm_weighted_smart5_results.json",
        "MLP avgGRM weighted (smart 3 inner islands)": project_root / "outputs" / "nested_cv" / "mlp_avggrm_weighted_smart3_results.json",
    }
    reference_paths = {
        "bpcrr_main": (
            project_root / "outputs" / "bpcrr_inla_2.0" / "bpcrr_inla_rank_select_results.csv",
            "BPCRR",
        ),
        "ridge_pca": (
            project_root / "outputs" / "pca_source_rank_weight_test4" / "pca_source_rank_weight_results.csv",
            "Ridge (PCA)",
        ),
        "ridge_avggrm": (
            project_root / "outputs" / "avggrm_rank_weight_all" / "body_mass" / "avggrm_rank_weight_results.csv",
            "Ridge (avgGRM)",
        ),
        "ridge_pev": (
            project_root / "outputs" / "ridge_pev" / "pevmean_ga_results.csv",
            "Ridge (PEV)",
        ),
    }
    ridge_nested_path = project_root / "outputs" / "nested_cv" / "ridge_avggrm_weighted_nested_results.json"
    ridge_importance_weighted_path = (
        project_root / "outputs" / "nested_cv" / "ridge" / "ridge_importance_weighted_nested_results.json"
    )

    mlp_payloads: dict[str, dict] = {}
    mlp_variant_frames = []
    for label, path in mlp_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing MLP nested CV results file: {path}")
        payload, df = _load_mlp_results(path, label)
        mlp_payloads[label] = payload
        mlp_variant_frames.append(df)

    mlp_variants_df = pd.concat(mlp_variant_frames, ignore_index=True)
    mlp_df = mlp_variants_df.loc[mlp_variants_df["model_label"] == "MLP avgGRM weighted"].copy()

    ridge_nested_payload: dict | None = None
    if ridge_nested_path.exists():
        ridge_nested_payload, ridge_nested_df = _load_nested_ridge_results(ridge_nested_path, ridge_nested_label)
    else:
        ridge_nested_df = pd.DataFrame()

    ridge_importance_weighted_payload: dict | None = None
    if ridge_importance_weighted_path.exists():
        ridge_importance_weighted_payload, ridge_importance_weighted_df = _load_nested_ridge_results(
            ridge_importance_weighted_path,
            ridge_importance_weighted_label,
        )
    else:
        ridge_importance_weighted_df = pd.DataFrame()

    reference_frames = []
    for model_key, (path, model_label) in reference_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing reference results file: {path}")
        reference_frames.append(_load_reference_results(path, model_key, model_label))
    reference_df = pd.concat(reference_frames, ignore_index=True)

    island_name_map = _build_island_name_map(mlp_variants_df, ridge_nested_df, ridge_importance_weighted_df, reference_df)

    mlp_variant_labels = list(mlp_paths.keys())

    full_baseline_specs = [
        *({"source": "mlp", "label": label} for label in mlp_variant_labels),
        *(
            [
                {
                    "source": "ridge_nested",
                    "label": ridge_nested_label,
                    "df_key": "ridge_nested_df",
                }
            ]
            if not ridge_nested_df.empty
            else []
        ),
        *(
            [
                {
                    "source": "ridge_nested",
                    "label": ridge_importance_weighted_label,
                    "df_key": "ridge_importance_weighted_df",
                }
            ]
            if not ridge_importance_weighted_df.empty
            else []
        ),
        {
            "source": "reference",
            "label": "BPCRR | full_source_unweighted",
            "model_key": "bpcrr_main",
            "method": "full_source_unweighted",
            "n_individuals": None,
        },
        {
            "source": "reference",
            "label": "Ridge (PCA) | full_source_unweighted",
            "model_key": "ridge_pca",
            "method": "full_source_unweighted",
            "n_individuals": None,
        },
        {
            "source": "reference",
            "label": "Ridge (avgGRM) | full_source_unweighted",
            "model_key": "ridge_avggrm",
            "method": "full_source_unweighted",
            "n_individuals": None,
        },
    ]

    topk_specs = [
        *({"source": "mlp", "label": label} for label in mlp_variant_labels),
        {
            "source": "reference",
            "label": f"BPCRR | bpcrr_topk_avggrm (k={topk_value})",
            "model_key": "bpcrr_main",
            "method": "bpcrr_topk_avggrm",
            "n_individuals": topk_value,
        },
        {
            "source": "reference",
            "label": f"BPCRR | bpcrr_topk_pc_distance (k={topk_value})",
            "model_key": "bpcrr_main",
            "method": "bpcrr_topk_pc_distance",
            "n_individuals": topk_value,
        },
        {
            "source": "reference",
            "label": f"Ridge (PCA) | pca_source_topk (k={topk_value})",
            "model_key": "ridge_pca",
            "method": "pca_source_topk",
            "n_individuals": topk_value,
        },
        {
            "source": "reference",
            "label": f"Ridge (avgGRM) | avggrm_topk (k={topk_value})",
            "model_key": "ridge_avggrm",
            "method": "avggrm_topk",
            "n_individuals": topk_value,
        },
        {
            "source": "reference",
            "label": f"Ridge (PEV) | pevmean_ga (k={topk_value})",
            "model_key": "ridge_pev",
            "method": "pevmean_ga",
            "n_individuals": topk_value,
        },
    ]

    report = {
        "project_root": project_root,
        "topk_value": topk_value,
        "mlp_payload": mlp_payloads["MLP avgGRM weighted"],
        "mlp_payloads": mlp_payloads,
        "mlp_df": mlp_df,
        "mlp_variants_df": mlp_variants_df,
        "mlp_variant_labels": mlp_variant_labels,
        "ridge_nested_path": ridge_nested_path,
        "ridge_nested_label": ridge_nested_label,
        "ridge_nested_payload": ridge_nested_payload,
        "ridge_nested_df": ridge_nested_df,
        "ridge_importance_weighted_path": ridge_importance_weighted_path,
        "ridge_importance_weighted_label": ridge_importance_weighted_label,
        "ridge_importance_weighted_payload": ridge_importance_weighted_payload,
        "ridge_importance_weighted_df": ridge_importance_weighted_df,
        "reference_df": reference_df,
        "island_name_map": island_name_map,
        "full_baseline_specs": full_baseline_specs,
        "topk_specs": topk_specs,
    }
    report["full_baseline_islands"] = _common_islands_for_specs(report, full_baseline_specs)
    report["topk_islands"] = _common_islands_for_specs(report, topk_specs)
    return report


def build_full_baseline_comparison(report: dict) -> pd.DataFrame:
    return _collect_comparison_points(report, report["full_baseline_specs"], report["full_baseline_islands"])


def build_topk_comparison(report: dict) -> pd.DataFrame:
    return _collect_comparison_points(report, report["topk_specs"], report["topk_islands"])


def build_full_baseline_summary(report: dict) -> pd.DataFrame:
    return _summarize_comparison(build_full_baseline_comparison(report))


def build_topk_summary(report: dict) -> pd.DataFrame:
    return _summarize_comparison(build_topk_comparison(report))


def build_model_summary(report: dict) -> pd.DataFrame:
    mlp_variants_df = report["mlp_variants_df"]
    full_summary = build_full_baseline_summary(report).copy()
    topk_summary = build_topk_summary(report).copy()

    rows = []
    for label in report["mlp_variant_labels"]:
        sub = mlp_variants_df.loc[mlp_variants_df["model_label"] == label].copy()
        inner_desc = sub["inner_top_k_related_islands"].dropna().iloc[0] if sub["inner_top_k_related_islands"].notna().any() else None
        group = "MLP weighted full" if inner_desc is None else f"MLP weighted full (smart inner top-{int(inner_desc)})"
        rows.append(
            {
                "comparison_group": group,
                "label": label,
                "n_islands": int(sub["target_island"].nunique()),
                "mean_corr": float(sub["corr"].mean()),
                "median_corr": float(sub["corr"].median()),
                "std_corr": float(sub["corr"].std()),
            }
        )

    for _, row in full_summary.iterrows():
        if row["model_label"] in report["mlp_variant_labels"]:
            continue
        rows.append(
            {
                "comparison_group": f"Full baselines ({len(report['full_baseline_islands'])} islands)",
                "label": row["model_label"],
                "n_islands": int(row["n"]),
                "mean_corr": float(row["mean_corr"]),
                "median_corr": float(row["median_corr"]),
                "std_corr": float(row["std_corr"]),
            }
        )

    for _, row in topk_summary.iterrows():
        if row["model_label"] in report["mlp_variant_labels"]:
            continue
        rows.append(
            {
                "comparison_group": f"Top-k (k={report['topk_value']}, {len(report['topk_islands'])} islands)",
                "label": row["model_label"],
                "n_islands": int(row["n"]),
                "mean_corr": float(row["mean_corr"]),
                "median_corr": float(row["median_corr"]),
                "std_corr": float(row["std_corr"]),
            }
        )

    return pd.DataFrame(rows)


def build_full_heatmap_df(report: dict) -> pd.DataFrame:
    heatmap_df = build_full_baseline_comparison(report).pivot(
        index="target_island",
        columns="model_label",
        values="corr",
    )
    heatmap_df = heatmap_df.reindex(report["full_baseline_islands"])
    heatmap_df.index = [report["island_name_map"].get(int(idx), str(idx)) for idx in heatmap_df.index]
    return heatmap_df


def build_full_winner_counts(report: dict) -> pd.DataFrame:
    heatmap_df = build_full_heatmap_df(report)
    order = [spec["label"] for spec in report["full_baseline_specs"]]
    return (
        heatmap_df.idxmax(axis=1)
        .value_counts()
        .reindex(order, fill_value=0)
        .rename_axis("model_label")
        .reset_index(name="wins")
    )


def plot_full_baseline_comparison(report: dict):
    data = build_full_baseline_comparison(report)
    order = [spec["label"] for spec in report["full_baseline_specs"]]
    title = "MLP vs ridge/BPCRR comparisons on the shared 16-island set"
    return _plot_comparison_boxplot(data, order, title, (13.8, 6.1))


def plot_topk_comparison(report: dict):
    data = build_topk_comparison(report)
    order = [spec["label"] for spec in report["topk_specs"]]
    title = f"MLP vs 16-island top-k reference runs (k={report['topk_value']})"
    return _plot_comparison_boxplot(data, order, title, (14.2, 6.4))


def plot_full_baseline_heatmap_and_winners(report: dict):
    heatmap_df = build_full_heatmap_df(report)
    winner_counts = build_full_winner_counts(report)

    with plt.rc_context(BOXPLOT_STYLE):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(15.8, 6.5),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [2.8, 1.0]},
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
        axes[0].set_title("Per-island Pearson r for the shared full-source baselines", fontsize=16)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("")

        palette = {label: MODEL_COLORS.get(label, "#808080") for label in winner_counts["model_label"]}
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
    return _build_preference_counts_df(mlp_df, column)


def _build_preference_counts_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df[column]
        .astype(str)
        .value_counts()
        .rename_axis("choice")
        .reset_index(name="count")
    )


def _weighting_detail(row: pd.Series) -> str:
    if row["weight_scheme"] == "pc_logistic":
        parts = []
        if pd.notna(row.get("weight_n_components")):
            parts.append(f"PCs={int(row['weight_n_components'])}")
        if pd.notna(row.get("weight_logistic_c")):
            parts.append(f"C={row['weight_logistic_c']:.2g}")
        if row.get("weight_pca_fit"):
            parts.append(f"fit={row['weight_pca_fit']}")
        if pd.notna(row.get("weight_clip_max")):
            parts.append(f"clip={row['weight_clip_max_label']}")
        return ", ".join(parts) if parts else "pc_logistic"
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


def _nested_ridge_overview(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "model_label",
                "n_folds",
                "n_islands",
                "mean_inner_r",
                "mean_outer_r",
                "median_outer_r",
                "alpha_median",
                "alpha_geom_mean",
                "alpha_min",
                "alpha_max",
                "snp_selection_count",
                "snp_selection_rate",
                "median_num_snps_when_selected",
                "most_common_weight_scheme",
                "median_effective_sample_size",
            ]
        )

    selected_num_snps = df.loc[df["use_snp_selection"] & df["num_snps"].notna(), "num_snps"]
    scheme_counts = df["weight_scheme"].value_counts()
    geom_alpha = float(np.exp(np.log(df["alpha"].clip(lower=1e-12)).mean()))

    overview = pd.DataFrame(
        [
            {
                "model_label": model_label,
                "n_folds": int(len(df)),
                "n_islands": int(df["target_island"].nunique()),
                "mean_inner_r": float(df["mean_inner_r"].mean()),
                "mean_outer_r": float(df["corr"].mean()),
                "median_outer_r": float(df["corr"].median()),
                "alpha_median": float(df["alpha"].median()),
                "alpha_geom_mean": geom_alpha,
                "alpha_min": float(df["alpha"].min()),
                "alpha_max": float(df["alpha"].max()),
                "snp_selection_count": int(df["use_snp_selection"].sum()),
                "snp_selection_rate": float(df["use_snp_selection"].mean()),
                "median_num_snps_when_selected": float(selected_num_snps.median()) if not selected_num_snps.empty else np.nan,
                "most_common_weight_scheme": str(scheme_counts.index[0]) if not scheme_counts.empty else "",
                "median_effective_sample_size": float(df["effective_sample_size"].median())
                if "effective_sample_size" in df.columns and df["effective_sample_size"].notna().any()
                else np.nan,
            }
        ]
    )
    optional_columns = ["median_effective_sample_size"]
    return overview.drop(columns=[column for column in optional_columns if overview[column].isna().all()])


def _nested_ridge_scheme_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "weight_scheme",
                "count",
                "mean_outer_r",
                "median_outer_r",
                "mean_inner_r",
                "median_alpha",
                "snp_selection_count",
                "median_effective_sample_size",
            ]
        )

    summary = (
        df.groupby("weight_scheme", as_index=False)
        .agg(
            count=("corr", "size"),
            mean_outer_r=("corr", "mean"),
            median_outer_r=("corr", "median"),
            mean_inner_r=("mean_inner_r", "mean"),
            median_alpha=("alpha", "median"),
            snp_selection_count=("use_snp_selection", "sum"),
            median_effective_sample_size=("effective_sample_size", "median"),
            median_n_components=("weight_n_components", "median"),
            median_logistic_c=("weight_logistic_c", "median"),
        )
        .sort_values(["count", "mean_outer_r"], ascending=[False, False])
        .reset_index(drop=True)
    )
    optional_columns = ["median_effective_sample_size", "median_n_components", "median_logistic_c"]
    return summary.drop(columns=[column for column in optional_columns if summary[column].isna().all()])


def _nested_ridge_winner_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "fold",
                "target_island_name",
                "corr",
                "mean_inner_r",
                "alpha",
                "weight_scheme",
                "weighting_detail",
                "use_snp_selection_label",
                "num_snps",
            ]
        )

    columns = [
        "fold",
        "target_island_name",
        "corr",
        "mean_inner_r",
        "alpha",
        "weight_scheme",
        "use_snp_selection_label",
        "num_snps",
    ]
    optional_columns = [
        "effective_sample_size",
        "mean_inner_effective_sample_size",
        "weight_n_components",
        "weight_logistic_c",
        "weight_pca_fit",
    ]
    columns.extend(column for column in optional_columns if column in df.columns and df[column].notna().any())

    winner_table = df[columns].copy()
    winner_table["weighting_detail"] = df.apply(_weighting_detail, axis=1)

    ordered_columns = [
        "fold",
        "target_island_name",
        "corr",
        "mean_inner_r",
        "alpha",
        "weight_scheme",
        "weighting_detail",
        "use_snp_selection_label",
        "num_snps",
        *[column for column in optional_columns if column in winner_table.columns],
    ]
    return winner_table[ordered_columns]


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


def build_ridge_nested_overview(report: dict) -> pd.DataFrame:
    ridge_df = report["ridge_nested_df"]
    return _nested_ridge_overview(ridge_df, report["ridge_nested_label"])


def build_importance_weighted_ridge_overview(report: dict) -> pd.DataFrame:
    ridge_df = report["ridge_importance_weighted_df"]
    return _nested_ridge_overview(ridge_df, report["ridge_importance_weighted_label"])


def build_ridge_nested_scheme_summary(report: dict) -> pd.DataFrame:
    ridge_df = report["ridge_nested_df"]
    return _nested_ridge_scheme_summary(ridge_df)


def build_importance_weighted_ridge_scheme_summary(report: dict) -> pd.DataFrame:
    ridge_df = report["ridge_importance_weighted_df"]
    return _nested_ridge_scheme_summary(ridge_df)


def build_ridge_nested_winner_table(report: dict) -> pd.DataFrame:
    ridge_df = report["ridge_nested_df"]
    return _nested_ridge_winner_table(ridge_df)


def build_importance_weighted_ridge_winner_table(report: dict) -> pd.DataFrame:
    ridge_df = report["ridge_importance_weighted_df"]
    return _nested_ridge_winner_table(ridge_df)


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


def _plot_nested_ridge_selection_summary(
    ridge_df: pd.DataFrame,
    scheme_summary: pd.DataFrame,
    model_title: str,
):
    if ridge_df.empty:
        raise ValueError("No optimized nested-CV ridge results are available.")

    scheme_order = scheme_summary["weight_scheme"].tolist()
    palette = {name: SCHEME_COLORS.get(name, "#808080") for name in scheme_order}
    snp_selected = ridge_df.loc[ridge_df["use_snp_selection"] & ridge_df["num_snps"].notna(), "num_snps"]
    has_effective_sample_size = "effective_sample_size" in ridge_df.columns and ridge_df["effective_sample_size"].notna().any()

    with plt.rc_context(BOXPLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.3), constrained_layout=True)

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
        axes[0].set_title(f"{model_title}: selected weighting schemes", fontsize=15)
        axes[0].set_xlabel("Winning outer folds")
        axes[0].set_ylabel("Weighting scheme")

        sns.scatterplot(
            data=ridge_df,
            x="alpha",
            y="corr",
            hue="weight_scheme",
            hue_order=scheme_order,
            palette=palette,
            s=85,
            ax=axes[1],
        )
        axes[1].set_xscale("log")
        axes[1].set_title("Chosen ridge alpha vs outer-fold Pearson r", fontsize=15)
        axes[1].set_xlabel("Ridge alpha (log scale)")
        axes[1].set_ylabel("Outer-fold Pearson r")
        axes[1].legend(title="Weighting", loc="best", frameon=True)

        axes[2].axis("off")
        axes[2].text(
            0.0,
            0.88,
            f"SNP selection chosen in {int(ridge_df['use_snp_selection'].sum())} / {len(ridge_df)} folds.",
            fontsize=12,
        )
        axes[2].text(
            0.0,
            0.68,
            f"Median alpha: {ridge_df['alpha'].median():.3g}",
            fontsize=12,
        )
        axes[2].text(
            0.0,
            0.52,
            f"Geometric mean alpha: {np.exp(np.log(ridge_df['alpha'].clip(lower=1e-12)).mean()):.3g}",
            fontsize=12,
        )
        axes[2].text(
            0.0,
            0.36,
            f"Median num_snps when selected: {float(snp_selected.median()):.0f}" if not snp_selected.empty else "Median num_snps when selected: n/a",
            fontsize=12,
        )
        axes[2].text(
            0.0,
            0.20,
            f"Mean inner r: {ridge_df['mean_inner_r'].mean():.3f}\nMean outer r: {ridge_df['corr'].mean():.3f}",
            fontsize=12,
        )
        if has_effective_sample_size:
            axes[2].text(
                0.0,
                0.04,
                f"Median outer effective sample size: {ridge_df['effective_sample_size'].median():.0f}",
                fontsize=12,
            )
        plt.show()
    return fig, axes


def plot_ridge_nested_selection_summary(report: dict):
    return _plot_nested_ridge_selection_summary(
        report["ridge_nested_df"],
        build_ridge_nested_scheme_summary(report),
        "avgGRM ridge",
    )


def plot_importance_weighted_ridge_selection_summary(report: dict):
    return _plot_nested_ridge_selection_summary(
        report["ridge_importance_weighted_df"],
        build_importance_weighted_ridge_scheme_summary(report),
        "Density-ratio ridge",
    )


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


# Compatibility wrappers for the earlier notebook draft.
build_comparison_common = build_full_baseline_comparison
build_comparison_mlp_bpcrr = build_topk_comparison
build_heatmap_df = build_full_heatmap_df
build_winner_counts = build_full_winner_counts
plot_common_model_comparison = plot_full_baseline_comparison
plot_mlp_bpcrr_comparison = plot_topk_comparison
plot_island_heatmap_and_winners = plot_full_baseline_heatmap_and_winners
