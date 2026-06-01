"""Plot thesis figures for E6 AvgGRM-weighted PC ridge diagnostics."""

from __future__ import annotations

import gc
import json
import sys
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

SCHEME_COLORS = {
    "uniform": "#4C78A8",
    "linear": "#59A14F",
    "exponential": "#F28E2B",
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


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denom = float(np.sum(weights**2))
    if denom <= 0:
        return 0.0
    return float((np.sum(weights) ** 2) / denom)


def load_e6_avggrm_per_fold_results(
    results_root: Path,
    trait_order: list[str] | None = None,
) -> pd.DataFrame:
    trait_order = trait_order or TRAIT_ORDER
    rows: list[dict[str, Any]] = []

    for trait in trait_order:
        result_path = results_root / trait / f"e6_pc_ridge_avggrm_{trait}_results.json"
        if not result_path.exists():
            continue

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        best_by_fold = {
            int(item.get("fold")): item
            for item in payload.get("best_params_per_fold", [])
            if item.get("fold") is not None
        }

        for metric in payload.get("per_fold_metrics", []):
            fold = int(metric["fold"])
            best_entry = best_by_fold.get(fold, {})
            best_params = best_entry.get("best_params", {})
            weighting = metric.get("weighting", {}) or best_params.get("weighting", {}) or {}
            inner_islands = metric.get("inner_validation_islands") or best_entry.get("inner_validation_islands") or []

            row = {
                "trait": trait,
                "trait_label": TRAIT_LABELS.get(trait, trait),
                "fold": fold,
                "test_island_code": metric.get("test_island"),
                "test_island": metric.get("test_island_name"),
                "pearson_r": metric.get("test_corr"),
                "mean_inner_r": best_entry.get("mean_inner_r"),
                "alpha": metric.get("alpha", best_params.get("alpha")),
                "n_pcs": best_params.get("n_pcs", metric.get("num_snps")),
                "num_snps": metric.get("num_snps", best_params.get("num_snps")),
                "weight_scheme": str(weighting.get("name", "uniform")).lower(),
                "weight_floor": weighting.get("floor"),
                "weight_clip_max": weighting.get("clip_max"),
                "weight_beta": weighting.get("beta"),
                "weight_min": weighting.get("min_weight"),
                "weight_max": weighting.get("max_weight"),
                "inner_validation_top_k_used": metric.get("inner_validation_top_k_related_islands_used"),
                "inner_validation_n_samples": int(sum(float(item.get("n_samples", 0)) for item in inner_islands)),
                "top_inner_avg_grm": inner_islands[0].get("avg_grm_to_outer_test") if inner_islands else np.nan,
                "source_file": str(result_path),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def _include_islands_mask(
    locality: np.ndarray,
    code_to_label: dict[int, str],
    include_islands: Any,
) -> np.ndarray:
    if not include_islands:
        return np.ones(len(locality), dtype=bool)

    if not isinstance(include_islands, (list, tuple, set, np.ndarray)):
        include_islands = [include_islands]

    label_to_code = {str(label): int(code) for code, label in (code_to_label or {}).items()}
    include_codes: set[int] = set()
    for value in include_islands:
        value_str = str(value)
        if value_str in label_to_code:
            include_codes.add(label_to_code[value_str])
            continue
        try:
            include_codes.add(int(value))
        except ValueError:
            pass

    if not include_codes:
        raise ValueError(f"include_islands={include_islands} did not match locality codes or labels.")
    return np.isin(locality.astype(int), np.fromiter(include_codes, dtype=int))


def _load_e6_config(config_dir: Path, trait: str) -> dict[str, Any]:
    config_path = config_dir / f"final_e6_pc_ridge_avggrm_{trait}_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing E6 config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _resolve_data_paths(paths: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    resolved = dict(paths)
    for key in ("npz", "npz_path", "grm_rds", "snp_feather", "phenotype_csv"):
        value = resolved.get(key)
        if value is None:
            continue
        path = Path(value)
        if not path.is_absolute():
            resolved[key] = str(repo_root / path)
    return resolved


def compute_e6_avggrm_diagnostics(
    repo_root: Path,
    results_root: Path,
    config_dir: Path,
    trait_order: list[str] | None = None,
) -> pd.DataFrame:
    trait_order = trait_order or TRAIT_ORDER
    repo_root = Path(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.avggrm_weighting import avg_grm_train_to_target, ranks_from_desc_scores, weights_from_scheme
    from src.data import load_data

    per_fold = load_e6_avggrm_per_fold_results(results_root, trait_order=trait_order)
    if per_fold.empty:
        return per_fold

    diagnostic_frames: list[pd.DataFrame] = []

    for trait in trait_order:
        trait_rows = per_fold[per_fold["trait"].eq(trait)].copy()
        if trait_rows.empty:
            continue

        cfg = _load_e6_config(config_dir, trait)
        base = cfg["base_train"]
        paths = _resolve_data_paths(dict(base["paths"]), repo_root)

        X, y, ids, grm_df, locality, code_to_label, y_eval = load_data(
            paths,
            target_column=base.get("target_column", "y_adjusted"),
            standardize_features=bool(base.get("standardize_features", False)),
            return_locality=True,
            min_count=int(base.get("min_count", 0) or 0),
            return_eval=True,
            eval_target_column=base.get("eval_target_column"),
        )
        del X, y, ids, y_eval

        include_mask = _include_islands_mask(locality, code_to_label, cfg.get("cv", {}).get("include_islands"))
        if not bool(np.all(include_mask)):
            locality = locality[include_mask]
            grm_df = grm_df.iloc[include_mask, include_mask]

        if grm_df is None:
            raise ValueError(f"Config for {trait} does not provide a GRM, so AvgGRM ESS cannot be computed.")
        grm_mat = grm_df.to_numpy(dtype=np.float64)

        diag_rows = []
        for row in trait_rows.itertuples(index=False):
            test_code = int(row.test_island_code)
            train_idx = np.where(locality.astype(int) != test_code)[0]
            test_idx = np.where(locality.astype(int) == test_code)[0]
            if train_idx.size == 0 or test_idx.size == 0:
                continue

            weight_spec = {
                "name": row.weight_scheme,
                "floor": 1e-6 if pd.isna(row.weight_floor) else row.weight_floor,
                "clip_max": None if pd.isna(row.weight_clip_max) else row.weight_clip_max,
            }
            if pd.notna(row.weight_beta):
                weight_spec["beta"] = row.weight_beta
            if pd.notna(row.weight_min):
                weight_spec["min_weight"] = row.weight_min
            if pd.notna(row.weight_max):
                weight_spec["max_weight"] = row.weight_max

            if row.weight_scheme == "uniform":
                weights = np.ones(train_idx.size, dtype=float)
                avg_grm = np.zeros(train_idx.size, dtype=float)
            else:
                avg_grm = avg_grm_train_to_target(grm_mat, train_idx, test_idx)
                ranks = ranks_from_desc_scores(avg_grm)
                weights = weights_from_scheme(avg_grm, ranks, weight_spec)

            sorted_weights = np.sort(weights)[::-1]
            top10_n = max(1, int(np.ceil(0.10 * len(sorted_weights))))
            top20_n = max(1, int(np.ceil(0.20 * len(sorted_weights))))
            ess = effective_sample_size(weights)

            diag_rows.append(
                {
                    "trait": trait,
                    "trait_label": TRAIT_LABELS.get(trait, trait),
                    "fold": int(row.fold),
                    "test_island_code": test_code,
                    "test_island": row.test_island,
                    "pearson_r": float(row.pearson_r),
                    "mean_inner_r": row.mean_inner_r,
                    "alpha": float(row.alpha),
                    "log10_alpha": float(np.log10(row.alpha)),
                    "n_pcs": int(row.n_pcs),
                    "num_snps": int(row.num_snps),
                    "weight_scheme": row.weight_scheme,
                    "weight_clip_max": row.weight_clip_max,
                    "weight_beta": row.weight_beta,
                    "inner_validation_top_k_used": row.inner_validation_top_k_used,
                    "inner_validation_n_samples": row.inner_validation_n_samples,
                    "top_inner_avg_grm": row.top_inner_avg_grm,
                    "n_train": int(train_idx.size),
                    "n_test": int(test_idx.size),
                    "kish_ess": ess,
                    "ess_ratio": ess / float(train_idx.size),
                    "ess_loss_pct": 100.0 * (1.0 - ess / float(train_idx.size)),
                    "max_weight": float(np.max(weights)),
                    "p95_weight": float(np.quantile(weights, 0.95)),
                    "p99_weight": float(np.quantile(weights, 0.99)),
                    "weight_cv": float(np.std(weights) / max(np.mean(weights), 1e-12)),
                    "top10_weight_share": float(np.sum(sorted_weights[:top10_n]) / np.sum(sorted_weights)),
                    "top20_weight_share": float(np.sum(sorted_weights[:top20_n]) / np.sum(sorted_weights)),
                    "avg_grm_mean": float(np.mean(avg_grm)),
                    "avg_grm_sd": float(np.std(avg_grm)),
                    "avg_grm_p95": float(np.quantile(avg_grm, 0.95)),
                }
            )

        diagnostic_frames.append(pd.DataFrame(diag_rows))
        del grm_df, grm_mat, locality
        gc.collect()

    return pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame()


def summarize_e6_avggrm_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics

    mode_scheme = (
        diagnostics.groupby("trait")["weight_scheme"]
        .agg(lambda values: values.value_counts().index[0])
        .rename("modal_weight_scheme")
    )
    summary = (
        diagnostics.groupby(["trait", "trait_label"], as_index=False)
        .agg(
            mean_r=("pearson_r", "mean"),
            sd_r=("pearson_r", "std"),
            median_n_pcs=("n_pcs", "median"),
            median_alpha=("alpha", "median"),
            median_ess=("kish_ess", "median"),
            median_ess_ratio=("ess_ratio", "median"),
            min_ess_ratio=("ess_ratio", "min"),
            median_top10_weight_share=("top10_weight_share", "median"),
        )
        .merge(mode_scheme.reset_index(), on="trait", how="left")
    )
    return summary.sort_values("trait").reset_index(drop=True)


def plot_e6_avggrm_diagnostic_dashboard(
    diagnostics: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
    file_stem: str = "e6_avggrm_diagnostic_dashboard",
) -> tuple[Path, Path] | None:
    if diagnostics.empty:
        print("No E6 AvgGRM diagnostics found.")
        return None

    trait_order = [trait for trait in TRAIT_ORDER if trait in set(diagnostics["trait"])]
    trait_labels = [TRAIT_LABELS.get(trait, trait) for trait in trait_order]
    palette = {"Body mass": "#4C78A8", "Tarsus length": "#59A14F", "Wing length": "#F28E2B"}

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
    ax = axes[0, 0]
    sns.boxplot(
        data=diagnostics,
        x="trait_label",
        y="n_pcs",
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
        y="n_pcs",
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
        y="alpha",
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
        y="alpha",
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
    counts = (
        diagnostics.groupby(["trait_label", "weight_scheme"], as_index=False)
        .size()
        .pivot(index="trait_label", columns="weight_scheme", values="size")
        .reindex(trait_labels)
        .fillna(0)
    )
    bottom = np.zeros(len(counts), dtype=float)
    x = np.arange(len(counts))
    for scheme in ["uniform", "linear", "exponential"]:
        values = counts[scheme].to_numpy(dtype=float) if scheme in counts.columns else np.zeros(len(counts))
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=SCHEME_COLORS.get(scheme, "0.5"),
            label=scheme.capitalize(),
            width=0.62,
            edgecolor="white",
            linewidth=0.6,
        )
        bottom += values
    ax.set_title("Selected weighting rule")
    ax.set_xlabel("")
    ax.set_ylabel("Outer folds")
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=20, ha="right")
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)

    ax = axes[1, 1]
    sns.boxplot(
        data=diagnostics,
        x="trait_label",
        y="ess_ratio",
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
        y="ess_ratio",
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
    ax.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.set_title("Kish effective sample size")
    ax.set_xlabel("")
    ax.set_ylabel("ESS / training N")
    ax.tick_params(axis="x", rotation=20)
    style_axes(ax)

    output_dir = Path(output_dir)
    pdf_path = save_figure(fig, output_dir / f"{file_stem}.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / f"{file_stem}.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def plot_e6_avggrm_island_diagnostics(
    diagnostics: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
    file_stem: str = "e6_avggrm_island_diagnostics",
) -> tuple[Path, Path] | None:
    if diagnostics.empty:
        print("No E6 AvgGRM diagnostics found.")
        return None

    diagnostics = diagnostics.copy()
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
    axes[0].set_title("E6 prediction by held-out island")
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


def make_e6_avggrm_figures(
    repo_root: Path | None = None,
    output_dir: Path | None = None,
    results_root: Path | None = None,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    repo_root = find_repo_root() if repo_root is None else Path(repo_root)
    output_dir = repo_root / "figures" if output_dir is None else Path(output_dir)
    results_root = (
        repo_root / "outputs" / "final_results" / "e6_pc_ridge_avggrm"
        if results_root is None
        else Path(results_root)
    )
    config_dir = repo_root / "config" / "e6" if config_dir is None else Path(config_dir)

    configure_plot_style()
    diagnostics = compute_e6_avggrm_diagnostics(
        repo_root=repo_root,
        results_root=results_root,
        config_dir=config_dir,
    )
    summary = summarize_e6_avggrm_diagnostics(diagnostics)

    local_csv = save_csv(
        diagnostics,
        repo_root / "outputs" / "final_results" / "e6_pc_ridge_avggrm" / "e6_avggrm_fold_diagnostics.csv",
        repo_root,
    )

    summary_csv = save_csv(
        summary,
        repo_root / "outputs" / "final_results" / "e6_pc_ridge_avggrm" / "e6_avggrm_diagnostic_summary.csv",
        repo_root,
    )

    paths = {
        "diagnostics": diagnostics,
        "summary": summary,
        "diagnostics_csv": local_csv,
        "summary_csv": summary_csv,
        "dashboard": plot_e6_avggrm_diagnostic_dashboard(diagnostics, output_dir, repo_root),
        "island_diagnostics": plot_e6_avggrm_island_diagnostics(diagnostics, output_dir, repo_root),
    }
    return paths
