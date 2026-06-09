"""Across-population diagnostic figure for the thesis Results section."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from thesis_style import TRAIT_COLORS, SEMANTIC_COLORS, configure_thesis_style, style_axes
except ModuleNotFoundError:  # pragma: no cover - allows package-style imports in tests/tools.
    from scripts.thesis_style import TRAIT_COLORS, SEMANTIC_COLORS, configure_thesis_style, style_axes


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

JSON_RUNS = {
    "MLP, uniform": {
        "root": "outputs/final_results/e8_mlp_uniform",
        "stem": "e8_mlp_uniform",
    },
    "MLP + importance": {
        "root": "outputs/final_results/e8_mlp_importance",
        "stem": "e8_mlp_importance",
    },
}

METHOD_COLORS = {
    "PC Ridge": "#6EA6D7",
    "PC Ridge + AvgGRM": "#3F7FB4",
    "PC Ridge + importance": "#1F4E79",
    "PC MLP, uniform": "#F4C06A",
    "MLP, uniform": "#D88A1D",
    "MLP + importance": "#9A5A00",
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

    for method, spec in JSON_RUNS.items():
        df = _load_json_run_diagnostics(repo_root, method, spec["root"], spec["stem"])
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    frames = [df.dropna(axis=1, how="all") for df in frames if not df.empty]
    diagnostics = pd.concat(frames, ignore_index=True, sort=False)
    diagnostics["trait"] = pd.Categorical(diagnostics["trait"], TRAIT_ORDER, ordered=True)
    diagnostics["trait_label"] = diagnostics["trait"].astype(str).map(TRAIT_LABELS)
    return diagnostics


def _load_json_run_diagnostics(repo_root: Path, method: str, rel_root: str, stem: str) -> pd.DataFrame:
    """Load fold diagnostics from nested-run JSON summaries."""
    rows: list[dict[str, object]] = []
    for trait in TRAIT_ORDER:
        path = repo_root / rel_root / trait / f"{stem}_{trait}_results.json"
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        folds = payload.get("per_fold_metrics", []) or []
        best_by_fold = {
            int(item.get("fold", idx + 1)): item
            for idx, item in enumerate(payload.get("best_params_per_fold", []) or [])
            if item.get("fold", idx + 1) is not None
        }
        total_n = sum(int(fold.get("test_size", 0) or 0) for fold in folds)

        for idx, fold in enumerate(folds):
            fold_id = int(fold.get("fold", idx + 1))
            best = best_by_fold.get(fold_id, {})
            test_size = int(fold.get("test_size", 0) or 0)
            n_train = int(total_n - test_size) if total_n and test_size else np.nan
            ess = fold.get("effective_sample_size")
            ess_float = float(ess) if ess is not None else np.nan
            ess_ratio = ess_float / n_train if np.isfinite(n_train) and n_train > 0 else np.nan
            mean_inner = best.get("mean_inner_r", fold.get("mean_inner_r"))
            pearson_r = float(fold.get("test_corr", fold.get("pearson_r", np.nan)))

            rows.append(
                {
                    "trait": trait,
                    "trait_label": TRAIT_LABELS[trait],
                    "fold": fold_id,
                    "test_island_code": int(fold.get("test_island", idx)),
                    "test_island": str(fold.get("test_island_name", "")),
                    "test_size": test_size,
                    "n_train": n_train,
                    "pearson_r": pearson_r,
                    "mean_inner_r": float(mean_inner) if mean_inner is not None else np.nan,
                    "outer_minus_inner_r": pearson_r - float(mean_inner) if mean_inner is not None else np.nan,
                    "model_type": fold.get("model_type"),
                    "weighting_method": (fold.get("weighting") or {}).get("name", "uniform"),
                    "effective_sample_size": ess_float,
                    "pre_shrink_effective_sample_size": fold.get("pre_shrink_effective_sample_size"),
                    "ess_ratio": ess_ratio,
                    "n_components": (fold.get("weighting") or {}).get("n_components", np.nan),
                    "n_components_used": fold.get("n_components_used", np.nan),
                    "source_file": str(path),
                    "method": method,
                }
            )

    return pd.DataFrame(rows)


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


def paired_importance_transfer(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Pair importance-weighted runs with their uniform counterparts."""
    if diagnostics.empty:
        return pd.DataFrame()

    key_cols = ["trait", "fold", "test_island_code", "test_island"]
    baselines = diagnostics[
        diagnostics["method"].isin(["PC Ridge", "MLP, uniform"])
    ][key_cols + ["method", "pearson_r", "mean_inner_r"]].rename(
        columns={"method": "baseline_method", "pearson_r": "baseline_r", "mean_inner_r": "baseline_inner_r"})
    weighted = diagnostics[
        diagnostics["method"].isin(["PC Ridge + importance", "MLP + importance"])
    ][
        key_cols
        + [
            "method",
            "pearson_r",
            "mean_inner_r",
            "ess_ratio",
            "effective_sample_size",
            "pre_shrink_effective_sample_size",
            "n_components_used",
            "weighting_method",
        ]
    ].copy()
    weighted["baseline_method"] = weighted["method"].map(
        {
            "PC Ridge + importance": "PC Ridge",
            "MLP + importance": "MLP, uniform",
        }
    )

    paired = weighted.merge(
        baselines,
        on=key_cols + ["baseline_method"],
        how="left",
    )
    paired["delta_r"] = paired["pearson_r"] - paired["baseline_r"]
    paired["delta_inner"] = paired["mean_inner_r"] - paired["baseline_inner_r"]
    paired["trait_label"] = paired["trait"].astype(str).map(TRAIT_LABELS)
    paired["weighted_model_label"] = paired["method"].map(
        {
            "PC Ridge + importance": "PC-ridge IW",
            "MLP + importance": "MLP IW",
        }
    )
    paired["weighting_label"] = paired["weighting_method"].map(
        {
            "pc_logistic": "PC-logistic",
            "uniform": "Uniform selected",
        }
    ).fillna(paired["weighting_method"].astype(str))
    return paired


def _style_axes(ax):
    style_axes(ax)
    ax.tick_params(axis="both", labelsize=10.5)


def plot_across_pop_diagnostics(
    diagnostics: pd.DataFrame,
    output_dir: Path | str = Path("figures"),
    file_stem: str = "across_pop_model_selection_diagnostics",
) -> tuple[Path, Path]:
    """Plot inner--outer transfer and importance-weighting diagnostics."""
    if diagnostics.empty:
        raise ValueError("No diagnostics available for across-population diagnostic plot.")

    configure_thesis_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (scatter_ax, ess_ax) = plt.subplots(1, 2, figsize=(10.2, 4.65), constrained_layout=True)

    # --- Left panel: inner vs outer for PC-ridge and MLP, uniform vs importance-weighted ---
    weight_color = {"Uniform": SEMANTIC_COLORS["observed"], "Importance-weighted": SEMANTIC_COLORS["adjusted"]}
    model_marker = {"PC-ridge": "o", "MLP": "^"}
    scatter_spec = [
        ("PC Ridge", "PC-ridge", "Uniform"),
        ("PC Ridge + importance", "PC-ridge", "Importance-weighted"),
        ("MLP, uniform", "MLP", "Uniform"),
        ("MLP + importance", "MLP", "Importance-weighted"),
    ]
    scatter_methods = [m for m, _, _ in scatter_spec]
    for method, model, weight in scatter_spec:
        sub = diagnostics[diagnostics["method"].eq(method)]
        if sub.empty:
            continue
        scatter_ax.scatter(
            sub["mean_inner_r"], sub["pearson_r"], s=34, marker=model_marker[model],
            color=weight_color[weight], edgecolor="white", linewidth=0.5, alpha=0.82,
        )
    finite = diagnostics[diagnostics["method"].isin(scatter_methods)][["mean_inner_r", "pearson_r"]].dropna()
    lim_min = float(np.floor((finite.min().min() - 0.025) * 20) / 20)
    lim_max = float(np.ceil((finite.max().max() + 0.025) * 20) / 20)
    scatter_ax.plot([lim_min, lim_max], [lim_min, lim_max], color=SEMANTIC_COLORS["reference"], lw=1.0, ls="--")
    # Mean inner score of each weighting group, to show the inner shift from weighting.
    for weight in ("Uniform", "Importance-weighted"):
        methods = [m for m, _, w in scatter_spec if w == weight]
        mean_inner = diagnostics[diagnostics["method"].isin(methods)]["mean_inner_r"].mean()
        if np.isfinite(mean_inner):
            scatter_ax.axvline(mean_inner, color=weight_color[weight], lw=1.3, ls=":")
    scatter_ax.set_xlim(lim_min, lim_max)
    scatter_ax.set_ylim(lim_min, lim_max)
    scatter_ax.set_xlabel("Inner validation Pearson $r$", fontsize=12.2)
    scatter_ax.set_ylabel("Outer LOIO Pearson $r$", fontsize=12.2)
    scatter_ax.set_title("Inner validation does not transfer", fontsize=13.2, pad=8)
    _style_axes(scatter_ax)
    weight_handles = [
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=c, markeredgecolor="white", label=w, markersize=7)
        for w, c in weight_color.items()
    ]
    model_handles = [
        Line2D([0], [0], marker=mk, linestyle="", color=SEMANTIC_COLORS["reference"], label=mdl, markersize=7)
        for mdl, mk in model_marker.items()
    ]
    leg1 = scatter_ax.legend(handles=weight_handles, title="Weighting", frameon=False, fontsize=9.4, title_fontsize=9.6, loc="upper left")
    scatter_ax.add_artist(leg1)
    scatter_ax.legend(handles=model_handles, title="Model", frameon=False, fontsize=9.4, title_fontsize=9.6, loc="lower right")

    # --- Right panel: weight concentration (Kish ESS), styled like the other boxplots ---
    weighted = diagnostics[
        diagnostics["method"].isin(["PC Ridge + AvgGRM", "PC Ridge + importance", "MLP + importance"])
    ].copy()
    weighted = weighted[weighted["ess_ratio"].notna()]
    ess_models = [
        ("PC Ridge + AvgGRM", "AvgGRM\nPC-ridge"),
        ("PC Ridge + importance", "Importance\nPC-ridge"),
        ("MLP + importance", "Importance\nMLP"),
    ]
    width = 0.8 / len(TRAIT_ORDER)
    for j, trait in enumerate(TRAIT_ORDER):
        offset = (j - (len(TRAIT_ORDER) - 1) / 2) * width
        color = TRAIT_COLORS[TRAIT_LABELS[trait]]
        positions, data = [], []
        for i, (method, _) in enumerate(ess_models):
            vals = weighted[(weighted["method"].eq(method)) & (weighted["trait"].astype(str).eq(trait))]["ess_ratio"].dropna().to_numpy()
            if vals.size:
                positions.append(i + offset)
                data.append(vals)
        if not data:
            continue
        bp = ess_ax.boxplot(
            data, positions=positions, widths=width * 0.9, patch_artist=True,
            manage_ticks=False, showfliers=True, showcaps=False,
            medianprops=dict(color="0.15", linewidth=0.9),
            whiskerprops=dict(color=color, linewidth=0.8),
            flierprops=dict(marker="o", markersize=2.4, markerfacecolor="white",
                            markeredgecolor=color, markeredgewidth=0.4, alpha=0.7),
        )
        for box in bp["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.6, linewidth=0.6)
    ess_ax.axhline(1.0, color=SEMANTIC_COLORS["reference"], lw=1.0, ls="--")
    ess_ax.set_xticks(range(len(ess_models)))
    ess_ax.set_xticklabels([label for _, label in ess_models])
    ess_ax.set_ylim(0.2, 1.05)
    ess_ax.set_xlabel("")
    ess_ax.set_ylabel("Kish ESS / source size", fontsize=12.2)
    ess_ax.set_title("Weighting keeps most of the data", fontsize=13.2, pad=8)
    trait_handles = [
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=TRAIT_COLORS[TRAIT_LABELS[t]],
               markeredgecolor="white", label=TRAIT_LABELS[t], markersize=7)
        for t in TRAIT_ORDER
    ]
    ess_ax.legend(handles=trait_handles, title="Trait", frameon=False, fontsize=9.4, title_fontsize=9.6, loc="lower left")
    _style_axes(ess_ax)

    pdf_path = output_dir / f"{file_stem}.pdf"
    png_path = output_dir / f"{file_stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def make_across_pop_diagnostic_figure(
    repo_root: Path | str = Path("."),
    output_dir: Path | str = Path("figures"),
) -> dict[str, object]:
    diagnostics = load_diagnostics(repo_root)
    summary = summarize_diagnostics(diagnostics)
    importance_transfer = paired_importance_transfer(diagnostics)
    pdf_path, png_path = plot_across_pop_diagnostics(diagnostics, output_dir=output_dir)
    return {
        "diagnostics": diagnostics,
        "summary": summary,
        "importance_transfer": importance_transfer,
        "figure": (pdf_path, png_path),
    }


if __name__ == "__main__":
    paths = make_across_pop_diagnostic_figure()
    print(paths["figure"])
