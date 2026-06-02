"""Plot thesis figures for E5 island-level Data Shapley results."""

from __future__ import annotations

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
    "Shapley order": "#2A9D8F",
    "Random individuals (E5)": "#4C78A8",
    "Random individuals (E3 fallback)": "#4C78A8",
    "Positive Shapley only": "#E45756",
    "Best Shapley prefix (post hoc)": "#F58518",
    "Full source pool": "0.35",
}

ISLAND_COORDINATES = {
    "Myken": (66.760027, 12.478567),
    "Lovund": (66.363291, 12.339011),
    "Sleneset": (66.360922, 12.595983),
    "Aldra": (66.408641, 13.059615),
    "Onøy og Lurøy": (66.406531, 12.863522),
    "Indre Kvarøy": (66.484892, 12.945714),
    "Nesøy": (66.574584, 12.646056),
    "Hestmannøy": (66.536102, 12.838656),
    "Gjerøy": (66.615165, 13.023725),
    "Træna": (66.498248, 12.082706),
    "Selvær": (66.586744, 12.237925),
    "Vega": (65.657315, 11.910286),
    "Leka": (65.083450, 11.634547),
    "Vikna": (64.905897, 11.023292),
    "Lauvøya": (63.928508, 9.933058),
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


def load_e5_shapley_results(root: Path, trait_order: list[str] | None = None) -> dict[str, pd.DataFrame]:
    trait_order = trait_order or TRAIT_ORDER
    filenames = {
        "summary": "shapley_island_summary_all_targets.csv",
        "repeats": "shapley_repeats_all_targets.csv",
        "add": "add_curve_repeats_all_targets.csv",
        "metadata": "tmc_metadata_all_repeats.csv",
        "rank": "tmc_intermediate_rankings_all_repeats.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, filename in filenames.items():
        frames = []
        for trait in trait_order:
            path = root / trait / filename
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "trait" not in df.columns:
                df["trait"] = trait
            frames.append(df)
        out[key] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    random_frames = []
    for trait in trait_order:
        path = root / trait / "random_baseline_repeats_all_targets.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "trait" not in df.columns:
            df["trait"] = trait
        random_frames.append(df)
    if random_frames:
        random_df = pd.concat(random_frames, ignore_index=True)
        out["add"] = pd.concat([out["add"], random_df], ignore_index=True)
        out["random_baseline"] = random_df
    else:
        out["random_baseline"] = pd.DataFrame()
    return out


def load_e3_random_fallback_results(root: Path, stem: str = "random_pc_ridge") -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame()
    frames = []
    for trait_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        path = trait_dir / f"{stem}_results.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "trait" not in df.columns:
            df["trait"] = trait_dir.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def island_order(summary_df: pd.DataFrame) -> tuple[list[int], list[str]]:
    codes = sorted(
        set(summary_df["target_island"].dropna().astype(int))
        | set(summary_df["source_island"].dropna().astype(int))
    )
    names: dict[int, str] = {}
    for row in summary_df[["target_island", "target_island_name"]].drop_duplicates().itertuples(index=False):
        names[int(row.target_island)] = str(row.target_island_name)
    for row in summary_df[["source_island", "source_island_name"]].drop_duplicates().itertuples(index=False):
        names[int(row.source_island)] = str(row.source_island_name)
    return codes, [names.get(code, str(code)) for code in codes]


def haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> np.ndarray:
    radius_km = 6371.0088
    lat1_rad = np.radians(lat1.astype(float).to_numpy())
    lon1_rad = np.radians(lon1.astype(float).to_numpy())
    lat2_rad = np.radians(lat2.astype(float).to_numpy())
    lon2_rad = np.radians(lon2.astype(float).to_numpy())
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_e5_geographic_distance(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    coord_df = (
        pd.DataFrame(
            [
                {"island_name": name, "lat": lat, "lon": lon}
                for name, (lat, lon) in ISLAND_COORDINATES.items()
            ]
        )
        .set_index("island_name")
        .sort_index()
    )
    out = out.join(coord_df.add_prefix("source_"), on="source_island_name")
    out = out.join(coord_df.add_prefix("target_"), on="target_island_name")
    missing = sorted(
        set(out.loc[out[["source_lat", "target_lat"]].isna().any(axis=1), "source_island_name"])
        | set(out.loc[out[["source_lat", "target_lat"]].isna().any(axis=1), "target_island_name"])
    )
    if missing:
        raise ValueError(f"Missing coordinates for island names: {missing}")
    out["geo_distance_km"] = haversine_km(out["source_lat"], out["source_lon"], out["target_lat"], out["target_lon"])
    out["phi_per_ind_1e4"] = out["phi_per_ind_mean"].astype(float) * 1e4
    out["shapley_sign"] = np.where(out["phi_per_ind_mean"].astype(float) >= 0.0, "Positive", "Negative")
    return out


def e5_full_corr_by_repeat(add_df: pd.DataFrame) -> pd.DataFrame:
    shap = add_df[add_df["method"].eq("shapley_mean")].copy()
    if shap.empty:
        return pd.DataFrame(columns=["trait", "target_island", "repeat", "corr_full", "n_full", "k_full"])
    idx = shap.groupby(["trait", "target_island", "repeat"], dropna=False)["n_islands"].idxmax()
    full = shap.loc[idx, ["trait", "target_island", "repeat", "corr_eval", "n_individuals", "n_islands"]].copy()
    return full.rename(columns={"corr_eval": "corr_full", "n_individuals": "n_full", "n_islands": "k_full"})


def add_e5_delta_columns(add_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["trait", "target_island", "repeat"]
    out = add_df.merge(e5_full_corr_by_repeat(add_df), on=keys, how="left")
    out["delta_full"] = out["corr_eval"] - out["corr_full"]
    out["method_label"] = (
        out["method"]
        .map(
            {
                "shapley_mean": "Shapley order",
                "random_individual": "Random individuals (E5)",
            }
        )
        .fillna(out["method"])
    )
    return out


def _random_fallback_groups(random_df: pd.DataFrame) -> dict[tuple[str, int, int], pd.DataFrame]:
    required = {"trait", "target_island", "repeat", "n_train_size", "pearson_r"}
    if random_df is None or random_df.empty or not required.issubset(random_df.columns):
        return {}
    groups = {}
    for key, group in random_df.groupby(["trait", "target_island", "repeat"], dropna=False):
        curve = (
            group.groupby("n_train_size", as_index=False)["pearson_r"]
            .mean()
            .sort_values("n_train_size")
        )
        if len(curve) >= 2:
            groups[(key[0], int(key[1]), int(key[2]))] = curve
    return groups


def interpolate_e3_random_to_e5_steps(add_df: pd.DataFrame, random_df: pd.DataFrame) -> pd.DataFrame:
    random_groups = _random_fallback_groups(random_df)
    if not random_groups:
        return pd.DataFrame()

    full_lookup = {
        (row.trait, int(row.target_island), int(row.repeat)): float(row.corr_full)
        for row in e5_full_corr_by_repeat(add_df).itertuples(index=False)
    }
    rows = []
    shap_steps = add_df[add_df["method"].eq("shapley_mean")].copy()
    for row in shap_steps.itertuples(index=False):
        key = (row.trait, int(row.target_island), int(row.repeat))
        curve = random_groups.get(key)
        if curve is None:
            continue
        x = curve["n_train_size"].to_numpy(dtype=float)
        y = curve["pearson_r"].to_numpy(dtype=float)
        n_ind = float(row.n_individuals)
        if n_ind < x.min() or n_ind > x.max():
            continue
        corr = float(np.interp(n_ind, x, y))
        corr_full = full_lookup.get(key, np.nan)
        rows.append(
            {
                "trait": row.trait,
                "target_island": int(row.target_island),
                "target_island_name": getattr(row, "target_island_name", ""),
                "repeat": int(row.repeat),
                "split_seed": getattr(row, "split_seed", np.nan),
                "n_islands": int(row.n_islands),
                "n_individuals": int(row.n_individuals),
                "corr_eval": corr,
                "mse_adj": np.nan,
                "method": "random_individual_e3_fallback",
                "order_seed": -1,
                "corr_full": corr_full,
                "delta_full": corr - corr_full,
                "method_label": "Random individuals (E3 fallback)",
            }
        )
    return pd.DataFrame(rows)


def prepare_e5_add_plot_df(e5: dict[str, pd.DataFrame], random_fallback_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    add_df = e5["add"].copy()
    base = add_e5_delta_columns(add_df)
    if base["method"].eq("random_individual").any():
        return base, "native_e5"
    fallback = interpolate_e3_random_to_e5_steps(add_df, random_fallback_df if random_fallback_df is not None else pd.DataFrame())
    if not fallback.empty:
        return pd.concat([base, fallback], ignore_index=True), "e3_fallback"
    return base, "none"


def prepare_e5_subset_rule_rows(e5: dict[str, pd.DataFrame], random_fallback_df: pd.DataFrame | None = None) -> pd.DataFrame:
    add_df = e5["add"].copy()
    repeats_df = e5["repeats"].copy()
    full_lookup = {
        (row.trait, int(row.target_island), int(row.repeat)): float(row.corr_full)
        for row in e5_full_corr_by_repeat(add_df).itertuples(index=False)
    }
    shap_curves = {
        key: group.sort_values("n_islands")
        for key, group in add_df[add_df["method"].eq("shapley_mean")].groupby(["trait", "target_island", "repeat"], dropna=False)
    }
    native_random = add_df[add_df["method"].eq("random_individual")].copy()
    has_native_random = not native_random.empty
    fallback_groups = {} if has_native_random else _random_fallback_groups(random_fallback_df if random_fallback_df is not None else pd.DataFrame())

    rows = []
    for key, phi_group in repeats_df.groupby(["trait", "target_island", "repeat"], dropna=False):
        trait, target, repeat = key[0], int(key[1]), int(key[2])
        curve = shap_curves.get((trait, target, repeat))
        if curve is None or curve.empty:
            continue
        target_name = str(phi_group["target_island_name"].iloc[0])
        corr_full = full_lookup.get((trait, target, repeat), np.nan)
        full_row = curve.loc[curve["n_islands"].idxmax()]
        rows.append(
            {
                "trait": trait,
                "target_island": target,
                "target_island_name": target_name,
                "repeat": repeat,
                "method_label": "Full source pool",
                "n_islands": int(full_row["n_islands"]),
                "n_individuals": int(full_row["n_individuals"]),
                "corr_eval": corr_full,
                "delta_full": 0.0,
            }
        )

        k_pos = int((phi_group["phi_per_individual"] > 0).sum())
        if k_pos <= 0:
            corr_pos = 0.0
            n_pos = 0
        else:
            pos_match = curve[curve["n_islands"].eq(k_pos)]
            if pos_match.empty:
                continue
            pos_row = pos_match.iloc[0]
            corr_pos = float(pos_row["corr_eval"])
            n_pos = int(pos_row["n_individuals"])
        rows.append(
            {
                "trait": trait,
                "target_island": target,
                "target_island_name": target_name,
                "repeat": repeat,
                "method_label": "Positive Shapley only",
                "n_islands": k_pos,
                "n_individuals": n_pos,
                "corr_eval": corr_pos,
                "delta_full": corr_pos - corr_full,
            }
        )

        best_row = curve.loc[curve["corr_eval"].idxmax()]
        rows.append(
            {
                "trait": trait,
                "target_island": target,
                "target_island_name": target_name,
                "repeat": repeat,
                "method_label": "Best Shapley prefix (post hoc)",
                "n_islands": int(best_row["n_islands"]),
                "n_individuals": int(best_row["n_individuals"]),
                "corr_eval": float(best_row["corr_eval"]),
                "delta_full": float(best_row["corr_eval"] - corr_full),
            }
        )

        if has_native_random and k_pos > 0:
            rand_rows = native_random[
                native_random["trait"].eq(trait)
                & native_random["target_island"].eq(target)
                & native_random["repeat"].eq(repeat)
                & native_random["n_islands"].eq(k_pos)
            ]
            if not rand_rows.empty:
                rand_corr = float(rand_rows["corr_eval"].mean())
                rows.append(
                    {
                        "trait": trait,
                        "target_island": target,
                        "target_island_name": target_name,
                        "repeat": repeat,
                        "method_label": "Random individuals (E5)",
                        "n_islands": k_pos,
                        "n_individuals": int(rand_rows["n_individuals"].median()),
                        "corr_eval": rand_corr,
                        "delta_full": rand_corr - corr_full,
                    }
                )
        elif k_pos > 0:
            rand_curve = fallback_groups.get((trait, target, repeat))
            if rand_curve is None:
                continue
            x = rand_curve["n_train_size"].to_numpy(dtype=float)
            y = rand_curve["pearson_r"].to_numpy(dtype=float)
            if n_pos < x.min() or n_pos > x.max():
                continue
            rand_corr = float(np.interp(float(n_pos), x, y))
            rows.append(
                {
                    "trait": trait,
                    "target_island": target,
                    "target_island_name": target_name,
                    "repeat": repeat,
                    "method_label": "Random individuals (E3 fallback)",
                    "n_islands": k_pos,
                    "n_individuals": n_pos,
                    "corr_eval": rand_corr,
                    "delta_full": rand_corr - corr_full,
                }
            )

    return pd.DataFrame(rows)


def plot_e5_shapley_heatmaps(e5: dict[str, pd.DataFrame], output_dir: Path, repo_root: Path) -> tuple[Path, Path] | None:
    summary_df = e5["summary"].copy()
    repeats_df = e5["repeats"].copy()
    if summary_df.empty:
        print("No E5 Shapley summary rows found.")
        return None

    import matplotlib.colors as mcolors

    trait_order = [trait for trait in TRAIT_ORDER if trait in set(summary_df["trait"])]
    order, names = island_order(summary_df)
    values = summary_df["phi_per_ind_mean"].to_numpy(dtype=float)
    vmax = float(np.nanquantile(np.abs(values), 0.98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(values)))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    sign_stability = pd.DataFrame()
    if not repeats_df.empty:
        sign_stability = (
            repeats_df.groupby(["trait", "target_island", "source_island"], as_index=False)
            .agg(pos_frac=("phi_per_individual", lambda s: float((s > 0).mean())))
        )
        sign_stability["sign_agreement"] = np.maximum(sign_stability["pos_frac"], 1.0 - sign_stability["pos_frac"])

    fig, axes = plt.subplots(1, len(trait_order), figsize=(4.6 * len(trait_order), 5.1), squeeze=False)
    axes = axes.ravel()
    for i, (ax, trait) in enumerate(zip(axes, trait_order)):
        sub = summary_df[summary_df["trait"].eq(trait)]
        mat = (
            sub.pivot(index="source_island", columns="target_island", values="phi_per_ind_mean")
            .reindex(index=order, columns=order)
        )
        sns.heatmap(
            mat,
            ax=ax,
            cmap="vlag",
            norm=norm,
            mask=mat.isna(),
            cbar=i == len(trait_order) - 1,
            cbar_kws={"label": "Mean Shapley value per individual"},
            linewidths=0.25,
            linecolor="white",
            square=True,
        )
        if not sign_stability.empty:
            stable = sign_stability[
                sign_stability["trait"].eq(trait) & sign_stability["sign_agreement"].ge(0.8)
            ]
            lookup = {code: idx for idx, code in enumerate(order)}
            x_pos = []
            y_pos = []
            for row in stable.itertuples(index=False):
                if int(row.target_island) in lookup and int(row.source_island) in lookup:
                    x_pos.append(lookup[int(row.target_island)] + 0.5)
                    y_pos.append(lookup[int(row.source_island)] + 0.5)
            ax.scatter(x_pos, y_pos, s=8, c="black", marker="o", linewidths=0, alpha=0.85)

        ax.set_title(TRAIT_LABELS.get(trait, trait))
        ax.set_xlabel("Target island")
        ax.set_ylabel("Source island" if i == 0 else "")
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.2)
        ax.set_yticklabels(names, rotation=0, fontsize=7.2)

    fig.text(0.01, 0.01, "Black dots mark source-target pairs with the same Shapley sign in at least 8 of 10 target splits.", fontsize=8)
    fig.subplots_adjust(wspace=0.18, bottom=0.24)
    pdf_path = save_figure(fig, output_dir / "e5_shapley_phi_heatmaps.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / "e5_shapley_phi_heatmaps.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def plot_e5_shapley_geographic_distance(
    e5: dict[str, pd.DataFrame],
    output_dir: Path,
    repo_root: Path,
) -> tuple[Path, Path] | None:
    summary_df = e5["summary"].copy()
    if summary_df.empty:
        print("No E5 Shapley summary rows found.")
        return None

    geo_df = add_e5_geographic_distance(summary_df)
    trait_order = [trait for trait in TRAIT_ORDER if trait in set(geo_df["trait"])]
    if not trait_order:
        return None

    x_max = float(np.ceil(geo_df["geo_distance_km"].max() / 25.0) * 25.0)
    y_abs = float(np.nanquantile(np.abs(geo_df["phi_per_ind_1e4"]), 0.99))
    if not np.isfinite(y_abs) or y_abs <= 0:
        y_abs = float(np.nanmax(np.abs(geo_df["phi_per_ind_1e4"])))
    y_lim = max(1.0, float(np.ceil(y_abs * 1.1)))

    colors = {"Positive": "#2A9D8F", "Negative": "#E45756"}
    fig, axes = plt.subplots(1, len(trait_order), figsize=(4.25 * len(trait_order), 3.75), sharex=True, sharey=True)
    if len(trait_order) == 1:
        axes = np.array([axes])

    handles_by_label = {}
    for ax, trait in zip(axes, trait_order):
        sub = geo_df[geo_df["trait"].eq(trait)].copy()
        for sign in ["Positive", "Negative"]:
            grp = sub[sub["shapley_sign"].eq(sign)]
            if grp.empty:
                continue
            handle = ax.scatter(
                grp["geo_distance_km"],
                grp["phi_per_ind_1e4"],
                s=22,
                color=colors[sign],
                alpha=0.62,
                edgecolor="white",
                linewidth=0.35,
                label=sign,
            )
            handles_by_label.setdefault(sign, handle)

        x = sub["geo_distance_km"].to_numpy(dtype=float)
        y = sub["phi_per_ind_1e4"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() >= 3 and np.nanstd(x[finite]) > 0:
            slope, intercept = np.polyfit(x[finite], y[finite], deg=1)
            x_line = np.linspace(0, x_max, 100)
            ax.plot(x_line, intercept + slope * x_line, color="0.15", linewidth=1.25, linestyle="-")
            pearson_r = pd.Series(x[finite]).corr(pd.Series(y[finite]))
            spearman_r = pd.Series(x[finite]).corr(pd.Series(y[finite]), method="spearman")
            ax.text(
                0.03,
                0.96,
                f"r = {pearson_r:.2f}\nρ = {spearman_r:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.88},
            )

        ax.axhline(0, color="0.35", linewidth=0.85, linestyle="--")
        ax.set_title(TRAIT_LABELS.get(trait, trait))
        ax.set_xlabel("Geographic distance between island centers (km)")
        ax.set_xlim(0, x_max)
        ax.set_ylim(-y_lim, y_lim)
        style_axes(ax)

    axes[0].set_ylabel(r"Mean Shapley value per individual ($\times 10^{-4}$)")
    labels = [label for label in ["Positive", "Negative"] if label in handles_by_label]
    fig.legend(
        [handles_by_label[label] for label in labels],
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(wspace=0.12, bottom=0.23)

    csv_path = repo_root / "outputs" / "final_results" / "e5_shapley_islands_pc_ridge" / "e5_shapley_geographic_distance.csv"
    try:
        geo_df.to_csv(csv_path, index=False)
    except PermissionError as exc:
        fallback = repo_root / "figures" / csv_path.name
        geo_df.to_csv(fallback, index=False)
        print(f"Could not write {csv_path} ({exc}). Wrote {fallback} instead.")

    pdf_path = save_figure(fig, output_dir / "e5_shapley_geographic_distance.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / "e5_shapley_geographic_distance.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def plot_e5_add_curves(
    e5: dict[str, pd.DataFrame],
    random_fallback_df: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
) -> tuple[Path, Path, str] | None:
    plot_df, random_source = prepare_e5_add_plot_df(e5, random_fallback_df=random_fallback_df)
    rules_df = prepare_e5_subset_rule_rows(e5, random_fallback_df=random_fallback_df)
    trait_order = [trait for trait in TRAIT_ORDER if trait in set(plot_df["trait"])]
    curve_methods = ["Shapley order", "Random individuals (E5)", "Random individuals (E3 fallback)"]

    fig, axes = plt.subplots(1, len(trait_order), figsize=(4.2 * len(trait_order), 3.8), sharey=True, squeeze=False)
    axes = axes.ravel()
    handles_by_label = {}
    rng = np.random.default_rng(14)
    for ax, trait in zip(axes, trait_order):
        sub = plot_df[plot_df["trait"].eq(trait)]
        for method_label in curve_methods:
            grp = sub[sub["method_label"].eq(method_label)]
            if grp.empty:
                continue
            agg = (
                grp.groupby("n_islands", as_index=False)
                .agg(delta_mean=("delta_full", "mean"), delta_std=("delta_full", "std"), n=("delta_full", "size"))
                .sort_values("n_islands")
            )
            agg["se"] = agg["delta_std"].fillna(0.0) / np.sqrt(agg["n"].clip(lower=1))
            color = METHOD_COLORS.get(method_label, "#888888")
            line = ax.plot(
                agg["n_islands"],
                agg["delta_mean"],
                marker="o" if method_label == "Shapley order" else None,
                markersize=3.8,
                linewidth=1.6,
                linestyle="-" if method_label == "Shapley order" else "--",
                color=color,
                label=method_label,
            )[0]
            handles_by_label.setdefault(method_label, line)
            ax.fill_between(
                agg["n_islands"],
                agg["delta_mean"] - agg["se"],
                agg["delta_mean"] + agg["se"],
                color=color,
                alpha=0.14,
                linewidth=0,
            )

        pos = rules_df[rules_df["trait"].eq(trait) & rules_df["method_label"].eq("Positive Shapley only")].copy()
        if not pos.empty:
            ax.scatter(
                pos["n_islands"] + rng.normal(0.0, 0.045, len(pos)),
                pos["delta_full"],
                s=12,
                color=METHOD_COLORS["Positive Shapley only"],
                alpha=0.20,
                linewidths=0,
            )
            mean_k = float(pos["n_islands"].mean())
            mean_delta = float(pos["delta_full"].mean())
            se_delta = float(pos["delta_full"].std() / np.sqrt(len(pos))) if len(pos) > 1 else 0.0
            marker = ax.errorbar(
                mean_k,
                mean_delta,
                yerr=se_delta,
                fmt="D",
                color=METHOD_COLORS["Positive Shapley only"],
                markersize=5.0,
                capsize=2.5,
                label="Positive Shapley only",
            )
            handles_by_label.setdefault("Positive Shapley only", marker)

        ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle=":")
        ax.set_title(TRAIT_LABELS.get(trait, trait))
        ax.set_xlabel("Number of source islands added")
        ax.set_xticks(np.arange(1, 15, 2))
        ax.margins(x=0.03, y=0.12)
        style_axes(ax)

    axes[0].set_ylabel("Pearson r relative to all source islands")
    labels = [label for label in curve_methods + ["Positive Shapley only"] if label in handles_by_label]
    fig.legend(
        [handles_by_label[label] for label in labels],
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(4, len(labels)),
        frameon=False,
    )
    if random_source == "e3_fallback":
        fig.text(0.01, 0.01, "Random baseline is interpolated from the E3 random PC-ridge individual runs.", fontsize=8)
    fig.subplots_adjust(bottom=0.25, wspace=0.18)
    pdf_path = save_figure(fig, output_dir / "e5_shapley_add_curves.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / "e5_shapley_add_curves.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path, random_source


def plot_e5_subset_rule_summary(
    e5: dict[str, pd.DataFrame],
    random_fallback_df: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
) -> tuple[Path, Path] | None:
    rules_df = prepare_e5_subset_rule_rows(e5, random_fallback_df=random_fallback_df)
    if rules_df.empty:
        print("No E5 subset rule rows available.")
        return None
    method_order = [
        "Random individuals (E5)",
        "Random individuals (E3 fallback)",
        "Positive Shapley only",
        "Best Shapley prefix (post hoc)",
    ]
    present_methods = [m for m in method_order if m in set(rules_df["method_label"])]
    trait_order = [trait for trait in TRAIT_ORDER if trait in set(rules_df["trait"])]
    fig, axes = plt.subplots(1, len(trait_order), figsize=(3.6 * len(trait_order), 3.7), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, trait in zip(axes, trait_order):
        sub = rules_df[rules_df["trait"].eq(trait) & rules_df["method_label"].isin(present_methods)]
        agg = (
            sub.groupby("method_label", as_index=False)
            .agg(delta_mean=("delta_full", "mean"), delta_std=("delta_full", "std"), n=("delta_full", "size"))
        )
        agg["se"] = agg["delta_std"].fillna(0.0) / np.sqrt(agg["n"].clip(lower=1))
        agg["method_label"] = pd.Categorical(agg["method_label"], categories=present_methods, ordered=True)
        agg = agg.sort_values("method_label")
        x = np.arange(len(agg))
        colors = [METHOD_COLORS.get(label, "#888888") for label in agg["method_label"].astype(str)]
        ax.bar(x, agg["delta_mean"], yerr=agg["se"], capsize=2.8, color=colors, alpha=0.86, edgecolor="white")
        ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle=":")
        labels = [
            str(label)
            .replace("Random individuals (E3 fallback)", "Random\n(E3)")
            .replace("Random individuals (E5)", "Random\n(E5)")
            .replace("Positive Shapley only", "Positive\nShapley")
            .replace("Best Shapley prefix (post hoc)", "Best prefix\n(post hoc)")
            for label in agg["method_label"].astype(str)
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_title(TRAIT_LABELS.get(trait, trait))
        ax.margins(y=0.16)
        style_axes(ax)
    axes[0].set_ylabel("Pearson r relative to all source islands")
    fig.subplots_adjust(bottom=0.24, wspace=0.18)
    pdf_path = save_figure(fig, output_dir / "e5_shapley_subset_rules.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / "e5_shapley_subset_rules.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def compute_e5_shapley_random_advantage(
    e5: dict[str, pd.DataFrame],
    random_fallback_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    plot_df, _random_source = prepare_e5_add_plot_df(e5, random_fallback_df=random_fallback_df)
    shap = plot_df[plot_df["method_label"].eq("Shapley order")].copy()
    random = plot_df[
        plot_df["method_label"].isin(["Random individuals (E5)", "Random individuals (E3 fallback)"])
    ].copy()
    if shap.empty or random.empty:
        return pd.DataFrame()

    keys = ["trait", "target_island", "repeat", "n_islands"]
    shap = (
        shap.groupby(keys, as_index=False)
        .agg(shap_delta=("delta_full", "mean"), shap_corr=("corr_eval", "mean"))
    )
    random = (
        random.groupby(keys, as_index=False)
        .agg(random_delta=("delta_full", "mean"), random_corr=("corr_eval", "mean"))
    )
    out = shap.merge(random, on=keys, how="inner")
    out["advantage"] = out["shap_delta"] - out["random_delta"]
    out["corr_advantage"] = out["shap_corr"] - out["random_corr"]
    return out


def plot_e5_discussion_worked(
    e5: dict[str, pd.DataFrame],
    random_fallback_df: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
) -> tuple[Path, Path] | None:
    advantage_df = compute_e5_shapley_random_advantage(e5, random_fallback_df=random_fallback_df)
    if advantage_df.empty:
        print("No Shapley-random advantage rows available.")
        return None

    trait_order = [trait for trait in TRAIT_ORDER if trait in set(advantage_df["trait"])]
    palette = sns.color_palette("colorblind", n_colors=len(trait_order))
    trait_colors = dict(zip(trait_order, palette))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.8, 3.8),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
        constrained_layout=True,
    )

    ax = axes[0]
    for trait in trait_order:
        sub = advantage_df[advantage_df["trait"].eq(trait)].copy()
        agg = (
            sub.groupby("n_islands", as_index=False)
            .agg(mean=("advantage", "mean"), std=("advantage", "std"), n=("advantage", "size"))
            .sort_values("n_islands")
        )
        agg["se"] = agg["std"].fillna(0.0) / np.sqrt(agg["n"].clip(lower=1))
        color = trait_colors[trait]
        ax.plot(
            agg["n_islands"],
            agg["mean"],
            marker="o",
            markersize=3.6,
            linewidth=1.6,
            color=color,
            label=TRAIT_LABELS.get(trait, trait),
        )
        ax.fill_between(
            agg["n_islands"],
            agg["mean"] - agg["se"],
            agg["mean"] + agg["se"],
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle=":")
    ax.set_title("Shapley prioritizes better than random")
    ax.set_xlabel("Number of source islands added")
    ax.set_ylabel("Pearson r gain over random baseline")
    ax.set_xticks(np.arange(1, 15, 2))
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)

    early = advantage_df[advantage_df["n_islands"].le(7)].copy()
    early = (
        early.groupby(["trait", "target_island", "repeat"], as_index=False)
        .agg(early_advantage=("advantage", "mean"))
    )
    ax = axes[1]
    summary = (
        early.groupby("trait", as_index=False)
        .agg(mean=("early_advantage", "mean"), std=("early_advantage", "std"), n=("early_advantage", "size"))
    )
    summary["se"] = summary["std"].fillna(0.0) / np.sqrt(summary["n"].clip(lower=1))
    summary["trait"] = pd.Categorical(summary["trait"], categories=trait_order, ordered=True)
    summary = summary.sort_values("trait")
    x = np.arange(len(summary))
    colors = [trait_colors[str(trait)] for trait in summary["trait"].astype(str)]
    ax.bar(x, summary["mean"], yerr=summary["se"], capsize=2.8, color=colors, alpha=0.88, edgecolor="white")
    rng = np.random.default_rng(14)
    for idx, trait in enumerate(summary["trait"].astype(str)):
        vals = early.loc[early["trait"].eq(trait), "early_advantage"].to_numpy(dtype=float)
        if len(vals) > 120:
            vals = rng.choice(vals, size=120, replace=False)
        ax.scatter(
            np.full(len(vals), idx) + rng.normal(0, 0.045, len(vals)),
            vals,
            s=8,
            color="0.15",
            alpha=0.16,
            linewidths=0,
        )
    ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([TRAIT_LABELS.get(str(t), str(t)) for t in summary["trait"]], rotation=20, ha="right")
    ax.set_ylabel("Mean gain for first seven islands")
    ax.set_title("Early-stage advantage")
    style_axes(ax)

    pdf_path = save_figure(fig, output_dir / "e5_shapley_discussion_worked.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / "e5_shapley_discussion_worked.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def compute_e5_repeat_rank_correlations(e5: dict[str, pd.DataFrame]) -> pd.DataFrame:
    repeats_df = e5["repeats"].copy()
    if repeats_df.empty:
        return pd.DataFrame()
    rows = []
    for (trait, target), group in repeats_df.groupby(["trait", "target_island"], sort=False):
        pivot = group.pivot(index="source_island", columns="repeat", values="phi_per_individual")
        repeats = list(pivot.columns)
        target_name = str(group["target_island_name"].iloc[0])
        for i, rep_i in enumerate(repeats):
            for rep_j in repeats[i + 1:]:
                corr = pivot[rep_i].corr(pivot[rep_j], method="spearman")
                if np.isfinite(corr):
                    rows.append(
                        {
                            "trait": trait,
                            "target_island": int(target),
                            "target_island_name": target_name,
                            "repeat_i": int(rep_i),
                            "repeat_j": int(rep_j),
                            "spearman_rho": float(corr),
                        }
                    )
    return pd.DataFrame(rows)


def plot_e5_validation_instability(
    e5: dict[str, pd.DataFrame],
    output_dir: Path,
    repo_root: Path,
) -> tuple[Path, Path] | None:
    metadata_df = e5["metadata"].copy()
    rank_corr_df = compute_e5_repeat_rank_correlations(e5)
    if metadata_df.empty or rank_corr_df.empty:
        print("No validation-instability rows available.")
        return None

    trait_order = [trait for trait in TRAIT_ORDER if trait in set(metadata_df["trait"])]
    palette = sns.color_palette("colorblind", n_colors=len(trait_order))
    trait_colors = dict(zip(trait_order, palette))

    fig = plt.figure(figsize=(12.0, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, len(trait_order), height_ratios=[1.25, 0.85])

    for col, trait in enumerate(trait_order):
        ax = fig.add_subplot(gs[0, col])
        sub = metadata_df[metadata_df["trait"].eq(trait)].copy()
        target_stats = (
            sub.groupby(["target_island", "target_island_name"], as_index=False)
            .agg(mean=("v_full", "mean"), std=("v_full", "std"))
            .sort_values("mean", ascending=False)
            .reset_index(drop=True)
        )
        order = target_stats["target_island"].astype(int).tolist()
        labels = target_stats["target_island_name"].astype(str).tolist()
        lookup = {target: idx for idx, target in enumerate(order)}
        rng = np.random.default_rng(100 + col)
        for row in sub.itertuples(index=False):
            x = lookup[int(row.target_island)] + rng.normal(0.0, 0.045)
            ax.scatter(x, float(row.v_full), s=12, color=trait_colors[trait], alpha=0.26, linewidths=0)
        ax.errorbar(
            np.arange(len(target_stats)),
            target_stats["mean"].to_numpy(dtype=float),
            yerr=target_stats["std"].fillna(0.0).to_numpy(dtype=float),
            fmt="o",
            color="0.15",
            markersize=3.4,
            capsize=2.2,
            linewidth=0.9,
        )
        ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle=":")
        ax.set_title(TRAIT_LABELS.get(trait, trait))
        ax.set_xlabel("Target island")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.1)
        if col == 0:
            ax.set_ylabel(r"Full-source utility $v_{\mathrm{full}}$")
        style_axes(ax)

    ax = fig.add_subplot(gs[1, :])
    rank_corr_df = rank_corr_df[rank_corr_df["trait"].isin(trait_order)].copy()
    rank_corr_df["trait_label"] = rank_corr_df["trait"].map(TRAIT_LABELS).fillna(rank_corr_df["trait"])
    label_order = [TRAIT_LABELS.get(trait, trait) for trait in trait_order]
    sns.violinplot(
        data=rank_corr_df,
        x="trait_label",
        y="spearman_rho",
        hue="trait_label",
        order=label_order,
        hue_order=label_order,
        palette=[trait_colors[trait] for trait in trait_order],
        inner="quartile",
        cut=0,
        linewidth=0.8,
        legend=False,
        ax=ax,
    )
    ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle=":")
    ax.set_xlabel("")
    ax.set_ylabel("Pairwise Spearman correlation")
    ax.set_title("Repeat-to-repeat agreement in Shapley island rankings")
    ax.set_ylim(-1.0, 1.0)
    style_axes(ax)

    pdf_path = save_figure(fig, output_dir / "e5_shapley_validation_instability.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / "e5_shapley_validation_instability.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def compute_e5_rank_convergence(rank_df: pd.DataFrame, top_k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rank_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    metrics = []
    overlaps = []
    for trait, sub in rank_df.groupby("trait", sort=False):
        max_perm = int(sub["permutation_index"].max())
        keys = ["target_island", "repeat", "source_island"]
        final = (
            sub[sub["permutation_index"].eq(max_perm)][keys + ["rank_running"]]
            .rename(columns={"rank_running": "rank_final"})
        )
        merged = sub.merge(final, on=keys, how="inner")
        merged["abs_rank_delta"] = (merged["rank_running"] - merged["rank_final"]).abs()
        by_perm = (
            merged.groupby("permutation_index", as_index=False)
            .agg(
                mean_abs_rank_delta=("abs_rank_delta", "mean"),
                median_abs_rank_delta=("abs_rank_delta", "median"),
                exact_rank_share=("abs_rank_delta", lambda s: float((s == 0).mean())),
            )
        )
        by_perm["trait"] = trait
        metrics.append(by_perm)

        final_sets = {}
        final_sub = sub[sub["permutation_index"].eq(max_perm)]
        for key, group in final_sub.groupby(["target_island", "repeat"], sort=False):
            final_sets[(int(key[0]), int(key[1]))] = set(group.loc[group["rank_running"].le(top_k), "source_island"].astype(int))
        for key, group in sub.groupby(["target_island", "repeat", "permutation_index"], sort=False):
            target, repeat, perm = int(key[0]), int(key[1]), int(key[2])
            final_top = final_sets.get((target, repeat), set())
            current_top = set(group.loc[group["rank_running"].le(top_k), "source_island"].astype(int))
            if final_top:
                overlaps.append(
                    {
                        "trait": trait,
                        "target_island": target,
                        "repeat": repeat,
                        "permutation_index": perm,
                        "top_overlap": len(current_top & final_top) / float(top_k),
                    }
                )

    metrics_df = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    overlap_df = pd.DataFrame(overlaps)
    if not overlap_df.empty:
        overlap_df = (
            overlap_df.groupby(["trait", "permutation_index"], as_index=False)
            .agg(top_overlap=("top_overlap", "mean"))
        )
    return metrics_df, overlap_df


def plot_e5_rank_convergence(e5: dict[str, pd.DataFrame], output_dir: Path, repo_root: Path) -> tuple[Path, Path] | None:
    metrics_df, overlap_df = compute_e5_rank_convergence(e5["rank"].copy(), top_k=3)
    if metrics_df.empty:
        print("No E5 rank convergence rows found.")
        return None
    trait_order = [trait for trait in TRAIT_ORDER if trait in set(metrics_df["trait"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), constrained_layout=True)
    palette = sns.color_palette("colorblind", n_colors=len(trait_order))
    for idx, trait in enumerate(trait_order):
        label = TRAIT_LABELS.get(trait, trait)
        color = palette[idx]
        sub = metrics_df[metrics_df["trait"].eq(trait)].sort_values("permutation_index")
        axes[0].plot(sub["permutation_index"], sub["mean_abs_rank_delta"], color=color, label=label, linewidth=1.6)
        if not overlap_df.empty:
            ov = overlap_df[overlap_df["trait"].eq(trait)].sort_values("permutation_index")
            axes[1].plot(ov["permutation_index"], ov["top_overlap"], color=color, label=label, linewidth=1.6)
    axes[0].set_title("Rank convergence")
    axes[0].set_xlabel("TMC permutations")
    axes[0].set_ylabel("Mean absolute rank change\nfrom final ranking")
    axes[1].set_title("Top-3 stability")
    axes[1].set_xlabel("TMC permutations")
    axes[1].set_ylabel("Overlap with final top 3")
    axes[1].set_ylim(0.6, 1.01)
    for ax in axes:
        style_axes(ax)
    axes[1].legend(frameon=False, loc="lower right")
    pdf_path = save_figure(fig, output_dir / "e5_shapley_rank_convergence.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / "e5_shapley_rank_convergence.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def plot_e5_intermediate_rankings(e5: dict[str, pd.DataFrame], output_dir: Path, repo_root: Path) -> tuple[Path, Path] | None:
    rank_df = e5["rank"].copy()
    if rank_df.empty:
        print("No E5 intermediate ranking rows found.")
        return None
    metadata_df = e5.get("metadata", pd.DataFrame()).copy()
    trait_order = [trait for trait in TRAIT_ORDER if trait in set(rank_df["trait"])]
    fig, axes = plt.subplots(1, len(trait_order), figsize=(4.0 * len(trait_order), 4.0), sharey=True, squeeze=False)
    axes = axes.ravel()
    palette = sns.color_palette("tab10", n_colors=5)

    for ax, trait in zip(axes, trait_order):
        target_code = None
        target_name = ""
        if not metadata_df.empty and "v_full" in metadata_df.columns:
            target_perf = (
                metadata_df[metadata_df["trait"].eq(trait)]
                .groupby(["target_island", "target_island_name"], as_index=False)["v_full"]
                .mean()
                .sort_values("v_full", ascending=False)
            )
            if not target_perf.empty:
                target_code = int(target_perf.iloc[0]["target_island"])
                target_name = str(target_perf.iloc[0]["target_island_name"])
        if target_code is None:
            target_code = int(rank_df.loc[rank_df["trait"].eq(trait), "target_island"].dropna().astype(int).iloc[0])
        sub = rank_df[rank_df["trait"].eq(trait) & rank_df["target_island"].eq(target_code)].copy()
        if not target_name and "target_island_name" in sub.columns and not sub.empty:
            target_name = str(sub["target_island_name"].iloc[0])
        max_perm = int(sub["permutation_index"].max())
        final_mean = (
            sub[sub["permutation_index"].eq(max_perm)]
            .groupby(["source_island", "source_island_name"], as_index=False)["rank_running"]
            .mean()
            .sort_values("rank_running")
        )
        top_sources = final_mean.head(5)["source_island"].astype(int).tolist()
        name_lookup = final_mean.set_index("source_island")["source_island_name"].to_dict()
        traj = (
            sub.groupby(["source_island", "permutation_index"], as_index=False)["rank_running"]
            .mean()
            .sort_values(["source_island", "permutation_index"])
        )
        for source, group in traj.groupby("source_island", sort=False):
            if int(source) in top_sources:
                continue
            ax.plot(group["permutation_index"], group["rank_running"], color="0.82", linewidth=0.75, alpha=0.65)
        for idx, source in enumerate(top_sources):
            group = traj[traj["source_island"].eq(source)]
            ax.plot(
                group["permutation_index"],
                group["rank_running"],
                color=palette[idx],
                linewidth=1.65,
                marker="o",
                markersize=2.5,
                markevery=5,
                label=str(name_lookup.get(source, source)),
            )
        ax.invert_yaxis()
        ax.set_title(f"{TRAIT_LABELS.get(trait, trait)}\nTarget: {target_name}")
        ax.set_xlabel("TMC permutations")
        ax.set_yticks(np.arange(1, 15, 1))
        ax.legend(frameon=False, fontsize=7, loc="lower left")
        style_axes(ax)
    axes[0].set_ylabel("Mean running rank\nacross target splits")
    fig.subplots_adjust(wspace=0.12, bottom=0.16)
    pdf_path = save_figure(fig, output_dir / "e5_shapley_intermediate_rankings.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / "e5_shapley_intermediate_rankings.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def make_e5_shapley_figures(
    repo_root: Path | None = None,
    output_dir: Path | None = None,
    e5_root: Path | None = None,
    random_fallback_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = find_repo_root() if repo_root is None else Path(repo_root)
    output_dir = repo_root / "figures" if output_dir is None else Path(output_dir)
    final_results_dir = repo_root / "outputs" / "final_results"
    e5_root = final_results_dir / "e5_shapley_islands_pc_ridge" if e5_root is None else Path(e5_root)
    random_fallback_root = final_results_dir / "e3_random_pc_ridge" if random_fallback_root is None else Path(random_fallback_root)

    configure_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    e5 = load_e5_shapley_results(e5_root)
    print({key: value.shape for key, value in e5.items()})
    if e5.get("random_baseline", pd.DataFrame()).empty:
        random_fallback = load_e3_random_fallback_results(random_fallback_root)
        print(f"Loaded {len(random_fallback)} E3 random fallback rows")
    else:
        random_fallback = pd.DataFrame()
        print(f"Loaded {len(e5['random_baseline'])} native E5 random-baseline rows")

    paths: dict[str, Any] = {}
    paths["heatmaps"] = plot_e5_shapley_heatmaps(e5, output_dir, repo_root)
    paths["geographic_distance"] = plot_e5_shapley_geographic_distance(e5, output_dir, repo_root)
    paths["add_curves"] = plot_e5_add_curves(e5, random_fallback, output_dir, repo_root)
    paths["subset_rules"] = plot_e5_subset_rule_summary(e5, random_fallback, output_dir, repo_root)
    paths["discussion_worked"] = plot_e5_discussion_worked(e5, random_fallback, output_dir, repo_root)
    paths["validation_instability"] = plot_e5_validation_instability(e5, output_dir, repo_root)
    paths["rank_convergence"] = plot_e5_rank_convergence(e5, output_dir, repo_root)
    paths["intermediate_rankings"] = plot_e5_intermediate_rankings(e5, output_dir, repo_root)
    return paths


if __name__ == "__main__":
    result_paths = make_e5_shapley_figures()
    for key, value in result_paths.items():
        print(key, value)
