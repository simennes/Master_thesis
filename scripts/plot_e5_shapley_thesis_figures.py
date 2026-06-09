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
FULL_WIDTH = 6.7
SOUTHERN_TARGET_ISLANDS = {11, 12, 13, 14}

METHOD_COLORS = {
    "Random individuals (E5)": "#8F8F8F",
    "Random individuals (E3 fallback)": "#8F8F8F",
    "Shapley order": "#3F8F5B",
    "Positive Shapley only": "#E45756",
    "Best Shapley prefix (post hoc)": "#D55E00",
    "Full source pool": "0.35",
}

ADD_CURVE_METHOD_ORDER = [
    "Random individuals (E5)",
    "Random individuals (E3 fallback)",
    "Shapley order",
]

METHOD_DISPLAY_LABELS = {
    "Shapley order": "Shapley order",
    "Random individuals (E5)": "Random individuals\n(size-matched)",
    "Random individuals (E3 fallback)": "Random individuals\n(E3 fallback)",
    "Positive Shapley only": "Positive Shapley\nonly",
    "All source islands": "All source islands",
}

REGION_COLORS = {
    "Northern islands": "#4C78A8",
    "Southern islands": "#D55E00",
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


def load_e4_avggrm_results(root: Path, trait_order: list[str] | None = None) -> pd.DataFrame:
    root = Path(root)
    if not root.exists():
        return pd.DataFrame()
    frames = []
    for trait in (trait_order or TRAIT_ORDER):
        path = root / trait / "e4_similarity_pc_ridge_results.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "trait" not in df.columns:
            df["trait"] = trait
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_e1_full_source_baseline(root: Path) -> pd.DataFrame:
    path = Path(root) / "e1_pc_ridge_fixed_loio_l10_per_fold_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"trait", "test_island_code", "pearson_r"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return (
        df.groupby(["trait", "test_island_code"], as_index=False)["pearson_r"]
        .mean()
        .rename(columns={"test_island_code": "target_island", "pearson_r": "full_source_r"})
    )


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


# Island ecological/geographic type, keyed by internal experiment code (0..14).
# Outer (non-farm) and inner (farm) split the northern Helgeland islands; the
# four southern islands form their own group.
ISLAND_TYPE_BY_CODE = {
    0: "inner", 1: "outer", 2: "outer", 3: "outer", 4: "inner", 5: "inner",
    6: "inner", 7: "inner", 8: "outer", 9: "outer", 10: "inner",
    11: "south", 12: "south", 13: "south", 14: "south",
}
ISLAND_TYPE_LABEL = {"outer": "Outer (non-farm)", "inner": "Inner (farm)", "south": "Southern"}
ISLAND_TYPE_ORDER = ["outer", "inner", "south"]


def type_grouped_order(codes: list[int]) -> list[int]:
    """Order island codes by type (outer, inner, south), then by code within type."""
    return sorted(codes, key=lambda c: (ISLAND_TYPE_ORDER.index(ISLAND_TYPE_BY_CODE.get(int(c), "south")), int(c)))


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


def _method_legend_handles(method_order: list[str]) -> list[plt.Line2D]:
    handles = []
    for method in method_order:
        color = METHOD_COLORS.get(method, "#888888")
        label = METHOD_DISPLAY_LABELS.get(method, method)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markersize=7,
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.62,
                label=label,
            )
        )
    return handles


def _draw_grouped_k_boxplots(
    ax: plt.Axes,
    df: pd.DataFrame,
    method_order: list[str],
    k_values: list[int],
    *,
    showfliers: bool,
    linewidth: float = 0.65,
) -> None:
    if df.empty or not method_order or not k_values:
        return
    n_methods = len(method_order)
    width = min(0.78 / max(n_methods, 1), 0.26)
    flierprops = dict(
        marker="o",
        markersize=2.2 if showfliers else 0,
        markerfacecolor="white",
        markeredgewidth=0.45,
        alpha=0.75,
    )
    for j, method in enumerate(method_order):
        color = METHOD_COLORS.get(method, "#888888")
        offset = (j - (n_methods - 1) / 2) * width
        data, positions = [], []
        for k in k_values:
            vals = df.loc[
                df["method_label"].eq(method) & df["n_islands"].eq(k),
                "corr_eval",
            ].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            data.append(vals)
            positions.append(k + offset)
        if not data:
            continue
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.86,
            patch_artist=True,
            manage_ticks=False,
            showfliers=showfliers,
            showcaps=False,
            medianprops=dict(color="0.12", linewidth=linewidth + 0.1),
            whiskerprops=dict(color=color, linewidth=linewidth),
            flierprops={**flierprops, "markeredgecolor": color},
        )
        for box in bp["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.55, linewidth=linewidth)


def _available_add_curve_methods(plot_df: pd.DataFrame, rules_df: pd.DataFrame) -> list[str]:
    available = set(plot_df["method_label"].dropna())
    return [method for method in ADD_CURVE_METHOD_ORDER if method in available]


def plot_e5_shapley_heatmaps(e5: dict[str, pd.DataFrame], output_dir: Path, repo_root: Path) -> tuple[Path, Path] | None:
    summary_df = e5["summary"].copy()
    if summary_df.empty:
        print("No E5 Shapley summary rows found.")
        return None

    import matplotlib.colors as mcolors

    trait_order = [trait for trait in TRAIT_ORDER if trait in set(summary_df["trait"])]
    base_order, base_names = island_order(summary_df)
    name_by_code = dict(zip(base_order, base_names))
    order = type_grouped_order(base_order)
    names = [name_by_code[c] for c in order]
    mean_df = (
        summary_df[summary_df["trait"].isin(trait_order)]
        .groupby(["target_island", "source_island"], as_index=False)
        .agg(phi_per_ind_mean=("phi_per_ind_mean", "mean"), n_traits=("trait", "nunique"))
    )
    sign_df = (
        summary_df[summary_df["trait"].isin(trait_order)]
        .assign(is_positive=lambda df: df["phi_per_ind_mean"].astype(float).ge(0.0))
        .groupby(["target_island", "source_island"], as_index=False)
        .agg(n_traits=("trait", "nunique"), n_positive=("is_positive", "sum"))
    )
    sign_df["same_sign_all_traits"] = (
        sign_df["n_traits"].eq(len(trait_order))
        & (sign_df["n_positive"].eq(0) | sign_df["n_positive"].eq(sign_df["n_traits"]))
    )

    values = mean_df["phi_per_ind_mean"].to_numpy(dtype=float)
    vmax = float(np.nanquantile(np.abs(values), 0.98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(values)))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    mat = (
        mean_df.pivot(index="source_island", columns="target_island", values="phi_per_ind_mean")
        .reindex(index=order, columns=order)
    )
    fig, ax = plt.subplots(figsize=(7.4, 7.1))
    sns.heatmap(
        mat,
        ax=ax,
        cmap="vlag",
        norm=norm,
        mask=mat.isna(),
        cbar=False,
        linewidths=0.35,
        linecolor="white",
        square=True,
    )
    stable = sign_df[sign_df["same_sign_all_traits"]]
    lookup = {code: idx for idx, code in enumerate(order)}
    x_pos = []
    y_pos = []
    for row in stable.itertuples(index=False):
        if int(row.target_island) in lookup and int(row.source_island) in lookup:
            x_pos.append(lookup[int(row.target_island)] + 0.5)
            y_pos.append(lookup[int(row.source_island)] + 0.5)
    ax.scatter(x_pos, y_pos, s=12, c="black", marker="o", linewidths=0, alpha=0.88)

    # Outline the diagonal block of each island type (outer / inner / south).
    from matplotlib.patches import Rectangle
    types_in_order = [ISLAND_TYPE_BY_CODE.get(int(c), "south") for c in order]
    for t in ISLAND_TYPE_ORDER:
        idxs = [i for i, tt in enumerate(types_in_order) if tt == t]
        if not idxs:
            continue
        s0, n = min(idxs), max(idxs) - min(idxs) + 1
        ax.add_patch(Rectangle((s0, s0), n, n, fill=False, edgecolor="black",
                               linewidth=1.8, zorder=10))

    ax.set_xlabel("Target island", fontsize=11)
    ax.set_ylabel("Source island", fontsize=11)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9.3)
    ax.set_yticklabels(names, rotation=0, fontsize=9.3)
    fig.subplots_adjust(bottom=0.18, right=0.88, top=0.98)
    # Keep the colorbar separate from the square heatmap geometry.
    sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("Mean Shapley value per individual\n(averaged over traits)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    pdf_path = save_figure(fig, output_dir / "e5_shapley_phi_heatmaps.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / "e5_shapley_phi_heatmaps.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def write_island_type_shapley_table(e5: dict[str, pd.DataFrame], output_dir: Path,
                                    stem: str = "shapley_island_type_table") -> Path | None:
    """LaTeX table of mean per-individual Shapley by source and target island type."""
    summary_df = e5["summary"].copy()
    if summary_df.empty:
        return None
    trait_order = [t for t in TRAIT_ORDER if t in set(summary_df["trait"])]
    d = summary_df[summary_df["trait"].isin(trait_order)].copy()
    d["src_type"] = d["source_island"].astype(int).map(ISLAND_TYPE_BY_CODE)
    d["tgt_type"] = d["target_island"].astype(int).map(ISLAND_TYPE_BY_CODE)
    cell = d.groupby(["src_type", "tgt_type"])["phi_per_ind_mean"].mean() * 1e4
    marg = d.groupby("src_type")["phi_per_ind_mean"].mean() * 1e4

    def fmt(v):
        return f"{v:.2f}" if np.isfinite(v) else "--"

    rows = []
    for st in ISLAND_TYPE_ORDER:
        vals = " & ".join(fmt(cell.get((st, tt), float("nan"))) for tt in ISLAND_TYPE_ORDER)
        rows.append(f"{ISLAND_TYPE_LABEL[st]} & {vals} & {fmt(marg.get(st, float('nan')))} \\\\")

    table = "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Target island type}} & \\",
        r"\cmidrule(lr){2-4}",
        r"\textbf{Source island type} & \textbf{Outer} & \textbf{Inner} & \textbf{Southern} & \textbf{All} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption[Shapley value by island type]{Mean per-individual Shapley value "
         r"($\times 10^{-4}$), averaged over the three traits, grouped by source and target "
         r"island type. Outer (non-farm) and inner (farm) split the northern Helgeland islands; "
         r"the four southern islands form a separate group. The final column averages over all "
         r"target islands.}"),
        r"\label{tab:shapley_island_type}",
        r"\end{table}",
        "",
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.tex"
    path.write_text(table, encoding="utf-8")
    return path


def _plot_e5_shapley_geographic_distance_by_trait(
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

    colors = {"Positive": "#4C78A8", "Negative": "#E45756"}  # palette accent pair (blue / red)
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


def plot_e5_shapley_geographic_distance_average(
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

    avg_df = (
        geo_df[geo_df["trait"].isin(trait_order)]
        .groupby(
            [
                "source_island",
                "target_island",
                "source_island_name",
                "target_island_name",
                "geo_distance_km",
            ],
            as_index=False,
        )
        .agg(phi_per_ind_mean=("phi_per_ind_mean", "mean"), n_traits=("trait", "nunique"))
    )
    avg_df["phi_per_ind_1e4"] = avg_df["phi_per_ind_mean"].astype(float) * 1e4
    avg_df["shapley_sign"] = np.where(avg_df["phi_per_ind_mean"].astype(float).ge(0.0), "Positive", "Negative")

    x_max = float(np.ceil(avg_df["geo_distance_km"].max() / 25.0) * 25.0)
    y_abs = float(np.nanquantile(np.abs(avg_df["phi_per_ind_1e4"]), 0.99))
    if not np.isfinite(y_abs) or y_abs <= 0:
        y_abs = float(np.nanmax(np.abs(avg_df["phi_per_ind_1e4"])))
    y_lim = max(1.0, float(np.ceil(y_abs * 1.1)))

    colors = {"Positive": "#4C78A8", "Negative": "#E45756"}
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.86, 4.15))
    handles_by_label = {}
    for sign in ["Positive", "Negative"]:
        grp = avg_df[avg_df["shapley_sign"].eq(sign)]
        if grp.empty:
            continue
        handle = ax.scatter(
            grp["geo_distance_km"],
            grp["phi_per_ind_1e4"],
            s=28,
            color=colors[sign],
            alpha=0.68,
            edgecolor="white",
            linewidth=0.38,
            label=sign,
        )
        handles_by_label[sign] = handle

    x = avg_df["geo_distance_km"].to_numpy(dtype=float)
    y = avg_df["phi_per_ind_1e4"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 3 and np.nanstd(x[finite]) > 0:
        slope, intercept = np.polyfit(x[finite], y[finite], deg=1)
        x_line = np.linspace(0, x_max, 100)
        ax.plot(x_line, intercept + slope * x_line, color="0.15", linewidth=1.35, linestyle="-")
        pearson_r = pd.Series(x[finite]).corr(pd.Series(y[finite]))
        spearman_r = pd.Series(x[finite]).corr(pd.Series(y[finite]), method="spearman")
        ax.text(
            0.03,
            0.96,
            fr"$r={pearson_r:.2f}$" "\n" fr"$\rho={spearman_r:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.9},
        )

    ax.axhline(0, color="0.35", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Geographic distance between island centers (km)")
    ax.set_ylabel(r"Mean Shapley value per individual ($\times 10^{-4}$)")
    ax.set_xlim(0, x_max)
    ax.set_ylim(-y_lim, y_lim)
    labels = [label for label in ["Positive", "Negative"] if label in handles_by_label]
    ax.legend([handles_by_label[label] for label in labels], labels, loc="lower left", frameon=False)
    style_axes(ax)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.96)

    csv_path = repo_root / "outputs" / "final_results" / "e5_shapley_islands_pc_ridge" / "e5_shapley_geographic_distance_trait_average.csv"
    try:
        avg_df.to_csv(csv_path, index=False)
    except PermissionError as exc:
        fallback = repo_root / "figures" / csv_path.name
        avg_df.to_csv(fallback, index=False)
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
    method_order = _available_add_curve_methods(plot_df, rules_df)
    if not method_order:
        print("No E5 add-curve rows available.")
        return None

    fig, axes = plt.subplots(
        len(trait_order),
        1,
        figsize=(FULL_WIDTH, 8.2),
        sharex=False,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, trait in zip(axes, trait_order):
        sub = plot_df[plot_df["trait"].eq(trait)].copy()
        pos = rules_df[rules_df["trait"].eq(trait) & rules_df["method_label"].eq("Positive Shapley only")].copy()
        full = rules_df[rules_df["trait"].eq(trait) & rules_df["method_label"].eq("Full source pool")].copy()
        box_df = sub[sub["method_label"].isin(method_order)][["n_islands", "corr_eval", "method_label"]].copy()
        k_values = sorted(int(k) for k in box_df["n_islands"].dropna().unique())
        _draw_grouped_k_boxplots(ax, box_df, method_order, k_values, showfliers=True)

        pos_median = float(pos["corr_eval"].dropna().median()) if not pos.empty else np.nan
        if np.isfinite(pos_median):
            ax.axhline(pos_median, color=METHOD_COLORS["Positive Shapley only"],
                       linewidth=1.0, linestyle=":", label="Positive Shapley only")
        full_median = float(full["corr_eval"].dropna().median()) if not full.empty else np.nan
        if np.isfinite(full_median):
            ax.axhline(full_median, color="0.35", linewidth=0.9, linestyle=":",
                       label="All source islands")
        ax.set_title(TRAIT_LABELS.get(trait, trait))
        ax.set_ylabel("Pearson $r$")
        ax.set_xlabel("Shapley prefix step $k$ (source islands)")
        ax.set_xticks(range(1, 15, 2))
        ax.margins(x=0.03, y=0.12)
        style_axes(ax)

    handles = _method_legend_handles(method_order)
    handles.append(plt.Line2D([0], [0], color=METHOD_COLORS["Positive Shapley only"],
                              linestyle=":", linewidth=1.1,
                              label=METHOD_DISPLAY_LABELS["Positive Shapley only"]))
    handles.append(plt.Line2D([0], [0], color="0.35", linestyle=":", linewidth=1.0,
                              label=METHOD_DISPLAY_LABELS["All source islands"]))
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(handles),
        frameon=False,
    )
    if random_source == "e3_fallback":
        fig.text(0.01, 0.01, "Random baseline is interpolated from the E3 random PC-ridge individual runs.", fontsize=8)
    fig.subplots_adjust(top=0.90, bottom=0.075, hspace=0.55)
    pdf_path = save_figure(fig, output_dir / "e5_shapley_add_curves.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / "e5_shapley_add_curves.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path, random_source


def plot_e5_add_curves_by_island(
    e5: dict[str, pd.DataFrame],
    random_fallback_df: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, tuple[Path, Path]]:
    plot_df, _random_source = prepare_e5_add_plot_df(e5, random_fallback_df=random_fallback_df)
    rules_df = prepare_e5_subset_rule_rows(e5, random_fallback_df=random_fallback_df)
    method_order = _available_add_curve_methods(plot_df, rules_df)
    if not method_order:
        print("No E5 add-curve rows available for island grids.")
        return {}

    target_names: dict[int, str] = {}
    for key in ("summary", "repeats", "add"):
        df = e5.get(key, pd.DataFrame())
        if {"target_island", "target_island_name"}.issubset(df.columns):
            for row in df[["target_island", "target_island_name"]].dropna().drop_duplicates().itertuples(index=False):
                target_names[int(row.target_island)] = str(row.target_island_name)

    paths: dict[str, tuple[Path, Path]] = {}
    trait_order = [trait for trait in TRAIT_ORDER if trait in set(plot_df["trait"])]
    for trait in trait_order:
        trait_df = plot_df[plot_df["trait"].eq(trait)].copy()
        if trait_df.empty:
            continue
        targets = sorted(int(v) for v in trait_df["target_island"].dropna().unique())
        ncols = 3
        nrows = int(np.ceil(len(targets) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(9.6, 2.42 * nrows + 0.9),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes_flat = axes.ravel()
        for ax, target in zip(axes_flat, targets):
            sub = trait_df[trait_df["target_island"].eq(target)].copy()
            pos = rules_df[
                rules_df["trait"].eq(trait)
                & rules_df["target_island"].eq(target)
                & rules_df["method_label"].eq("Positive Shapley only")
            ].copy()
            full = rules_df[
                rules_df["trait"].eq(trait)
                & rules_df["target_island"].eq(target)
                & rules_df["method_label"].eq("Full source pool")
            ].copy()
            box_df = sub[sub["method_label"].isin(method_order)][["n_islands", "corr_eval", "method_label"]].copy()
            _draw_grouped_k_boxplots(ax, box_df, method_order, list(range(1, 15)), showfliers=False, linewidth=0.42)
            pos_value = float(pos["corr_eval"].dropna().median()) if not pos.empty else np.nan
            if np.isfinite(pos_value):
                ax.axhline(pos_value, color=METHOD_COLORS["Positive Shapley only"],
                           linewidth=0.75, linestyle=":")
            full_value = float(full["corr_eval"].dropna().median()) if not full.empty else np.nan
            if np.isfinite(full_value):
                ax.axhline(full_value, color="0.35", linewidth=0.75, linestyle=":")
            ax.set_title(target_names.get(target, str(target)), fontsize=10.2, pad=4)
            ax.set_xticks([1, 4, 7, 10, 13])
            ax.tick_params(axis="both", labelsize=8.6)
            style_axes(ax)
        for ax in axes_flat[len(targets):]:
            ax.axis("off")

        handles = _method_legend_handles(method_order)
        handles.append(plt.Line2D([0], [0], color=METHOD_COLORS["Positive Shapley only"],
                                  linestyle=":", linewidth=1.0,
                                  label=METHOD_DISPLAY_LABELS["Positive Shapley only"]))
        handles.append(plt.Line2D([0], [0], color="0.35", linestyle=":", linewidth=1.0,
                                  label=METHOD_DISPLAY_LABELS["All source islands"]))
        fig.suptitle(f"E5 Shapley add-curve distributions by target island, {TRAIT_LABELS.get(trait, trait)}", y=0.995, fontsize=14)
        fig.legend(
            handles,
            [handle.get_label() for handle in handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.963),
            ncol=4,
            frameon=False,
            fontsize=9.4,
        )
        fig.supxlabel("Shapley prefix step $k$ (source islands)", y=0.025, fontsize=11.5)
        fig.supylabel("Pearson $r$", x=0.014, fontsize=11.5)
        fig.subplots_adjust(left=0.07, right=0.995, bottom=0.07, top=0.91, wspace=0.14, hspace=0.45)
        pdf_path = save_figure(fig, output_dir / f"e5_shapley_add_curves_by_island_{trait}.pdf", repo_root, bbox_inches="tight")
        png_path = save_figure(fig, output_dir / f"e5_shapley_add_curves_by_island_{trait}.png", repo_root, bbox_inches="tight")
        plt.close(fig)
        paths[trait] = (pdf_path, png_path)
    return paths


def plot_e5_add_curves_by_island_representative(
    e5: dict[str, pd.DataFrame],
    random_fallback_df: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
    trait: str = "body_mass",
    selected_targets: tuple[int, ...] = (3, 5, 1, 12),
) -> tuple[Path, Path] | None:
    plot_df, _random_source = prepare_e5_add_plot_df(e5, random_fallback_df=random_fallback_df)
    rules_df = prepare_e5_subset_rule_rows(e5, random_fallback_df=random_fallback_df)
    method_order = _available_add_curve_methods(plot_df, rules_df)
    if not method_order:
        print("No E5 add-curve rows available for representative island plot.")
        return None

    trait_df = plot_df[plot_df["trait"].eq(trait)].copy()
    if trait_df.empty:
        print(f"No E5 add-curve rows available for {trait}.")
        return None

    available_targets = set(trait_df["target_island"].dropna().astype(int))
    targets = [target for target in selected_targets if target in available_targets]
    if not targets:
        targets = sorted(available_targets)[:4]

    target_names: dict[int, str] = {}
    for key in ("summary", "repeats", "add"):
        df = e5.get(key, pd.DataFrame())
        if {"target_island", "target_island_name"}.issubset(df.columns):
            for row in df[["target_island", "target_island_name"]].dropna().drop_duplicates().itertuples(index=False):
                target_names[int(row.target_island)] = str(row.target_island_name)

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.15), sharex=False, sharey=True, squeeze=False)
    axes_flat = axes.ravel()
    for ax, target in zip(axes_flat, targets):
        sub = trait_df[trait_df["target_island"].astype(int).eq(target)].copy()
        pos = rules_df[
            rules_df["trait"].eq(trait)
            & rules_df["target_island"].astype(int).eq(target)
            & rules_df["method_label"].eq("Positive Shapley only")
        ].copy()
        full = rules_df[
            rules_df["trait"].eq(trait)
            & rules_df["target_island"].astype(int).eq(target)
            & rules_df["method_label"].eq("Full source pool")
        ].copy()
        box_df = sub[sub["method_label"].isin(method_order)][["n_islands", "corr_eval", "method_label"]].copy()
        _draw_grouped_k_boxplots(ax, box_df, method_order, list(range(1, 15)), showfliers=False, linewidth=0.5)
        pos_value = float(pos["corr_eval"].dropna().median()) if not pos.empty else np.nan
        if np.isfinite(pos_value):
            ax.axhline(pos_value, color=METHOD_COLORS["Positive Shapley only"], linewidth=0.85, linestyle=":")
        full_value = float(full["corr_eval"].dropna().median()) if not full.empty else np.nan
        if np.isfinite(full_value):
            ax.axhline(full_value, color="0.35", linewidth=0.85, linestyle=":")
        ax.set_title(target_names.get(target, str(target)), fontsize=10.8, pad=4)
        ax.set_xticks([1, 4, 7, 10, 13])
        ax.tick_params(axis="both", labelsize=8.8)
        style_axes(ax)
    for ax in axes_flat[len(targets):]:
        ax.axis("off")

    handles = _method_legend_handles(method_order)
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color=METHOD_COLORS["Positive Shapley only"],
            linestyle=":",
            linewidth=1.0,
            label=METHOD_DISPLAY_LABELS["Positive Shapley only"],
        )
    )
    handles.append(
        plt.Line2D([0], [0], color="0.35", linestyle=":", linewidth=1.0, label=METHOD_DISPLAY_LABELS["All source islands"])
    )
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        fontsize=9.1,
    )
    fig.supxlabel("Shapley prefix step $k$ (source islands)", y=0.01, fontsize=11.2)
    fig.supylabel("Pearson $r$", x=0.02, fontsize=11.2)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.12, top=0.88, wspace=0.12, hspace=0.42)
    pdf_path = save_figure(fig, output_dir / f"e5_shapley_add_curves_by_island_{trait}_representative.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / f"e5_shapley_add_curves_by_island_{trait}_representative.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def plot_e5_positive_island_count_histograms(
    e5: dict[str, pd.DataFrame],
    random_fallback_df: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
) -> tuple[Path, Path] | None:
    rules_df = prepare_e5_subset_rule_rows(e5, random_fallback_df=random_fallback_df)
    pos = rules_df[rules_df["method_label"].eq("Positive Shapley only")].copy()
    if pos.empty:
        print("No positive-Shapley rows available for count histogram.")
        return None
    pos["target_region"] = np.where(
        pos["target_island"].astype(int).isin(SOUTHERN_TARGET_ISLANDS),
        "Southern targets",
        "Northern targets",
    )

    k_values = np.arange(0, 15)
    counts = pos.groupby("n_islands").size().reindex(k_values, fill_value=0).astype(float)
    shares = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, 3.15))
    ax.bar(
        k_values,
        shares,
        width=0.72,
        color="#9A9A9A",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.45,
    )
    mean_lines = [
        ("Northern targets", REGION_COLORS["Northern islands"], 4, "left", 0.94),
        ("Southern targets", REGION_COLORS["Southern islands"], -4, "right", 0.88),
    ]
    ymax = max(0.34, float(shares.max()) * 1.22)
    for label, color, x_offset, ha, y_frac in mean_lines:
        sub = pos[pos["target_region"].eq(label)]
        if sub.empty:
            continue
        mean_k = float(sub["n_islands"].mean())
        ax.axvline(mean_k, color=color, linestyle="--", linewidth=1.25)
        ax.annotate(
            f"{mean_k:.1f}",
            xy=(mean_k, ymax * y_frac),
            xytext=(x_offset, 0),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=7.6,
            color=color,
        )
    ax.set_title("Number of positive-Shapley source islands")
    ax.set_xlabel("Source islands with positive Shapley value")
    ax.set_ylabel("Share of cases")
    ax.set_xlim(-0.6, 14.6)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks(k_values)
    style_axes(ax)
    handles = [
        plt.Line2D([0], [0], color=REGION_COLORS["Northern islands"], linestyle="--",
                   linewidth=1.1, label="Northern mean"),
        plt.Line2D([0], [0], color=REGION_COLORS["Southern islands"], linestyle="--",
                   linewidth=1.1, label="Southern mean"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.18, top=0.86)
    pdf_path = save_figure(fig, output_dir / "e5_shapley_positive_island_counts.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / "e5_shapley_positive_island_counts.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def prepare_e5_shapley_avggrm_small_subset_df(
    e5: dict[str, pd.DataFrame],
    avggrm_df: pd.DataFrame,
    full_source_df: pd.DataFrame,
    shapley_k: int = 7,
    avggrm_n: int = 2500,
) -> pd.DataFrame:
    add_df = e5.get("add", pd.DataFrame())
    if add_df.empty or avggrm_df.empty or full_source_df.empty:
        return pd.DataFrame()
    shapley_required = {
        "trait", "target_island", "method", "n_islands", "n_individuals", "corr_eval",
    }
    avggrm_required = {"trait", "target_island", "method", "n_train_size", "n_train", "pearson_r"}
    full_required = {"trait", "target_island", "full_source_r"}
    if (
        not shapley_required.issubset(add_df.columns)
        or not avggrm_required.issubset(avggrm_df.columns)
        or not full_required.issubset(full_source_df.columns)
    ):
        return pd.DataFrame()

    name_lookup: dict[int, str] = {}
    for key in ("summary", "repeats", "metadata"):
        meta = e5.get(key, pd.DataFrame())
        if {"target_island", "target_island_name"}.issubset(meta.columns):
            for row in meta[["target_island", "target_island_name"]].dropna().drop_duplicates().itertuples(index=False):
                name_lookup[int(row.target_island)] = str(row.target_island_name)

    shapley = (
        add_df[
            add_df["method"].eq("shapley_mean")
            & add_df["n_islands"].eq(shapley_k)
        ]
        .groupby(["trait", "target_island"], as_index=False)
        .agg(
            shapley_r=("corr_eval", "median"),
            shapley_n_individuals=("n_individuals", "median"),
            shapley_n_repeats=("corr_eval", "count"),
        )
    )
    avggrm = (
        avggrm_df[
            avggrm_df["method"].eq("avggrm_topk")
            & avggrm_df["n_train_size"].eq(avggrm_n)
        ]
        .groupby(["trait", "target_island"], as_index=False)
        .agg(
            avggrm_r=("pearson_r", "median"),
            avggrm_n_train=("n_train", "median"),
            avggrm_n_repeats=("pearson_r", "count"),
        )
    )
    out = shapley.merge(avggrm, on=["trait", "target_island"], how="inner")
    out = out.merge(full_source_df, on=["trait", "target_island"], how="inner")
    if out.empty:
        return out
    out["shapley_delta_r"] = out["shapley_r"] - out["full_source_r"]
    out["avggrm_delta_r"] = out["avggrm_r"] - out["full_source_r"]
    out["target_island_name"] = out["target_island"].astype(int).map(name_lookup).fillna(out["target_island"].astype(str))
    out["trait_label"] = out["trait"].map(TRAIT_LABELS).fillna(out["trait"])
    out["target_region"] = np.where(
        out["target_island"].astype(int).isin(SOUTHERN_TARGET_ISLANDS),
        "Southern targets",
        "Northern targets",
    )
    return out.sort_values(["trait", "target_island"]).reset_index(drop=True)


def plot_e5_shapley_avggrm_small_subset_agreement(
    e5: dict[str, pd.DataFrame],
    avggrm_df: pd.DataFrame,
    full_source_df: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
    shapley_k: int = 7,
    avggrm_n: int = 2500,
) -> tuple[Path, Path, Path] | None:
    df = prepare_e5_shapley_avggrm_small_subset_df(
        e5,
        avggrm_df,
        full_source_df,
        shapley_k=shapley_k,
        avggrm_n=avggrm_n,
    )
    if df.empty:
        print("No matched E5 Shapley and E4 AvgGRM rows available for small-subset scatter.")
        return None

    trait_colors = {
        "body_mass": "#4C78A8",
        "thr_tarsus": "#F58518",
        "thr_wing": "#54A24B",
    }
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, 4.25))
    for trait in TRAIT_ORDER:
        sub = df[df["trait"].eq(trait)]
        if sub.empty:
            continue
        ax.scatter(
            sub["avggrm_delta_r"],
            sub["shapley_delta_r"],
            s=36,
            color=trait_colors.get(trait, "0.35"),
            edgecolor="white",
            linewidth=0.45,
            alpha=0.86,
            label=TRAIT_LABELS.get(trait, trait),
        )

    all_values = pd.concat([df["avggrm_delta_r"], df["shapley_delta_r"]]).dropna()
    span = max(abs(float(all_values.min())), abs(float(all_values.max())), 0.05)
    hi = float(np.ceil((span + 0.015) / 0.05) * 0.05)
    lo = -hi
    ax.plot([lo, hi], [lo, hi], color="0.35", linestyle="--", linewidth=1.0, label="1:1")
    ax.axvline(0.0, color="0.70", linestyle=":", linewidth=0.9)
    ax.axhline(0.0, color="0.70", linestyle=":", linewidth=0.9)

    pearson = df["avggrm_delta_r"].corr(df["shapley_delta_r"], method="pearson")
    spearman = df["avggrm_delta_r"].corr(df["shapley_delta_r"], method="spearman")
    ax.text(
        0.03,
        0.97,
        fr"$n={len(df)}$, Pearson $r={pearson:.2f}$, Spearman $\rho={spearman:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.85", linewidth=0.5, alpha=0.9),
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Small-subset gain agreement")
    ax.set_xlabel(fr"AvgGRM top-$k$, $n={avggrm_n}$ source individuals" "\n" r"$\Delta r$ vs full source")
    ax.set_ylabel(fr"Shapley order, $k={shapley_k}$ source islands" "\n" r"$\Delta r$ vs full source")
    ax.legend(loc="lower right", frameon=False)
    style_axes(ax)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.14, top=0.90)

    csv_path = output_dir / "e5_shapley_avggrm_small_subset_agreement.csv"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    except PermissionError as exc:
        fallback = repo_root / "figures" / csv_path.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(fallback, index=False)
        print(f"Could not write {csv_path} ({exc}). Wrote {fallback} instead.")
        csv_path = fallback
    pdf_path = save_figure(fig, output_dir / "e5_shapley_avggrm_small_subset_agreement.pdf", repo_root, bbox_inches="tight")
    png_path = save_figure(fig, output_dir / "e5_shapley_avggrm_small_subset_agreement.png", repo_root, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path, csv_path


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
    fig, axes = plt.subplots(1, len(trait_order), figsize=(4.15 * len(trait_order), 4.3), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, trait in zip(axes, trait_order):
        sub = rules_df[rules_df["trait"].eq(trait) & rules_df["method_label"].isin(present_methods)]
        values = [
            sub.loc[sub["method_label"].eq(method), "delta_full"].dropna().to_numpy(dtype=float)
            for method in present_methods
        ]
        bp = ax.boxplot(
            values,
            positions=np.arange(len(present_methods)),
            widths=0.58,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
            showcaps=False,
            medianprops=dict(color="0.12", linewidth=1.15),
            whiskerprops=dict(linewidth=0.85),
            flierprops=dict(marker="o", markersize=2.4, markerfacecolor="white", markeredgewidth=0.45, alpha=0.75),
        )
        for box, method in zip(bp["boxes"], present_methods):
            color = METHOD_COLORS.get(method, "#888888")
            box.set(facecolor=color, edgecolor=color, alpha=0.62, linewidth=0.9)
        for whisker, method in zip(bp["whiskers"], np.repeat(present_methods, 2)):
            whisker.set(color=METHOD_COLORS.get(method, "#888888"), linewidth=0.85)
        for flier, method in zip(bp["fliers"], present_methods):
            flier.set(markeredgecolor=METHOD_COLORS.get(method, "#888888"))
        ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle=":")
        labels = [
            str(label)
            .replace("Random individuals (E3 fallback)", "Random\n(E3)")
            .replace("Random individuals (E5)", "Random\n(E5)")
            .replace("Positive Shapley only", "Positive\nShapley")
            .replace("Best Shapley prefix (post hoc)", "Best prefix\n(post hoc)")
            for label in present_methods
        ]
        ax.set_xticks(np.arange(len(present_methods)))
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_title(TRAIT_LABELS.get(trait, trait), fontsize=13.5)
        style_axes(ax)
    axes[0].set_ylabel("Pearson r relative to all source islands", fontsize=12.5)
    fig.subplots_adjust(bottom=0.26, wspace=0.18, left=0.085, right=0.995, top=0.90)
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
    if metadata_df.empty:
        print("No validation-instability rows available.")
        return None

    trait_order = [trait for trait in TRAIT_ORDER if trait in set(metadata_df["trait"])]
    palette = sns.color_palette("colorblind", n_colors=len(trait_order))
    trait_colors = dict(zip(trait_order, palette))

    if "summary" in e5 and not e5["summary"].empty:
        base_order, base_names = island_order(e5["summary"])
        name_lookup = dict(zip(base_order, base_names))
        order = type_grouped_order(base_order)
        target_codes = [code for code in order if code in set(metadata_df["target_island"].astype(int))]
    else:
        target_codes = sorted(int(code) for code in metadata_df["target_island"].dropna().unique())
        name_lookup = (
            metadata_df[["target_island", "target_island_name"]]
            .dropna()
            .drop_duplicates()
            .assign(target_island=lambda df: df["target_island"].astype(int))
            .set_index("target_island")["target_island_name"]
            .astype(str)
            .to_dict()
        )
    labels = [name_lookup.get(code, str(code)) for code in target_codes]

    fig, ax = plt.subplots(figsize=(12.9, 5.05))
    offsets = np.linspace(-0.26, 0.26, len(trait_order)) if len(trait_order) > 1 else np.array([0.0])
    handles = []
    for trait, offset in zip(trait_order, offsets):
        sub = metadata_df[metadata_df["trait"].eq(trait)].copy()
        values = [
            sub.loc[sub["target_island"].astype(int).eq(target), "v_full"].dropna().to_numpy(dtype=float)
            for target in target_codes
        ]
        positions = np.arange(len(target_codes), dtype=float) + offset
        bp = ax.boxplot(
            values,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
            medianprops=dict(color="0.12", linewidth=1.15),
            boxprops=dict(facecolor=trait_colors[trait], edgecolor=trait_colors[trait], alpha=0.62, linewidth=0.9),
            whiskerprops=dict(color=trait_colors[trait], linewidth=0.9),
            capprops=dict(linewidth=0.0),
            flierprops=dict(
                marker="o",
                markersize=2.5,
                markerfacecolor="white",
                markeredgecolor=trait_colors[trait],
                markeredgewidth=0.45,
                alpha=0.8,
            ),
        )
        for cap in bp["caps"]:
            cap.set_visible(False)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markersize=7,
                markerfacecolor=trait_colors[trait],
                markeredgecolor=trait_colors[trait],
                alpha=0.75,
                label=TRAIT_LABELS.get(trait, trait),
            )
        )

    ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Target island", fontsize=15)
    ax.set_ylabel(r"Full-source utility $v_{\mathrm{full}}$", fontsize=15)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12.5)
    ax.tick_params(axis="y", labelsize=12.5)
    ax.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False, fontsize=13)
    style_axes(ax)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.26, top=0.95)

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
    avggrm_root: Path | None = None,
    full_source_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = find_repo_root() if repo_root is None else Path(repo_root)
    output_dir = repo_root / "figures" if output_dir is None else Path(output_dir)
    final_results_dir = repo_root / "outputs" / "final_results"
    e5_root = final_results_dir / "e5_shapley_islands_pc_ridge" if e5_root is None else Path(e5_root)
    random_fallback_root = final_results_dir / "e3_random_pc_ridge" if random_fallback_root is None else Path(random_fallback_root)
    avggrm_root = final_results_dir / "e4_similarity_pc_ridge" if avggrm_root is None else Path(avggrm_root)
    full_source_root = final_results_dir / "e1_pc_ridge_fixed_loio_l10" if full_source_root is None else Path(full_source_root)

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
    avggrm = load_e4_avggrm_results(avggrm_root)
    print(f"Loaded {len(avggrm)} E4 AvgGRM/similarity rows")
    full_source = load_e1_full_source_baseline(full_source_root)
    print(f"Loaded {len(full_source)} E1 full-source baseline rows")

    paths: dict[str, Any] = {}
    paths["heatmaps"] = plot_e5_shapley_heatmaps(e5, output_dir, repo_root)
    paths["island_type_table"] = write_island_type_shapley_table(e5, output_dir)
    paths["geographic_distance"] = plot_e5_shapley_geographic_distance_average(e5, output_dir, repo_root)
    paths["add_curves"] = plot_e5_add_curves(e5, random_fallback, output_dir, repo_root)
    paths["add_curves_by_island_representative"] = plot_e5_add_curves_by_island_representative(
        e5,
        random_fallback,
        output_dir,
        repo_root,
    )
    paths["add_curves_by_island"] = plot_e5_add_curves_by_island(e5, random_fallback, output_dir, repo_root)
    paths["positive_island_counts"] = plot_e5_positive_island_count_histograms(e5, random_fallback, output_dir, repo_root)
    paths["small_subset_agreement"] = plot_e5_shapley_avggrm_small_subset_agreement(
        e5,
        avggrm,
        full_source,
        output_dir,
        repo_root,
    )
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
