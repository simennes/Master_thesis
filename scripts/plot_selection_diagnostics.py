"""Subset-selection diagnostic figures for the thesis Results section.

Produces:
  * per_island_delta_r       - oracle gain of the best k for one subset method over the full pool, per island
  * gain_vs_isolation (N1)   - that gain against the target island's relatedness to the source pool
  * diversity_vs_k    (N2)   - internal relatedness of the selected subset, AvgGRM vs diversity

Run directly:  python scripts/plot_selection_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from thesis_style import (
    FULL_WIDTH,
    SEMANTIC_COLORS,
    TRAIT_COLORS,
    configure_thesis_style,
    style_axes,
)

# ---------------------------------------------------------------- constants

TRAITS = [
    ("body_mass", "Body mass", "Data/npz/snp_body_mass_ALL.npz"),
    ("thr_tarsus", "Tarsus length", "Data/npz/snp_thr_tarsus_ALL.npz"),
    ("thr_wing", "Wing length", "Data/npz/snp_thr_wing_ALL.npz"),
]
TRAIT_LABELS = {k: lab for k, lab, _ in TRAITS}
TRAIT_NPZ = {k: p for k, _, p in TRAITS}

GRM_RDS = "Data/GRM/GRM_vanraden.rds"

# Original island codes -> name (Norwegian).
ISLAND_ID_TO_NAME = {
    "20": "Nesøy", "22": "Myken", "23": "Træna", "24": "Selvær",
    "26": "Gjerøy", "27": "Hestmannøy", "28": "Indre Kvarøy", "33": "Onøy og Lurøy",
    "34": "Lovund", "35": "Sleneset", "38": "Aldra", "60": "Leka",
    "61": "Vega", "63": "Vikna", "67": "Lauvøya",
}
# Internal experiment codes 0..14 follow the sorted original codes.
_SORTED_ORIG = sorted(ISLAND_ID_TO_NAME, key=int)
INTERNAL_TO_NAME = {i: ISLAND_ID_TO_NAME[c] for i, c in enumerate(_SORTED_ORIG)}

# Selection methods (exclude the random baseline) available for subset-selection gains.
SELECTION_METHODS = [
    "pevmean_ga_pc_ridge",
    "avggrm_topk",
    "pca_target_topk",
    "avggrm_diversity_lam0p25",
    "avggrm_diversity_lam0p5",
    "avggrm_diversity_lam1",
    "avggrm_diversity_lam2",
]

SUBSET_METHOD_LABELS = {
    "random_pc_ridge": "Random",
    "pevmean_ga_pc_ridge": "PEV-mean",
    "avggrm_topk": "AvgGRM",
    "avggrm_diversity_lam1": "AvgGRM diversity",
    "pca_target_topk": "PC distance",
}
DEFAULT_DELTA_R_METHOD = "avggrm_diversity_lam1"

FINAL = "outputs/final_results"


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "Data").exists() and (path / "src").exists():
            return path
        candidate = path / "Master_thesis"
        if (candidate / "Data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not find the Master_thesis repo root.")


# ---------------------------------------------------------------- loaders

def _load_results_one(repo_root: Path, sub: str, stem: str) -> pd.DataFrame:
    base = repo_root / FINAL / sub
    if not base.exists():
        return pd.DataFrame()
    frames = []
    for trait_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        csv = trait_dir / f"{stem}_results.csv"
        if csv.exists():
            frames.append(pd.read_csv(csv))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_subset_results(repo_root: Path) -> pd.DataFrame:
    """Per-fold subset-selection results (PEVmean GA, random, E4 similarity)."""
    pev = _load_results_one(repo_root, "e3_pevmean_ga_pc_ridge", "pevmean_ga_pc_ridge")
    rand = _load_results_one(repo_root, "e3_random_pc_ridge", "random_pc_ridge")
    sim = _load_results_one(repo_root, "e4_similarity_pc_ridge", "e4_similarity_pc_ridge")
    return pd.concat([d for d in (pev, rand, sim) if not d.empty], ignore_index=True)


def load_full_pool_baseline(repo_root: Path) -> pd.DataFrame:
    """Full-source PC-ridge LOIO baseline: mean Pearson r per (trait, island code)."""
    csv = repo_root / FINAL / "e1_pc_ridge_fixed_loio_l10" / "e1_pc_ridge_fixed_loio_l10_per_fold_results.csv"
    df = pd.read_csv(csv)
    out = (
        df.groupby(["trait", "test_island_code"], as_index=False)["pearson_r"]
        .mean()
        .rename(columns={"test_island_code": "target_island", "pearson_r": "full_pool_r"})
    )
    return out


def delta_r_per_island(repo_root: Path,
                       method: str = DEFAULT_DELTA_R_METHOD) -> pd.DataFrame:
    """Oracle gain of the best subset size for one method over the full pool.

    best_subset_r is the maximum repeat-averaged Pearson r over subset sizes
    for the selected method, computed separately per trait and target island.
    full_pool_r is the E1 full-source baseline. If the selected method is worse
    than the full pool, the oracle gain is zero because the full pool would be
    chosen instead.
    """
    res = load_subset_results(repo_root)
    if method not in set(res["method"]):
        available = ", ".join(sorted(res["method"].dropna().unique()))
        raise ValueError(f"Method {method!r} not found. Available methods: {available}")
    sel = res[res["method"] == method].copy()
    per_k = (
        sel.groupby(["trait", "target_island", "method", "n_train_size"], as_index=False)["pearson_r"]
        .mean()
    )
    idx = per_k.groupby(["trait", "target_island"])["pearson_r"].idxmax()
    best = (
        per_k.loc[idx, ["trait", "target_island", "method", "n_train_size", "pearson_r"]]
        .rename(columns={"pearson_r": "best_subset_r", "n_train_size": "best_k"})
    )
    base = load_full_pool_baseline(repo_root)
    out = best.merge(base, on=["trait", "target_island"], how="left")
    out["delta_r_raw"] = out["best_subset_r"] - out["full_pool_r"]
    out["use_full_source"] = out["delta_r_raw"] < 0
    out["oracle_r"] = np.maximum(out["best_subset_r"], out["full_pool_r"])
    out["delta_r"] = out["delta_r_raw"].clip(lower=0.0)
    out["island_name"] = out["target_island"].map(INTERNAL_TO_NAME)
    out["trait_label"] = out["trait"].map(TRAIT_LABELS)
    out["method_label"] = SUBSET_METHOD_LABELS.get(method, method)
    return out


def load_grm_islands(repo_root: Path, trait: str = "body_mass", min_count: int = 20):
    """Return GRM submatrix, ids, and per-id island name for the experiment cohort."""
    import pyreadr

    data = np.load(repo_root / TRAIT_NPZ[trait], allow_pickle=False)
    ids = data["ids"].astype(str)
    locality = data["locality"].astype(str)
    locality = np.where(locality == "68", "67", locality)
    unique, counts = np.unique(locality, return_counts=True)
    keep_codes = set(unique[counts >= min_count])
    mask = np.isin(locality, list(keep_codes))
    ids, locality = ids[mask], locality[mask]

    grm_df = next(iter(pyreadr.read_r(str(repo_root / GRM_RDS)).values()))
    grm_df.index = grm_df.index.astype(str)
    grm_df.columns = grm_df.columns.astype(str)

    present = [i for i in ids if i in grm_df.index]
    order = pd.Index(present)
    keep = np.isin(ids, present)
    ids, locality = ids[keep], locality[keep]
    # reorder GRM to match ids
    grm = grm_df.loc[ids, ids].to_numpy(dtype=np.float64)
    island_name = np.array([ISLAND_ID_TO_NAME[c] for c in locality])
    return grm, ids, island_name


def island_avggrm_matrix(grm: np.ndarray, island_name: np.ndarray) -> pd.DataFrame:
    """Mean GRM between every pair of islands (diagonal excludes self-pairs)."""
    names = sorted(set(island_name))
    idx = {n: np.flatnonzero(island_name == n) for n in names}
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            block = grm[np.ix_(idx[a], idx[b])]
            if a == b:
                n = block.shape[0]
                val = (block.sum() - np.trace(block)) / (n * (n - 1)) if n > 1 else np.nan
            else:
                val = block.mean()
            mat.loc[a, b] = val
    return mat


# ---------------------------------------------------------------- figures

def _save(fig, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        p = output_dir / f"{stem}.{suffix}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths[0], paths[1]


def plot_per_island_delta_r(delta_df: pd.DataFrame, output_dir: Path,
                            stem: str = "selection_per_island_delta_r"):
    configure_thesis_style()
    order = (
        delta_df.groupby("island_name")["delta_r"].mean().sort_values().index.tolist()
    )
    ypos = {name: i for i, name in enumerate(order)}
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.6), constrained_layout=True)
    ax.axvline(0, color=SEMANTIC_COLORS["reference"], linewidth=0.9, linestyle="--")
    for trait, lab, _ in TRAITS:
        sub = delta_df[delta_df["trait"] == trait]
        ax.scatter(
            sub["delta_r"], [ypos[n] for n in sub["island_name"]],
            s=34, color=TRAIT_COLORS[lab], label=lab, alpha=0.9,
            edgecolor="white", linewidth=0.4, zorder=3,
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    method_label = (
        delta_df["method_label"].dropna().iloc[0]
        if "method_label" in delta_df and not delta_df["method_label"].dropna().empty
        else "selected method"
    )
    ax.set_xlabel(fr"Oracle $\Delta r$ (best $k$ with {method_label} or full source pool)")
    ax.set_ylabel("Target island")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    return _save(fig, output_dir, stem)


def plot_gain_vs_isolation(delta_df: pd.DataFrame, repo_root: Path, output_dir: Path,
                           stem: str = "selection_gain_vs_isolation"):
    grm, ids, island_name = load_grm_islands(repo_root, "body_mass")
    mat = island_avggrm_matrix(grm, island_name)
    relatedness = {}
    for target in mat.index:
        others = mat.loc[target, [c for c in mat.columns if c != target]].astype(float)
        relatedness[target] = others.max()  # AvgGRM to most-related source island

    df = delta_df.copy()
    df["relatedness"] = df["island_name"].map(relatedness)
    df = df.dropna(subset=["relatedness"])

    configure_thesis_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.2), constrained_layout=True)
    ax.axhline(0, color=SEMANTIC_COLORS["reference"], linewidth=0.9, linestyle="--")
    for trait, lab, _ in TRAITS:
        sub = df[df["trait"] == trait]
        ax.scatter(sub["relatedness"], sub["delta_r"], s=36, color=TRAIT_COLORS[lab],
                   label=lab, alpha=0.9, edgecolor="white", linewidth=0.4)
    rho, p = spearmanr(df["relatedness"], df["delta_r"])
    ax.text(0.04, 0.05, fr"Spearman $\rho = {rho:.2f}$", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5))
    ax.set_xlabel("AvgGRM to most-related source island")
    ax.set_ylabel(r"Oracle $\Delta r$")
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)
    return _save(fig, output_dir, stem)


def plot_diversity_vs_k(repo_root: Path, output_dir: Path,
                        target_island_name: str = "Sleneset",
                        lambdas=(1.0, 2.0), max_k: int = 4500,
                        stem: str = "selection_diversity_vs_k"):
    import sys
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.avggrm_weighting import avg_grm_train_to_target, greedy_avggrm_diversity_order

    grm, ids, island_name = load_grm_islands(repo_root, "body_mass")
    target_idx = np.flatnonzero(island_name == target_island_name)
    source_idx = np.flatnonzero(island_name != target_island_name)
    src_grm = grm[np.ix_(source_idx, source_idx)]
    avg_grm = avg_grm_train_to_target(grm, source_idx, target_idx)
    k_eff = int(min(max_k, source_idx.size))

    orders = {"AvgGRM top-k": np.argsort(-avg_grm, kind="mergesort")[:k_eff]}
    for lam in lambdas:
        orders[fr"AvgGRM diversity ($\lambda={lam:g}$)"] = greedy_avggrm_diversity_order(
            avg_grm_to_target=avg_grm, train_train_grm=src_grm,
            lambda_div=float(lam), max_size=k_eff, include_diagonal=True,
        )["order"][:k_eff]

    ks = [k for k in (50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500) if k <= k_eff]

    def mean_within(order, k):
        s = order[:k]
        block = src_grm[np.ix_(s, s)]
        return (block.sum() - np.trace(block)) / (k * (k - 1))

    configure_thesis_style()
    # AvgGRM top-k green (matches the learning-curve colour); diversity lines blue/red.
    line_colors = [TRAIT_COLORS["Tarsus length"], SEMANTIC_COLORS["observed"],
                   SEMANTIC_COLORS["adjusted"]]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.0), constrained_layout=True)
    for (label, order), color in zip(orders.items(), line_colors):
        ys = [mean_within(order, k) for k in ks]
        ax.plot(ks, ys, marker="o", markersize=4.5, linewidth=1.6,
                color=color, label=label)
    ax.set_xlabel(r"Selected subset size $k$")
    ax.set_ylabel("Mean within-subset relatedness (AvgGRM)")
    ax.set_title(f"Selected source individuals for {target_island_name}, body mass")
    ax.legend(frameon=False)
    style_axes(ax)
    return _save(fig, output_dir, stem)


def plot_avggrm_vs_shapley(repo_root: Path, output_dir: Path,
                           stem: str = "selection_avggrm_vs_shapley"):
    grm, ids, island_name = load_grm_islands(repo_root, "body_mass")
    mat = island_avggrm_matrix(grm, island_name)

    frames = []
    for trait, lab, _ in TRAITS:
        csv = (repo_root / FINAL / "e5_shapley_islands_pc_ridge" / trait
               / "shapley_island_summary_all_targets.csv")
        if not csv.exists():
            continue
        s = pd.read_csv(csv)
        s["target_name"] = s["target_island"].map(INTERNAL_TO_NAME)
        s["source_name"] = s["source_island"].map(INTERNAL_TO_NAME)
        s["avggrm"] = [mat.loc[t, src] for t, src in zip(s["target_name"], s["source_name"])]
        s["trait"] = trait
        frames.append(s)
    df = pd.concat(frames, ignore_index=True).dropna(subset=["avggrm", "phi_per_ind_mean"])

    configure_thesis_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.2), constrained_layout=True)
    ax.axhline(0, color=SEMANTIC_COLORS["reference"], linewidth=0.9, linestyle="--")
    for trait, lab, _ in TRAITS:
        sub = df[df["trait"] == trait]
        ax.scatter(sub["avggrm"], sub["phi_per_ind_mean"] * 1e4, s=26,
                   color=TRAIT_COLORS[lab], label=lab, alpha=0.75,
                   edgecolor="white", linewidth=0.3)
    rho, p = spearmanr(df["avggrm"], df["phi_per_ind_mean"])
    ax.text(0.04, 0.93, fr"Spearman $\rho = {rho:.2f}$", transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5))
    ax.set_xlabel("Island-level AvgGRM (target to source island)")
    ax.set_ylabel(r"Mean Shapley value per individual ($\times 10^{-4}$)")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    return _save(fig, output_dir, stem)


# Subset-selection method colours (palette; consistent across the comparison figures).
SUBSET_METHOD_STYLE = [
    ("random_pc_ridge", SUBSET_METHOD_LABELS["random_pc_ridge"], "#8F8F8F"),
    ("pevmean_ga_pc_ridge", SUBSET_METHOD_LABELS["pevmean_ga_pc_ridge"], "#D55E00"),
    ("avggrm_topk", SUBSET_METHOD_LABELS["avggrm_topk"], "#3F8F5B"),
    ("avggrm_diversity_lam1", SUBSET_METHOD_LABELS["avggrm_diversity_lam1"], "#83BF73"),
    ("pca_target_topk", SUBSET_METHOD_LABELS["pca_target_topk"], "#4C78A8"),
]


def plot_subset_boxplots(repo_root: Path, output_dir: Path,
                         stem: str = "e3_subset_boxplots"):
    """Per-trait grouped boxplots of Pearson r by method at each subset size k.

    Three stacked panels (one per trait); at each k the methods are drawn as
    small boxplots side by side, summarising the distribution across the 15
    target islands (and repeats). The dotted line is the median full-source
    baseline across target islands.
    """
    res = load_subset_results(repo_root)
    base = load_full_pool_baseline(repo_root).groupby("trait")["full_pool_r"].median()

    methods = [m for m in SUBSET_METHOD_STYLE if m[0] in set(res["method"])]
    k_values = sorted(
        k for k in res["n_train_size"].unique()
        if all(not res[(res.method == m[0]) & (res.n_train_size == k)].empty for m in methods)
    )
    n_methods = len(methods)
    width = 0.8 / n_methods

    configure_thesis_style()
    fig, axes = plt.subplots(len(TRAITS), 1, figsize=(FULL_WIDTH, 7.6),
                             sharex=False, constrained_layout=False)
    for ax, (trait, lab, _) in zip(axes, TRAITS):
        for j, (mkey, mlabel, color) in enumerate(methods):
            offset = (j - (n_methods - 1) / 2) * width
            positions, data = [], []
            for i, k in enumerate(k_values):
                vals = res[(res.trait == trait) & (res.method == mkey)
                           & (res.n_train_size == k)]["pearson_r"].dropna().to_numpy()
                if vals.size:
                    positions.append(i + offset)
                    data.append(vals)
            bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                            patch_artist=True, manage_ticks=False, showfliers=False,
                            showcaps=False,
                            medianprops=dict(color="0.15", linewidth=0.9),
                            whiskerprops=dict(color=color, linewidth=0.8))
            for box in bp["boxes"]:
                box.set(facecolor=color, edgecolor=color, alpha=0.55, linewidth=0.6)
        if trait in base.index:
            ax.axhline(base[trait], color=SEMANTIC_COLORS["reference"],
                       linewidth=1.0, linestyle=":")
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_ylabel("Pearson $r$")
        ax.set_xlabel(r"Training-set size $k$")
        ax.set_title(lab)
        style_axes(ax)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="none", markersize=8,
                          markerfacecolor=c, markeredgecolor=c, alpha=0.65, label=l)
               for _, l, c in methods]
    handles.append(plt.Line2D([0], [0], color=SEMANTIC_COLORS["reference"],
                              linestyle=":", label="Full-source median"))
    fig.suptitle("Subset-selection performance by training-set size",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.legend(handles=handles, frameon=False, ncol=len(handles), loc="upper center",
               bbox_to_anchor=(0.5, 0.94), fontsize=8.2,
               columnspacing=0.9, handlelength=1.2, handletextpad=0.35)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.08, top=0.855, hspace=0.58)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        p = output_dir / f"{stem}.{suffix}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths[0], paths[1]


def plot_pc1_vs_latitude(repo_root: Path, output_dir: Path,
                         trait: str = "body_mass", stem: str = "pc1_vs_latitude"):
    """Scatter of genotype PC1 against island latitude (north--south coordinate)."""
    data = np.load(repo_root / TRAIT_NPZ[trait], allow_pickle=False)
    X = data["snp"].astype(np.float32)
    ids = data["ids"].astype(str)
    locality = data["locality"].astype(str)
    locality = np.where(locality == "68", "67", locality)
    unique, counts = np.unique(locality, return_counts=True)
    keep = set(unique[counts >= 20])
    mask = np.isin(locality, list(keep))
    X, locality = X[mask], locality[mask]

    from sklearn.decomposition import PCA
    pc1 = PCA(n_components=2, svd_solver="randomized", random_state=14).fit_transform(X)[:, 0]

    # island latitude from the Shapley geographic-distance table
    geo = pd.read_csv(repo_root / FINAL / "e5_shapley_islands_pc_ridge"
                      / "e5_shapley_geographic_distance.csv")
    g = geo.drop_duplicates("source_island")[["source_island", "source_lat"]].copy()
    g["name"] = g["source_island"].map(INTERNAL_TO_NAME)
    lat_by_name = g.set_index("name")["source_lat"].to_dict()
    orig_to_name = ISLAND_ID_TO_NAME
    lat = np.array([lat_by_name.get(orig_to_name.get(c), np.nan) for c in locality])
    southern = {"Leka", "Vega", "Vikna", "Lauvøya"}
    region = np.array(["Southern islands" if orig_to_name.get(c) in southern
                       else "Helgeland islands" for c in locality])

    keep2 = np.isfinite(lat)
    pc1, lat, region = pc1[keep2], lat[keep2], region[keep2]

    configure_thesis_style()
    colors = {"Helgeland islands": SEMANTIC_COLORS["observed"],
              "Southern islands": SEMANTIC_COLORS["adjusted"]}
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.2), constrained_layout=True)
    for reg in ["Helgeland islands", "Southern islands"]:
        m = region == reg
        ax.scatter(pc1[m], lat[m], s=10, alpha=0.35, color=colors[reg],
                   edgecolor="none", label=reg, rasterized=True)
    rho, _ = spearmanr(pc1, lat)
    ax.text(0.04, 0.05, fr"Spearman $\rho = {rho:.2f}$", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5))
    ax.set_xlabel("Genotype PC1")
    ax.set_ylabel(r"Island latitude ($^\circ$N)")
    ax.legend(frameon=False, loc="upper right", markerscale=1.6)
    style_axes(ax)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        p = output_dir / f"{stem}.{suffix}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths[0], paths[1]


def plot_delta_vs_baseline(repo_root: Path, output_dir: Path, delta_df: pd.DataFrame | None = None,
                           stem: str = "selection_delta_vs_baseline"):
    """Per-island subset gain against the full-source baseline accuracy, per trait."""
    if delta_df is None:
        delta_df = delta_r_per_island(repo_root)
    configure_thesis_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.2), constrained_layout=True)
    ax.axhline(0, color=SEMANTIC_COLORS["reference"], linewidth=0.9, linestyle="--")
    for trait, lab, _ in TRAITS:
        s = delta_df[delta_df["trait"] == trait]
        rho, _ = spearmanr(s["full_pool_r"], s["delta_r"])
        ax.scatter(s["full_pool_r"], s["delta_r"], s=40, color=TRAIT_COLORS[lab],
                   label=fr"{lab} ($\rho={rho:+.2f}$)", alpha=0.85,
                   edgecolor="white", linewidth=0.4)
    ax.set_xlabel(r"Full-source Pearson $r$")
    ax.set_ylabel(r"$\Delta r$ (best subset $-$ full pool)")
    ax.legend(frameon=False, loc="upper right", title="Spearman $\\rho$ per trait")
    style_axes(ax)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        p = output_dir / f"{stem}.{suffix}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths[0], paths[1]


def _pevmean_selected_ringnrs(repo_root: Path, trait: str, k: int) -> dict[int, set]:
    """Best-objective PEVmean-GA selected ringnumbers per target island code, at size k."""
    base = repo_root / FINAL / "e3_pevmean_ga" / trait
    res_csv, sel_csv = base / "pevmean_ga_results.csv", base / "selected_individuals" / f"k_{k}.csv"
    if not res_csv.exists() or not sel_csv.exists():
        return {}
    res = pd.read_csv(res_csv)
    sel = pd.read_csv(sel_csv, dtype={"ringnumber": str})
    out: dict[int, set] = {}
    for code, grp in res[res["n_individuals"] == k].groupby("target_island"):
        best_rep = int(grp.sort_values("pevmean_obj").iloc[0]["repeat"])
        ids = sel[(sel["target_island"] == code) & (sel["repeat"] == best_rep)
                  & (sel["n_train_size"] == k)]["ringnumber"]
        out[int(code)] = set(ids.astype(str))
    return out


def _shapley_island_order(repo_root: Path, trait: str) -> dict[int, list]:
    """Source-island codes ordered by descending mean Shapley value, per target code."""
    csv = (repo_root / FINAL / "e5_shapley_islands_pc_ridge" / trait
           / "shapley_island_summary_all_targets.csv")
    if not csv.exists():
        return {}
    s = pd.read_csv(csv)
    out: dict[int, list] = {}
    for tcode, grp in s.groupby("target_island"):
        out[int(tcode)] = (grp.sort_values("phi_per_ind_mean", ascending=False)
                           ["source_island"].astype(int).tolist())
    return out


# Methods compared in the selection-signature / PC-space figures (key, label, colour).
SIGNATURE_METHODS = [
    ("random_pc_ridge", "Random", "#8F8F8F"),
    ("pevmean_ga_pc_ridge", "PEVmean-GA", "#D55E00"),
    ("avggrm_topk", "AvgGRM top-$k$", "#3F8F5B"),
    ("avggrm_diversity_lam1", r"AvgGRM diversity ($\lambda=1$)", "#83BF73"),
    ("pca_target_topk", "PC distance", "#4C78A8"),
    ("shapley", "Data Shapley", "#9467BD"),
]


def _selection_indices_for_target(avg, src_grm, pcd, source_idx, island_name, ids, k,
                                   pev_ringnrs, shap_order, rng):
    """Return {method: source-local indices} for one target island and size k."""
    from src.avggrm_weighting import greedy_avggrm_diversity_order
    div = greedy_avggrm_diversity_order(
        avg_grm_to_target=avg, train_train_grm=src_grm,
        lambda_div=1.0, max_size=k, include_diagonal=True)["order"][:k]
    sels = {
        "random_pc_ridge": rng.choice(source_idx.size, size=k, replace=False),
        "avggrm_topk": np.argsort(-avg, kind="mergesort")[:k],
        "avggrm_diversity_lam1": np.asarray(div, dtype=int),
        "pca_target_topk": np.argsort(pcd, kind="mergesort")[:k],
    }
    src_ids = ids[source_idx]
    id_to_local = {sid: j for j, sid in enumerate(src_ids)}
    pidx = np.array([id_to_local[r] for r in pev_ringnrs if r in id_to_local], dtype=int)
    if pidx.size >= k // 2:
        sels["pevmean_ga_pc_ridge"] = pidx
    # Data Shapley: take individuals from the top Shapley-ranked source islands up to k.
    src_names = island_name[source_idx]
    chosen: list[int] = []
    for sc in shap_order:
        members = np.flatnonzero(src_names == INTERNAL_TO_NAME.get(sc))
        if members.size == 0:
            continue
        members = members[np.argsort(-avg[members])]
        chosen.extend(members.tolist())
        if len(chosen) >= k:
            break
    if len(chosen) >= k:
        sels["shapley"] = np.asarray(chosen[:k], dtype=int)
    return sels


def _selection_signature_data(repo_root: Path, k: int, pca_distance_pcs: int,
                              cache: bool = True) -> pd.DataFrame:
    """Per (trait, target, method) relatedness and within-set redundancy of the selection."""
    cache_path = repo_root / "figures" / f"selection_signature_data_k{k}.csv"
    if cache and cache_path.exists():
        return pd.read_csv(cache_path)

    import sys
    from sklearn.decomposition import PCA
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.avggrm_weighting import avg_grm_train_to_target

    name_to_code = {v: kk for kk, v in INTERNAL_TO_NAME.items()}
    rows = []
    for trait, _, _ in TRAITS:
        X, grm, ids, island_name = _load_similarity_benchmark_problem(repo_root, trait)
        n_comp = int(min(max(3, pca_distance_pcs), X.shape[0], X.shape[1]))
        pc = PCA(n_components=n_comp, svd_solver="randomized", random_state=14).fit_transform(X)
        pcs = int(min(pca_distance_pcs, pc.shape[1]))
        pev_map = _pevmean_selected_ringnrs(repo_root, trait, k)
        shap_order = _shapley_island_order(repo_root, trait)
        rng = np.random.default_rng(14)
        for target in sorted(set(island_name)):
            tmask = island_name == target
            target_idx, source_idx = np.flatnonzero(tmask), np.flatnonzero(~tmask)
            if source_idx.size < k:
                continue
            avg = avg_grm_train_to_target(grm, source_idx, target_idx)
            src_grm = grm[np.ix_(source_idx, source_idx)]
            centroid = pc[target_idx, :pcs].mean(axis=0)
            pcd = np.linalg.norm(pc[source_idx, :pcs] - centroid[None, :], axis=1)
            code = name_to_code.get(target)
            sels = _selection_indices_for_target(
                avg, src_grm, pcd, source_idx, island_name, ids, k,
                pev_map.get(code, set()), shap_order.get(code, []), rng)
            for mkey, S in sels.items():
                S = np.asarray(S, dtype=int)
                sub = src_grm[np.ix_(S, S)]
                redundancy = (sub.sum() - np.trace(sub)) / (S.size * (S.size - 1))
                rows.append({"trait": trait, "target": target, "method": mkey,
                             "relatedness": float(avg[S].mean()), "redundancy": float(redundancy)})
    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def plot_selection_signature(repo_root: Path, output_dir: Path, k: int = 1000,
                             pca_distance_pcs: int = 20, cache: bool = True,
                             stem: str = "selection_signature"):
    """Each method in a target-relatedness vs. within-set-redundancy plane.

    All three traits are shown in one panel: colour encodes the selection method
    and marker shape encodes the trait. Each point is a method mean over the 15
    target islands; AvgGRM-diversity uses the penalty $\\lambda=1$.
    """
    from matplotlib.lines import Line2D
    df = _selection_signature_data(repo_root, k, pca_distance_pcs, cache=cache)
    configure_thesis_style()
    methods = [m for m in SIGNATURE_METHODS if m[0] in set(df["method"])]
    trait_markers = {"body_mass": "o", "thr_tarsus": "s", "thr_wing": "^"}

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.8), constrained_layout=True)
    for mkey, _, mcolor in methods:
        for trait, _, _ in TRAITS:
            s = df[(df["method"] == mkey) & (df["trait"] == trait)]
            if s.empty:
                continue
            ax.scatter(s["relatedness"].mean(), s["redundancy"].mean(), s=95, color=mcolor,
                       marker=trait_markers[trait], edgecolor="black", linewidth=0.6, zorder=5)
    ax.set_xlabel("Relatedness of selected set to target (mean AvgGRM)")
    ax.set_ylabel("Within-set relatedness (redundancy)")

    method_handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=c,
                             markeredgecolor="black", markersize=8, label=l) for _, l, c in methods]
    trait_handles = [Line2D([0], [0], marker=trait_markers[t], linestyle="none",
                            markerfacecolor="0.6", markeredgecolor="black", markersize=8, label=lab)
                     for t, lab, _ in TRAITS]
    leg1 = ax.legend(handles=method_handles, frameon=False, loc="upper left",
                     fontsize=8, title="Method")
    ax.add_artist(leg1)
    ax.legend(handles=trait_handles, frameon=False, loc="lower right", fontsize=8, title="Trait")
    style_axes(ax)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        p = output_dir / f"{stem}.{suffix}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths[0], paths[1]


def plot_selection_pca(repo_root: Path, output_dir: Path, trait: str = "body_mass",
                       target_name: str = "Sleneset", k: int = 1000,
                       pca_distance_pcs: int = 20, stem: str = "selection_pca_methods"):
    """Source individuals selected by each method, in PC1--PC3 space, one panel per method."""
    import sys
    from sklearn.decomposition import PCA
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.avggrm_weighting import avg_grm_train_to_target

    X, grm, ids, island_name = _load_similarity_benchmark_problem(repo_root, trait)
    n_comp = int(min(max(3, pca_distance_pcs), X.shape[0], X.shape[1]))
    pc = PCA(n_components=n_comp, svd_solver="randomized", random_state=14).fit_transform(X)
    pcs = int(min(pca_distance_pcs, pc.shape[1]))
    name_to_code = {v: kk for kk, v in INTERNAL_TO_NAME.items()}
    code = name_to_code[target_name]

    tmask = island_name == target_name
    target_idx, source_idx = np.flatnonzero(tmask), np.flatnonzero(~tmask)
    avg = avg_grm_train_to_target(grm, source_idx, target_idx)
    src_grm = grm[np.ix_(source_idx, source_idx)]
    centroid = pc[target_idx, :pcs].mean(axis=0)
    pcd = np.linalg.norm(pc[source_idx, :pcs] - centroid[None, :], axis=1)
    sels = _selection_indices_for_target(
        avg, src_grm, pcd, source_idx, island_name, ids, k,
        _pevmean_selected_ringnrs(repo_root, trait, k).get(code, set()),
        _shapley_island_order(repo_root, trait).get(code, []),
        np.random.default_rng(14))

    panels = [(lab, sels[key]) for key, lab, _ in SIGNATURE_METHODS
              if key in sels and key != "random_pc_ridge"]

    x_all, y_all = pc[:, 0], pc[:, 2]
    src_x, src_y = x_all[source_idx], y_all[source_idx]
    tgt_x, tgt_y = x_all[target_idx], y_all[target_idx]
    cx, cy = float(tgt_x.mean()), float(tgt_y.mean())
    other, tcol, scol = SEMANTIC_COLORS["context"], SEMANTIC_COLORS["observed"], SEMANTIC_COLORS["adjusted"]

    configure_thesis_style()
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, 3.25 * nrows),
                             sharex=True, sharey=True, constrained_layout=False)
    axes = np.atleast_1d(axes).ravel()
    for idx, (ax, (title, S)) in enumerate(zip(axes, panels)):
        S = np.asarray(S, dtype=int)
        mask = np.zeros(source_idx.size, dtype=bool)
        mask[S] = True
        ax.scatter(src_x[~mask], src_y[~mask], s=5, color=other, alpha=0.35,
                   edgecolor="none", label="Other source")
        ax.scatter(tgt_x, tgt_y, s=19, facecolors="none", alpha=0.95,
                   edgecolors=tcol, linewidth=0.75, label="Target island", zorder=3)
        ax.scatter(src_x[mask], src_y[mask], s=10, color=scol, alpha=0.82,
                   edgecolor="none", label="Selected source", zorder=4)
        ax.scatter([cx], [cy], marker="D", s=42, color="black", zorder=6,
                   edgecolor="white", linewidth=0.5, label="Target centroid")
        ax.set_title(title, pad=7)
        ax.tick_params(axis="x", labelbottom=(idx // ncols == nrows - 1))
        style_axes(ax)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for idx, ax in enumerate(axes[:len(panels)]):
        ax.set_xlabel("PC1" if idx // ncols == nrows - 1 else "")
    for r in range(nrows):
        axes[r * ncols].set_ylabel("PC3")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.89, bottom=0.10, hspace=0.36, wspace=0.13)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        p = output_dir / f"{stem}.{suffix}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths[0], paths[1]


def plot_subset_curves_by_island(repo_root: Path, output_dir: Path, trait: str = "body_mass",
                                 stem: str = "e3_learning_curves_by_island_main"):
    """Per-island learning curves (lines + markers, mean +/- SD band).

    Shows only random, PEVmean-GA, and the best similarity method
    (AvgGRM-diversity) to keep the small panels readable.
    """
    keep_keys = ("random_pc_ridge", "pevmean_ga_pc_ridge", "avggrm_diversity_lam1")
    res = load_subset_results(repo_root)
    res = res[res["trait"] == trait]
    base = load_full_pool_baseline(repo_root)
    base = dict(zip(base[base["trait"] == trait]["target_island"], base[base["trait"] == trait]["full_pool_r"]))
    methods = [m for m in SUBSET_METHOD_STYLE if m[0] in keep_keys and m[0] in set(res["method"])]
    islands = sorted(res["target_island"].unique())

    configure_thesis_style()
    ncols = 3
    nrows = int(np.ceil(len(islands) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, 1.3 * nrows + 0.5),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, isl in zip(axes, islands):
        sub = res[res["target_island"] == isl]
        for mkey, mlabel, color in methods:
            g = (sub[sub["method"] == mkey].groupby("n_train_size")["pearson_r"]
                 .agg(["mean", "std"]).reset_index().sort_values("n_train_size"))
            if g.empty:
                continue
            ax.plot(g["n_train_size"], g["mean"], marker="o", markersize=3.0,
                    linewidth=1.2, color=color, label=mlabel)
            sd = g["std"].fillna(0.0)
            if (sd > 0).any():
                ax.fill_between(g["n_train_size"], g["mean"] - sd, g["mean"] + sd,
                                color=color, alpha=0.13, linewidth=0)
        if isl in base and np.isfinite(base[isl]):
            ax.axhline(base[isl], color=SEMANTIC_COLORS["reference"], linewidth=0.9, linestyle="--")
        ax.set_title(INTERNAL_TO_NAME.get(isl, str(isl)), fontsize=9, pad=3)
        ax.margins(y=0.10)
        style_axes(ax)
    for ax in axes[len(islands):]:
        ax.set_visible(False)
    for ax in axes[len(islands) - ncols: len(islands)]:
        ax.set_xlabel(r"Training-set size $k$")
    for r in range(nrows):
        axes[r * ncols].set_ylabel("Pearson $r$")

    handles = [plt.Line2D([0], [0], marker="o", color=c, linewidth=1.2, markersize=4, label=l)
               for _, l, c in methods]
    handles.append(plt.Line2D([0], [0], color=SEMANTIC_COLORS["reference"], linestyle="--",
                              label="Full source pool"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=len(handles), frameon=False, fontsize=8)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        p = output_dir / f"{stem}_{trait}.{suffix}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths[0], paths[1]


def load_pevmean_ga_runtime(repo_root: Path) -> pd.DataFrame:
    """Recorded PEVmean-GA selection time from the GA-only E3 runs."""
    frames = []
    for trait, _, _ in TRAITS:
        csv = repo_root / FINAL / "e3_pevmean_ga" / trait / "pevmean_ga_results.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["trait"] = trait
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_similarity_benchmark_problem(repo_root: Path, trait: str = "body_mass"):
    """Load SNPs and GRM in the same order for timing the cheap selectors."""
    import pyreadr

    data = np.load(repo_root / TRAIT_NPZ[trait], allow_pickle=False)
    X = data["snp"]
    ids = data["ids"].astype(str)
    locality = data["locality"].astype(str)
    locality = np.where(locality == "68", "67", locality)
    unique, counts = np.unique(locality, return_counts=True)
    keep_codes = set(unique[counts >= 20])
    keep = np.isin(locality, list(keep_codes))
    X, ids, locality = X[keep], ids[keep], locality[keep]

    grm_df = next(iter(pyreadr.read_r(str(repo_root / GRM_RDS)).values()))
    grm_df.index = grm_df.index.astype(str)
    grm_df.columns = grm_df.columns.astype(str)
    present = np.array([i in grm_df.index for i in ids], dtype=bool)
    X, ids, locality = X[present], ids[present], locality[present]
    grm = grm_df.loc[ids, ids].to_numpy(dtype=np.float64)
    island_name = np.array([ISLAND_ID_TO_NAME[c] for c in locality])
    return X.astype(np.float32, copy=False), grm, ids, island_name


def benchmark_similarity_selection_runtime(repo_root: Path, output_dir: Path,
                                           trait: str = "body_mass",
                                           max_k: int = 4500,
                                           pca_distance_pcs: int = 20,
                                           force: bool = False) -> pd.DataFrame:
    """Time ranking construction for the cheap selectors on one representative trait."""
    cache = output_dir / "selection_similarity_runtime_benchmark.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)

    import sys
    from sklearn.decomposition import PCA

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.avggrm_weighting import avg_grm_train_to_target, greedy_avggrm_diversity_order

    X, grm, ids, island_name = _load_similarity_benchmark_problem(repo_root, trait)
    n_components = int(min(max(3, pca_distance_pcs), X.shape[0], X.shape[1]))
    t0 = time.perf_counter()
    pc_scores = PCA(n_components=n_components, svd_solver="randomized", random_state=14).fit_transform(X)
    pca_fit_seconds = float(time.perf_counter() - t0)
    pca_distance_pcs = int(min(pca_distance_pcs, pc_scores.shape[1]))

    rows = []
    target_names = sorted(set(island_name))
    pca_share = pca_fit_seconds / max(1, len(target_names))
    for target in target_names:
        target_idx = np.flatnonzero(island_name == target)
        source_idx = np.flatnonzero(island_name != target)
        k_eff = int(min(max_k, source_idx.size))

        t0 = time.perf_counter()
        avg_grm = avg_grm_train_to_target(grm, source_idx, target_idx)
        _ = np.argsort(-avg_grm, kind="mergesort")[:k_eff]
        elapsed = float(time.perf_counter() - t0)
        rows.append({
            "trait": trait,
            "target_island_name": target,
            "method": "AvgGRM top-k",
            "seconds": elapsed,
            "pca_fit_seconds": pca_fit_seconds,
            "max_k": k_eff,
        })

        t0 = time.perf_counter()
        target_centroid = pc_scores[target_idx, :pca_distance_pcs].mean(axis=0)
        pc_distance = np.linalg.norm(
            pc_scores[source_idx, :pca_distance_pcs].astype(np.float64)
            - target_centroid.astype(np.float64)[None, :],
            axis=1,
        )
        _ = np.argsort(pc_distance, kind="mergesort")[:k_eff]
        elapsed = float(time.perf_counter() - t0)
        rows.append({
            "trait": trait,
            "target_island_name": target,
            "method": "PC distance",
            "seconds": elapsed + pca_share,
            "ranking_seconds": elapsed,
            "pca_fit_seconds": pca_fit_seconds,
            "max_k": k_eff,
        })

        t0 = time.perf_counter()
        avg_grm = avg_grm_train_to_target(grm, source_idx, target_idx)
        _ = greedy_avggrm_diversity_order(
            avg_grm_to_target=avg_grm,
            train_train_grm=grm[np.ix_(source_idx, source_idx)],
            lambda_div=1.0,
            max_size=k_eff,
            include_diagonal=True,
        )["order"][:k_eff]
        elapsed = float(time.perf_counter() - t0)
        rows.append({
            "trait": trait,
            "target_island_name": target,
            "method": r"AvgGRM diversity ($\lambda=1$)",
            "seconds": elapsed,
            "pca_fit_seconds": pca_fit_seconds,
            "max_k": k_eff,
        })

    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def plot_selection_runtime(repo_root: Path, output_dir: Path,
                           stem: str = "selection_runtime"):
    """Runtime comparison for PEVmean-GA and the cheap ranking rules."""
    pev = load_pevmean_ga_runtime(repo_root)
    cheap = benchmark_similarity_selection_runtime(repo_root, output_dir)
    if pev.empty or cheap.empty:
        raise FileNotFoundError("Missing runtime inputs for selection-runtime plot.")

    pev_summary = (
        pev.groupby("n_individuals")["ga_elapsed_sec"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("n_individuals")
    )
    method_order = ["AvgGRM top-k", "PC distance", r"AvgGRM diversity ($\lambda=1$)"]
    cheap_data = [
        cheap.loc[cheap["method"] == method, "seconds"].dropna().to_numpy()
        for method in method_order
    ]

    configure_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    ax.plot(pev_summary["n_individuals"], pev_summary["mean"],
            marker="o", markersize=4.5, linewidth=1.6,
            color=SEMANTIC_COLORS["adjusted"], label="Mean")
    ax.fill_between(
        pev_summary["n_individuals"],
        pev_summary["mean"] - pev_summary["std"],
        pev_summary["mean"] + pev_summary["std"],
        color=SEMANTIC_COLORS["adjusted"],
        alpha=0.16,
        linewidth=0,
        label="SD",
    )
    ax.set_xlabel(r"Selected subset size $k$")
    ax.set_ylabel("PEVmean-GA elapsed time (s)")
    ax.set_title("Model-based selection")
    style_axes(ax)

    ax = axes[1]
    bp = ax.boxplot(
        cheap_data,
        tick_labels=["AvgGRM", "PC dist.", "AvgGRM div."],
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="0.15", linewidth=1.0),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    colors = [TRAIT_COLORS["Tarsus length"], TRAIT_COLORS["Body mass"], SEMANTIC_COLORS["observed"]]
    for box, color in zip(bp["boxes"], colors):
        box.set(facecolor=color, edgecolor=color, alpha=0.65)
    ax.set_yscale("log")
    ax.set_ylabel("Ranking time per target island (s)")
    ax.set_title("Similarity-based rankings")
    style_axes(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        p = output_dir / f"{stem}.{suffix}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths[0], paths[1]


def write_per_island_table(repo_root: Path, output_dir: Path, delta_df: pd.DataFrame | None = None,
                           trait: str = "body_mass", stem: str = "selection_per_island_table") -> Path:
    """LaTeX table: per target island, full-pool r, best-subset r, gain, best k (one trait)."""
    if delta_df is None:
        delta_df = delta_r_per_island(repo_root)
    sub = delta_df[delta_df["trait"] == trait].copy()

    frames = []
    for t in ("body_mass", "thr_tarsus", "thr_wing"):
        f = (repo_root / FINAL / "e1_pc_ridge_nested_loio_10" / t
             / "e1_pc_ridge_nested_loio_10_per_fold_results.csv")
        if f.exists():
            d = pd.read_csv(f)
            if "trait" not in d or d["trait"].isna().all():
                d["trait"] = t
            frames.append(d)
    e1 = pd.concat(frames, ignore_index=True)
    e1 = e1[e1["trait"] == trait]
    ntest = dict(zip(e1["test_island_code"], e1["n_test"]))

    sub["n_test"] = sub["target_island"].map(ntest)
    sub = sub.sort_values("delta_r", ascending=False)

    rows = [
        f"{r.island_name} & {int(r.n_test)} & {r.full_pool_r:.3f} & "
        f"{r.best_subset_r:.3f} & {r.delta_r:+.3f} & {int(r.best_k)} \\\\"
        for r in sub.itertuples(index=False)
    ]
    table = "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        (r"\textbf{Target island} & \textbf{$n_{\mathrm{test}}$} & "
         r"\textbf{$r_{\mathrm{full}}$} & \textbf{$r_{\mathrm{best}}$} & "
         r"\textbf{$\Delta r$} & \textbf{$k^{\star}$} \\"),
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption[Per-island subset-selection gain]{For each body-mass target island, "
         r"the table shows the full-source baseline accuracy $r_{\mathrm{full}}$, the best "
         r"AvgGRM-diversity accuracy $r_{\mathrm{best}}$ obtained by choosing $k$ after evaluation, "
         r"the gain $\Delta r = r_{\mathrm{best}} - r_{\mathrm{full}}$, and the subset size "
         r"$k^{\star}$ at which the best accuracy is reached. Islands are sorted by $\Delta r$.}"),
        r"\label{tab:selection_per_island}",
        r"\end{table}",
        "",
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.tex"
    path.write_text(table, encoding="utf-8")
    return path


def main() -> None:
    repo_root = find_repo_root()
    out = repo_root / "figures"
    delta_df = delta_r_per_island(repo_root)
    delta_df.to_csv(out / "selection_per_island_delta_r.csv", index=False)
    plot_per_island_delta_r(delta_df, out)
    plot_subset_boxplots(repo_root, out)
    for trait in ("body_mass", "thr_tarsus", "thr_wing"):
        plot_subset_curves_by_island(repo_root, out, trait)
    plot_selection_runtime(repo_root, out)
    plot_pc1_vs_latitude(repo_root, out)
    plot_selection_pca(repo_root, out)
    plot_selection_signature(repo_root, out)
    print(f"Wrote selection diagnostics to {out}")
    print(delta_df.groupby("trait_label")["delta_r"].describe()[["mean", "min", "max"]])


if __name__ == "__main__":
    main()
