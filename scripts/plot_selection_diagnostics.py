"""Subset-selection diagnostic figures for the thesis Results section.

Produces:
  * per_island_delta_r       - gain of the best selected subset over the full pool, per island
  * gain_vs_isolation (N1)   - that gain against the target island's relatedness to the source pool
  * diversity_vs_k    (N2)   - internal relatedness of the selected subset, AvgGRM vs diversity
  * avggrm_vs_shapley (N3)   - island-level AvgGRM vs island Shapley value

Run directly:  python scripts/plot_selection_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

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

# Selection methods (exclude the random baseline) used for the "best subset" gain.
SELECTION_METHODS = [
    "pevmean_ga_pc_ridge",
    "avggrm_topk",
    "pca_target_topk",
    "avggrm_diversity_lam0p25",
    "avggrm_diversity_lam0p5",
    "avggrm_diversity_lam1",
    "avggrm_diversity_lam2",
]

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


def delta_r_per_island(repo_root: Path) -> pd.DataFrame:
    """Gain of the best selected subset over the full pool, per (trait, island).

    best_subset_r = max over selection methods and subset sizes of the
    repeat-averaged Pearson r; full_pool_r is the E1 full-source baseline.
    """
    res = load_subset_results(repo_root)
    sel = res[res["method"].isin(SELECTION_METHODS)].copy()
    per_k = (
        sel.groupby(["trait", "target_island", "method", "n_train_size"], as_index=False)["pearson_r"]
        .mean()
    )
    best = (
        per_k.groupby(["trait", "target_island"], as_index=False)["pearson_r"]
        .max()
        .rename(columns={"pearson_r": "best_subset_r"})
    )
    # subset size at which the best value occurs
    idx = per_k.groupby(["trait", "target_island"])["pearson_r"].idxmax()
    best_k = per_k.loc[idx, ["trait", "target_island", "n_train_size"]].rename(
        columns={"n_train_size": "best_k"}
    )
    best = best.merge(best_k, on=["trait", "target_island"])
    base = load_full_pool_baseline(repo_root)
    out = best.merge(base, on=["trait", "target_island"], how="left")
    out["delta_r"] = out["best_subset_r"] - out["full_pool_r"]
    out["island_name"] = out["target_island"].map(INTERNAL_TO_NAME)
    out["trait_label"] = out["trait"].map(TRAIT_LABELS)
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
    ax.set_xlabel(r"$\Delta r$ (best selected subset $-$ full source pool)")
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
    ax.set_ylabel(r"$\Delta r$ (best subset $-$ full pool)")
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
    ("random_pc_ridge", "Random", "#4C78A8"),
    ("avggrm_topk", "AvgGRM top-k", "#59A14F"),
    ("pca_target_topk", "PC-distance top-k", "#F28E2B"),
    ("pevmean_ga_pc_ridge", "PEV-mean GA", "#E45756"),
    ("avggrm_diversity_lam1", r"AvgGRM diversity ($\lambda=1$)", "#9C755F"),
]


def plot_subset_boxplots(repo_root: Path, output_dir: Path,
                         stem: str = "e3_subset_boxplots"):
    """Per-trait grouped boxplots of Pearson r by method at each subset size k.

    Three stacked panels (one per trait); at each k the methods are drawn as
    small boxplots side by side, summarising the distribution across the 15
    target islands (and repeats). The dashed line is the full-source baseline.
    """
    res = load_subset_results(repo_root)
    base = load_full_pool_baseline(repo_root).groupby("trait")["full_pool_r"].mean()

    methods = [m for m in SUBSET_METHOD_STYLE if m[0] in set(res["method"])]
    k_values = sorted(
        k for k in res["n_train_size"].unique()
        if all(not res[(res.method == m[0]) & (res.n_train_size == k)].empty for m in methods)
    )
    n_methods = len(methods)
    width = 0.8 / n_methods

    configure_thesis_style()
    fig, axes = plt.subplots(len(TRAITS), 1, figsize=(FULL_WIDTH, 7.6),
                             sharex=True, constrained_layout=True)
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
                            medianprops=dict(color="0.15", linewidth=0.9),
                            whiskerprops=dict(color=color, linewidth=0.8),
                            capprops=dict(color=color, linewidth=0.8))
            for box in bp["boxes"]:
                box.set(facecolor=color, edgecolor=color, alpha=0.55, linewidth=0.6)
        if trait in base.index:
            ax.axhline(base[trait], color=SEMANTIC_COLORS["reference"],
                       linewidth=1.0, linestyle="--")
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_ylabel("Pearson $r$")
        ax.set_title(lab)
        style_axes(ax)
    axes[-1].set_xlabel(r"Training-set size $k$")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="none", markersize=8,
                          markerfacecolor=c, markeredgecolor=c, alpha=0.65, label=l)
               for _, l, c in methods]
    handles.append(plt.Line2D([0], [0], color=SEMANTIC_COLORS["reference"],
                              linestyle="--", label="Full source pool"))
    axes[0].legend(handles=handles, frameon=False, ncol=2, loc="lower right", fontsize=8)
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


def main() -> None:
    repo_root = find_repo_root()
    out = repo_root / "figures"
    delta_df = delta_r_per_island(repo_root)
    delta_df.to_csv(out / "selection_per_island_delta_r.csv", index=False)
    plot_per_island_delta_r(delta_df, out)
    plot_avggrm_vs_shapley(repo_root, out)
    plot_subset_boxplots(repo_root, out)
    plot_pc1_vs_latitude(repo_root, out)
    print(f"Wrote selection diagnostics to {out}")
    print(delta_df.groupby("trait_label")["delta_r"].describe()[["mean", "min", "max"]])


if __name__ == "__main__":
    main()
