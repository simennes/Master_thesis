"""One-off script to update visualize_bpcrr_inla.ipynb to:
  - Point TMC_INDIVIDUAL_DIR to the new tmc_shapley_individuals_grouped output.
  - Replace Figure 1c (cell 10) with a ridge-only mean-over-completed-islands plot.
  - Replace Figure 2 (cell 14) with a ridge-only grid restricted to completed islands.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "visualize_bpcrr_inla.ipynb"

# Islands fully completed by the grouped TMC run.
COMPLETED = [0, 3, 4, 5, 7, 8, 9, 11, 12]

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# --- Cell 2: switch TMC_INDIVIDUAL_DIR to grouped output ---
cell2_src = "".join(nb["cells"][2]["source"])
old_tmc = 'TMC_INDIVIDUAL_DIR = PROJECT_ROOT / "outputs" / "tmc_shapley_individuals"  # set to None to disable'
new_tmc = 'TMC_INDIVIDUAL_DIR = PROJECT_ROOT / "outputs" / "tmc_shapley_individuals_grouped"  # set to None to disable'
assert old_tmc in cell2_src, "Could not find TMC dir line in cell 2"
cell2_new = cell2_src.replace(old_tmc, new_tmc)
nb["cells"][2]["source"] = cell2_new.splitlines(keepends=True)

# --- Cell 9 markdown: update title ---
nb["cells"][9]["source"] = [
    "## Figure 1c: Ridge-only mean corr_eval over completed TMC target islands\n"
]

# --- Cell 10: replace Figure 1c with new averaged-over-completed ridge-only plot ---
cell10 = f"""# Figure 1c: Ridge-only mean corr_eval averaged over completed TMC target islands.
# Restricts to islands where the grouped TMC run has all 6 repeats finished.

TMC_COMPLETED_ISLANDS = {COMPLETED}

RIDGE_ONLY_MODELS_1c = [
    "ridge_pca",
    "ridge_avggrm",
    "ridge_pev",
    "ridge_tmc_individual",
]

ridge_only_1c = combined.copy()
ridge_only_1c = ridge_only_1c[
    ridge_only_1c["model"].isin(RIDGE_ONLY_MODELS_1c)
    & ridge_only_1c["target_island"].isin(TMC_COMPLETED_ISLANDS)
].copy()

if "analysis" in ridge_only_1c.columns:
    ranked_or_untyped = ridge_only_1c[
        (ridge_only_1c["analysis"] == "ranked_subset") | (ridge_only_1c["analysis"].isna())
    ]
    if len(ranked_or_untyped):
        ridge_only_1c = ranked_or_untyped.copy()

ridge_only_1c = ridge_only_1c[ridge_only_1c["method"] != "full_source_unweighted"].copy()
if "order_seed" in ridge_only_1c.columns:
    ridge_only_1c = ridge_only_1c[ridge_only_1c["order_seed"] != -2].copy()

# Drop tmc_shapley_positive marker rows from the curves (shown only in Figure 2).
ridge_only_1c = ridge_only_1c[ridge_only_1c["method"] != "tmc_shapley_positive"].copy()

# Keep random_individual only from ridge_tmc_individual to avoid duplicates.
RANDOM_KEEP_MODELS_1c = {{"ridge_tmc_individual"}}
mask_drop_rand = (
    (ridge_only_1c["method"] == "random_individual")
    & (~ridge_only_1c["model"].isin(RANDOM_KEEP_MODELS_1c))
)
ridge_only_1c = ridge_only_1c[~mask_drop_rand].copy()

if ridge_only_1c.empty:
    raise ValueError("No ridge rows for the completed TMC subset plot")

# Mean over islands and repeats: first average per (model, method, island, n) to
# get one value per island, then average across islands so each island contributes
# equally regardless of the number of repeats.
per_island = (
    ridge_only_1c.groupby(["model", "method", "target_island", "n_individuals"], as_index=False)["corr_eval"]
    .mean()
)
ridge_plot_df_1c = (
    per_island.groupby(["model", "method", "n_individuals"], as_index=False)["corr_eval"]
    .mean()
    .sort_values(["model", "method", "n_individuals"])
)
ridge_plot_df_1c["curve"] = ridge_plot_df_1c["model"] + " | " + ridge_plot_df_1c["method"].astype(str)

curve_order_1c = (
    ridge_plot_df_1c[["curve", "model", "method"]]
    .drop_duplicates()
    .sort_values(["model", "method"])
    ["curve"]
    .tolist()
)
palette_1c = sns.color_palette("tab10", n_colors=max(len(curve_order_1c), 1))
curve_colors_1c = dict(zip(curve_order_1c, palette_1c))

plt.figure(figsize=(12, 6))
for curve in curve_order_1c:
    sub = ridge_plot_df_1c[ridge_plot_df_1c["curve"] == curve]
    is_tmc = "tmc_individual" in curve
    lw = 2.5 if is_tmc else 2.0
    plt.plot(
        sub["n_individuals"],
        sub["corr_eval"],
        marker="o",
        linewidth=lw,
        alpha=1.0 if is_tmc else 0.85,
        color=curve_colors_1c[curve],
        label=curve,
    )

ridge_refs_1c = full_source_refs[
    full_source_refs["model"].isin(RIDGE_ONLY_MODELS_1c)
    & full_source_refs["target_island"].isin(TMC_COMPLETED_ISLANDS)
].copy()
if not ridge_refs_1c.empty:
    ridge_ref_mean_1c = (
        ridge_refs_1c.groupby(["model", "target_island"], as_index=False)["corr_eval"]
        .mean()
        .groupby("model", as_index=False)["corr_eval"]
        .mean()
        .sort_values("model")
    )
    for _, row in ridge_ref_mean_1c.iterrows():
        ref_label = f"{{row['model']}} | full_source_unweighted"
        ref_color = curve_colors_1c.get(ref_label, "#555555")
        plt.axhline(
            row["corr_eval"],
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            color=ref_color,
            label=ref_label,
        )

# TMC full-source reference (SNP subset, all source individuals).
if not tmc_individual.empty:
    tmc_full = tmc_individual[
        (tmc_individual["method"] == "full_source_unweighted")
        & (tmc_individual["target_island"].isin(TMC_COMPLETED_ISLANDS))
    ]
    if not tmc_full.empty:
        tmc_full_mean = (
            tmc_full.groupby("target_island")["corr_eval"].mean().mean()
        )
        plt.axhline(
            tmc_full_mean,
            linestyle=":",
            linewidth=1.6,
            alpha=0.9,
            color="#8c564b",
            label="ridge_tmc_individual | full_source (SNP subset)",
        )

n_islands_used = ridge_only_1c["target_island"].nunique()
plt.title(f"Body mass: ridge-only mean corr_eval, averaged over {{n_islands_used}} completed TMC islands")
plt.xlabel("n_individuals")
plt.ylabel("corr_eval")
plt.grid(True, alpha=0.25)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Ridge curve")
plt.tight_layout()
plt.show()

ridge_curve_summary_1c = (
    ridge_plot_df_1c.groupby("curve", as_index=False)["corr_eval"]
    .agg(max_corr="max", mean_corr="mean")
    .sort_values("max_corr", ascending=False)
)
display(ridge_curve_summary_1c)
"""
nb["cells"][10]["source"] = cell10.splitlines(keepends=True)

# --- Cell 13 markdown: update title ---
nb["cells"][13]["source"] = [
    "## Figure 2: Ridge-only per-island grid (completed TMC islands)\n"
]

# --- Cell 14: replace Figure 2 with ridge-only grid for completed islands ---
cell14 = f"""# Figure 2: ridge-only grid, one panel per completed TMC target island.

TMC_COMPLETED_ISLANDS = {COMPLETED}

RIDGE_ONLY_MODELS_F2 = [
    "ridge_pca",
    "ridge_avggrm",
    "ridge_pev",
    "ridge_tmc_individual",
]

panel_df = combined.copy()
panel_df = panel_df[
    panel_df["model"].isin(RIDGE_ONLY_MODELS_F2)
    & panel_df["target_island"].isin(TMC_COMPLETED_ISLANDS)
].copy()

if "analysis" in panel_df.columns:
    ranked_or_untyped = panel_df[(panel_df["analysis"] == "ranked_subset") | (panel_df["analysis"].isna())]
    if len(ranked_or_untyped):
        panel_df = ranked_or_untyped.copy()

panel_df = panel_df[panel_df["method"] != "full_source_unweighted"].copy()
if "order_seed" in panel_df.columns:
    panel_df = panel_df[panel_df["order_seed"] != -2].copy()

panel_df = panel_df[panel_df["method"] != "tmc_shapley_positive"].copy()

# Keep random only from ridge_tmc_individual.
RANDOM_KEEP_F2 = {{"ridge_tmc_individual"}}
mask_drop_rand = (
    (panel_df["method"] == "random_individual")
    & (~panel_df["model"].isin(RANDOM_KEEP_F2))
)
panel_df = panel_df[~mask_drop_rand].copy()

panel_plot_df = (
    panel_df.groupby(["target_island", "model", "method", "n_individuals"], as_index=False)["corr_eval"]
    .mean()
    .sort_values(["target_island", "model", "method", "n_individuals"])
)
panel_plot_df["curve"] = panel_plot_df["model"] + " | " + panel_plot_df["method"].astype(str)

if panel_plot_df.empty:
    raise ValueError("No ridge rows for per-island grid")

curve_order_f2 = (
    panel_plot_df[["curve", "model", "method"]]
    .drop_duplicates()
    .sort_values(["model", "method"])
    ["curve"]
    .tolist()
)
palette_f2 = sns.color_palette("tab10", n_colors=max(len(curve_order_f2), 1))
curve_colors_f2 = dict(zip(curve_order_f2, palette_f2))

# Try to use real island names if available.
island_name_map = {{}}
for col in ["target_island_name", "target_name", "island_name", "target_island_label"]:
    if col in combined.columns:
        name_df = combined[["target_island", col]].dropna().drop_duplicates()
        if not name_df.empty:
            island_name_map = dict(zip(name_df["target_island"].astype(int), name_df[col].astype(str)))
            break


def island_title(island_id: int) -> str:
    n = island_name_map.get(int(island_id))
    return n if n else f"Target island {{island_id}}"


islands = sorted(panel_plot_df["target_island"].unique().tolist())
n = len(islands)
n_cols = min(4, n)
n_rows = int(np.ceil(n / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), sharey=True)
axes = np.array(axes).reshape(n_rows, n_cols)

panel_full_source_refs = full_source_refs.copy()
panel_full_source_refs = panel_full_source_refs[
    panel_full_source_refs["model"].isin(RIDGE_ONLY_MODELS_F2)
    & panel_full_source_refs["target_island"].isin(TMC_COMPLETED_ISLANDS)
]

for i, island in enumerate(islands):
    r, c = divmod(i, n_cols)
    ax = axes[r, c]
    sub = panel_plot_df[panel_plot_df["target_island"] == island]

    for curve in curve_order_f2:
        s = sub[sub["curve"] == curve]
        if s.empty:
            continue
        is_tmc = "tmc_individual" in curve
        ax.plot(
            s["n_individuals"],
            s["corr_eval"],
            marker="o",
            linewidth=2.5 if is_tmc else 1.8,
            alpha=1.0 if is_tmc else 0.85,
            color=curve_colors_f2[curve],
            label=curve,
        )

    # Per-model full-source hlines.
    fs_island = panel_full_source_refs[panel_full_source_refs["target_island"] == island]
    if not fs_island.empty:
        fs_model = fs_island.groupby("model", as_index=False)["corr_eval"].mean()
        for _, row in fs_model.iterrows():
            ref_label = f"{{row['model']}} | full_source_unweighted"
            ax.axhline(
                row["corr_eval"],
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                color=curve_colors_f2.get(ref_label, "#555555"),
                label=ref_label,
            )

    # TMC full-source baseline for this island (SNP subset).
    if not tmc_individual.empty:
        tmc_full_isl = tmc_individual[
            (tmc_individual["method"] == "full_source_unweighted")
            & (tmc_individual["target_island"] == island)
        ]
        if not tmc_full_isl.empty:
            ax.axhline(
                tmc_full_isl["corr_eval"].mean(),
                linestyle=":",
                linewidth=1.4,
                alpha=0.8,
                color="#8c564b",
                label="ridge_tmc_individual | full_source (SNP subset)",
            )

    ax.set_title(island_title(island))
    ax.set_xlabel("n_individuals")
    if c == 0:
        ax.set_ylabel("corr_eval")
    ax.grid(True, alpha=0.25)

for j in range(n, n_rows * n_cols):
    r, c = divmod(j, n_cols)
    axes[r, c].axis("off")

# Collect legend handles from any populated axis.
handles, labels = [], []
for ax in axes.flatten():
    h, l = ax.get_legend_handles_labels()
    if l:
        handles, labels = h, l
        break
if handles:
    uniq = {{}}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    fig.legend(
        list(uniq.values()),
        list(uniq.keys()),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Ridge curve",
    )

fig.suptitle("Body mass: corr_eval per completed TMC target island (ridge methods only)", y=1.02)
fig.tight_layout(rect=[0, 0, 0.86, 1])
plt.show()
"""
nb["cells"][14]["source"] = cell14.splitlines(keepends=True)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"Updated {NB_PATH}")
