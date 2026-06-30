"""Generate descriptive-data figures and tables for the thesis Results section."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from thesis_style import (
        SEMANTIC_COLORS,
        TRAIT_COLORS,
        configure_thesis_style,
        style_axes,
    )
except ModuleNotFoundError:  # pragma: no cover - package-style import
    from scripts.thesis_style import (
        SEMANTIC_COLORS,
        TRAIT_COLORS,
        configure_thesis_style,
        style_axes,
    )


@dataclass(frozen=True)
class TraitSpec:
    key: str
    label: str
    unit: str
    npz_path: str


TRAITS = [
    TraitSpec("body_mass", "Body mass", "g", "Data/npz/snp_body_mass_ALL.npz"),
    TraitSpec("thr_tarsus", "Tarsus length", "mm", "Data/npz/snp_thr_tarsus_ALL.npz"),
    TraitSpec("thr_wing", "Wing length", "mm", "Data/npz/snp_thr_wing_ALL.npz"),
]

MORPHOLOGY_PATH = "Data/AdultMorphology_20240201_fix.csv"


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "Data").exists() and (path / "src").exists():
            return path
        candidate = path / "Master_thesis"
        if (candidate / "Data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not find the Master_thesis repo root.")


def _experiment_metadata(npz_path: Path, min_count: int = 20) -> pd.DataFrame:
    data = np.load(npz_path, allow_pickle=False)
    ids = data["ids"].astype(str)
    y_adjusted = data["y_adjusted"].astype(float)
    y_mean = data["y_mean"].astype(float)
    locality = data["locality"].astype(str)

    unique, counts = np.unique(locality, return_counts=True)
    keep = set(unique[counts >= min_count])
    mask = np.isin(locality, list(keep))

    return pd.DataFrame(
        {
            "ringnr": ids[mask],
            "y_adjusted": y_adjusted[mask],
            "y_mean": y_mean[mask],
            "locality": np.where(locality[mask] == "68", "67", locality[mask]),
        }
    )


def load_descriptive_data(repo_root: Path) -> pd.DataFrame:
    """Per-trait descriptive summary (sample counts, raw/adjusted spread, correlation)."""
    summary_rows: list[dict[str, object]] = []

    for trait in TRAITS:
        meta = _experiment_metadata(repo_root / trait.npz_path)
        pheno = pd.read_csv(
            repo_root / "Data" / f"adjusted_{trait.key}.csv",
            dtype={"ringnr": str},
        )
        obs_map = pheno.drop_duplicates("ringnr").set_index("ringnr")["n_obs"]
        n_obs = obs_map.reindex(meta["ringnr"]).fillna(0).astype(int)

        summary_rows.append(
            {
                "trait": trait.key,
                "trait_label": trait.label,
                "unit": trait.unit,
                "n_individuals": int(len(meta)),
                "n_records": int(n_obs.sum()),
                "raw_mean": float(meta["y_mean"].mean()),
                "raw_sd": float(meta["y_mean"].std(ddof=1)),
                "adjusted_sd": float(meta["y_adjusted"].std(ddof=1)),
                "raw_adjusted_r": float(meta["y_mean"].corr(meta["y_adjusted"])),
                "median_records_per_individual": float(n_obs.median()),
            }
        )

    return pd.DataFrame(summary_rows)


def load_sex_by_ringnr(repo_root: Path) -> pd.Series:
    """Return one sex label ("Male"/"Female") per genotyped individual."""
    morph = pd.read_csv(repo_root / MORPHOLOGY_PATH, sep=";", dtype={"ringnr": str})
    morph = morph[morph["adult_sex"].isin([1, 2])]
    sex_code = morph.groupby("ringnr")["adult_sex"].agg(lambda s: s.mode().iloc[0])
    return sex_code.map({1: "Male", 2: "Female"})


def load_sex_distribution_data(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw per-individual means labelled by sex, plus a per-trait sex summary.

    The summary reports the sex gap in the raw mean, the share of raw
    between-individual variance explained by sex, and the correlation of the
    adjusted phenotype with the raw mean before and after removing the sex
    mean. The latter isolates the effect of the sex fixed effect in the
    phenotype-adjustment model.
    """
    sex = load_sex_by_ringnr(repo_root)
    distribution_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for trait in TRAITS:
        meta = _experiment_metadata(repo_root / trait.npz_path)
        meta["sex"] = meta["ringnr"].map(sex)
        meta = meta.dropna(subset=["sex"]).copy()

        grand = meta["y_mean"].mean()
        group_mean = meta.groupby("sex")["y_mean"].transform("mean")
        ss_between = float(((group_mean - grand) ** 2).sum())
        ss_total = float(((meta["y_mean"] - grand) ** 2).sum())
        meta["y_mean_within_sex"] = meta["y_mean"] - group_mean

        mean_male = float(meta.loc[meta["sex"] == "Male", "y_mean"].mean())
        mean_female = float(meta.loc[meta["sex"] == "Female", "y_mean"].mean())

        summary_rows.append(
            {
                "trait": trait.key,
                "trait_label": trait.label,
                "unit": trait.unit,
                "n_male": int((meta["sex"] == "Male").sum()),
                "n_female": int((meta["sex"] == "Female").sum()),
                "mean_male": mean_male,
                "mean_female": mean_female,
                "sex_gap": mean_male - mean_female,
                "var_explained_by_sex": ss_between / ss_total,
                "r_raw_adj": float(meta["y_mean"].corr(meta["y_adjusted"])),
                "r_within_sex_adj": float(meta["y_mean_within_sex"].corr(meta["y_adjusted"])),
            }
        )

        raw = meta[["y_mean", "sex"]].rename(columns={"y_mean": "value"})
        raw["trait"] = trait.key
        raw["trait_label"] = trait.label
        raw["unit"] = trait.unit
        distribution_rows.append(raw)

    distribution_df = pd.concat(distribution_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    return distribution_df, summary_df


def plot_phenotype_by_sex(
    distribution_df: pd.DataFrame,
    output_dir: Path,
    file_stem: str = "phenotype_by_sex",
) -> tuple[Path, Path]:
    """Raw per-individual mean distributions split by sex, one panel per trait."""
    configure_thesis_style()
    colors = {"Male": SEMANTIC_COLORS["observed"], "Female": SEMANTIC_COLORS["adjusted"]}
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(TRAITS),
        figsize=(6.7, 3.4),
        constrained_layout=False,
    )

    legend_handles = []
    for idx, (ax, trait) in enumerate(zip(axes, TRAITS)):
        subset = distribution_df[distribution_df["trait"] == trait.key]
        for sex_label in ["Male", "Female"]:
            sns.kdeplot(
                data=subset[subset["sex"] == sex_label],
                x="value",
                fill=True,
                color=colors[sex_label],
                alpha=0.35,
                linewidth=1.1,
                label=sex_label,
                ax=ax,
            )
        ax.set_title(trait.label)
        ax.set_xlabel(f"Individual mean ({trait.unit})")
        ax.set_ylabel("Density" if trait is TRAITS[0] else "")
        if idx == 0:
            legend_handles = ax.get_legend_handles_labels()[0]
        style_axes(ax)

    fig.suptitle("Observed phenotype distributions by sex", y=0.985)
    fig.legend(
        legend_handles, ["Male", "Female"],
        loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=2, frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.27, top=0.80, wspace=0.45)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for suffix in ["pdf", "png"]:
        path = output_dir / f"{file_stem}.{suffix}"
        fig.savefig(path)
        written_paths.append(path)
    plt.close(fig)
    return written_paths[0], written_paths[1]


def plot_phenotype_overview(
    repo_root: Path,
    output_dir: Path,
    file_stem: str = "phenotype_overview",
) -> tuple[Path, Path]:
    """Per-trait overview: observed and adjusted distributions plus their scatter.

    Rows are the three traits; columns are the observed individual-mean
    distribution, the adjusted-phenotype distribution, and a scatter of the two
    with the Pearson correlation annotated. Each row is drawn in its trait
    colour, since the observed/adjusted distinction is already carried by the
    columns.
    """
    configure_thesis_style()
    ref_color = SEMANTIC_COLORS["reference"]

    fig, axes = plt.subplots(
        nrows=len(TRAITS),
        ncols=3,
        figsize=(6.7, 5.9),
        constrained_layout=False,
    )

    for row_idx, trait in enumerate(TRAITS):
        trait_color = TRAIT_COLORS[trait.label]
        meta = _experiment_metadata(repo_root / trait.npz_path)
        observed = meta["y_mean"]
        adjusted = meta["y_adjusted"]
        r = float(observed.corr(adjusted))

        ax_obs = axes[row_idx, 0]
        sns.histplot(
            x=observed, bins=38, stat="density", color=trait_color,
            edgecolor="white", linewidth=0.25, alpha=0.78, ax=ax_obs,
        )
        sns.kdeplot(x=observed, color="0.18", linewidth=1.0, ax=ax_obs)
        ax_obs.axvline(observed.mean(), color=ref_color, linewidth=0.8, linestyle="--")
        ax_obs.set_xlabel(f"Observed mean ({trait.unit})")
        ax_obs.set_ylabel(f"{trait.label}\nDensity")
        if row_idx == 0:
            ax_obs.set_title("Observed phenotype")
        ax_obs.tick_params(axis="both")
        style_axes(ax_obs)

        ax_adj = axes[row_idx, 1]
        sns.histplot(
            x=adjusted, bins=38, stat="density", color=trait_color,
            edgecolor="white", linewidth=0.25, alpha=0.78, ax=ax_adj,
        )
        sns.kdeplot(x=adjusted, color="0.18", linewidth=1.0, ax=ax_adj)
        ax_adj.axvline(adjusted.mean(), color=ref_color, linewidth=0.8, linestyle="--")
        ax_adj.set_xlabel(f"Adjusted ({trait.unit})")
        ax_adj.set_ylabel("Density")
        if row_idx == 0:
            ax_adj.set_title("Adjusted phenotype")
        ax_adj.tick_params(axis="both")
        style_axes(ax_adj)

        ax_sc = axes[row_idx, 2]
        ax_sc.scatter(
            observed, adjusted, s=5, alpha=0.15, color=trait_color,
            edgecolor="none", rasterized=True,
        )
        ax_sc.set_xlabel(f"Observed ({trait.unit})")
        ax_sc.set_ylabel(f"Adjusted ({trait.unit})")
        if row_idx == 0:
            ax_sc.set_title("Observed vs. adjusted")
        ax_sc.text(
            0.05, 0.93, f"$r = {r:.3f}$", transform=ax_sc.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", lw=0.5),
        )
        ax_sc.tick_params(axis="both")
        style_axes(ax_sc)

    fig.suptitle("Observed and adjusted phenotypes", y=0.985)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.075, top=0.88, hspace=0.55, wspace=0.36)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for suffix in ["pdf", "png"]:
        path = output_dir / f"{file_stem}.{suffix}"
        fig.savefig(path)
        written_paths.append(path)
    plt.close(fig)
    return written_paths[0], written_paths[1]


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_float(value: float) -> str:
    return f"{value:.3f}"


def write_summary_outputs(summary_df: pd.DataFrame, output_dir: Path, write_csv: bool = True) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_csv:
        summary_df.to_csv(output_dir / "descriptive_trait_summary.csv", index=False)

    table_rows = []
    for row in summary_df.itertuples(index=False):
        trait_label = f"{row.trait_label} ({row.unit})"
        table_rows.append(
            " & ".join(
                [
                    trait_label,
                    _fmt_int(int(row.n_individuals)),
                    _fmt_int(int(row.n_records)),
                    _fmt_float(float(row.raw_mean)),
                    _fmt_float(float(row.raw_sd)),
                    _fmt_float(float(row.adjusted_sd)),
                    _fmt_float(float(row.raw_adjusted_r)),
                ]
            )
            + r" \\"
        )

    table = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\renewcommand{\arraystretch}{1.08}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            (
                r"\textbf{Trait} & \textbf{$n_{\mathrm{ind}}$} & "
                r"\textbf{$n_{\mathrm{rec}}$} & \textbf{Raw mean} & "
                r"\textbf{Raw SD} & \textbf{Adjusted SD} & "
                r"\textbf{$r_{\mathrm{raw,adj}}$} \\"
            ),
            r"\midrule",
            *table_rows,
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption[Descriptive phenotype summary]{Trait-level descriptive summary after "
                r"aligning phenotypes to the genotyped individuals used in the experiments. "
                r"The raw phenotype is the per-individual mean over capture records, and the adjusted "
                r"phenotype is the individual-level value from the phenotype-adjustment model.}"
            ),
            r"\label{tab:descriptive_trait_summary}",
            r"\end{table}",
            "",
        ]
    )
    tex_path = output_dir / "descriptive_trait_summary.tex"
    tex_path.write_text(table, encoding="utf-8")
    return tex_path


def main() -> None:
    repo_root = find_repo_root()
    output_dir = repo_root / "figures"
    summary_df = load_descriptive_data(repo_root)
    plot_phenotype_overview(repo_root, output_dir)
    write_summary_outputs(summary_df, output_dir)

    sex_distribution_df, sex_summary_df = load_sex_distribution_data(repo_root)
    plot_phenotype_by_sex(sex_distribution_df, output_dir)
    sex_summary_df.to_csv(output_dir / "descriptive_sex_summary.csv", index=False)
    print(sex_summary_df.to_string(index=False))
    print(f"Wrote descriptive outputs to {output_dir}")


if __name__ == "__main__":
    main()
