# Master thesis — code repository

NB: This readme was mainly produced by the AI tool Claude Code.

Code accompanying the master's thesis on genomic prediction in house sparrows
(*Passer domesticus*) under across-population and within-population settings.

Every figure in the thesis is produced by `notebooks/thesis_figures.ipynb`,
which reads pre-computed results from `outputs/final_results/`. The notebook
`notebooks/compare_avggrm_diversity_greedy_vs_bqp.ipynb` produces the
diversity-GA-vs-SA runtime comparison.

---

## Layout

```
config/         Per-experiment JSON configs, grouped by experiment family
  across_pop/   Ridge across-population baseline
  bpcrr/        BPCRR-INLA variants (full-source, LOIO v2/v3, importance-from-ridge)
  e1/           PC ridge LOIO baselines (nested + fixed-l10)
  e3/           PEVmean-GA subset selection
  e4/           Similarity-based subset selection (avgGRM, PC-distance, diversity)
  e5/           Island-level Data-Shapley
  e6/           PC ridge nested LOIO with avgGRM / importance weighting
  e8/           MLP and PC-MLP nested LOIO with weighting knobs
  within_pop/   Within-population subset experiments

src/            Library modules (see "Library overview" below)
  SLURM/        SLURM submission scripts, one per experiment
  training_set_optimization/  PEVmean and GA helpers shared across E3/E4

scripts/        Entry-point scripts and post-hoc plotters
notebooks/      Final analysis notebooks (only 2 are kept; everything else
                that lived here was exploratory and has been removed)

outputs/
  final_results/  All result CSV/JSON files consumed by the thesis figures.
                  Sub-folder per experiment family (e1, e3, e4, e5, e6, e8,
                  bpcrr_*, ridge_across_pop_alpha1e5, within_pop_*).

Data/           SNP genotypes, GRM, adjusted phenotypes (not version-controlled).
```

---

## How to reproduce the experiments

Each experiment is launched via SLURM with one config per (trait, variant).
The general pattern is

```bash
sbatch --export=ALL,CONFIG_PATH=<config> src/SLURM/<job>.slurm
```

| Thesis section | Submit | Configs | Output |
|---|---|---|---|
| Across-population ridge | `src/SLURM/final_e1_pc_ridge_nested_loio.slurm` (uses `src/SLURM/final_e3_random_pc_ridge_*.slurm` analogue) | `config/across_pop/` | `outputs/final_results/ridge_across_pop_alpha1e5/` |
| BPCRR baselines | `src/SLURM/final_bpcrr_inla_full_source_1000pc.slurm`, `final_bpcrr_inla_loio_v2.slurm`, `final_bpcrr_inla_loio_v3.slurm` | `config/bpcrr/` | `outputs/final_results/bpcrr_inla_*` |
| E1 — PC ridge LOIO | `src/SLURM/final_e1_pc_ridge_nested_loio.slurm` | `config/e1/` | `outputs/final_results/e1_pc_ridge_nested_loio_10/`, `e1_pc_ridge_fixed_loio_l10/` |
| E3 — PEVmean GA | `final_e3_pevmean_ga.slurm`, then `final_e3_pevmean_selected_pc_ridge*.slurm` + `final_e3_random_pc_ridge*.slurm` | `config/e3/` | `e3_pevmean_ga/`, `e3_pevmean_ga_pc_ridge/`, `e3_random_pc_ridge/` |
| E4 — Similarity subsets | (run locally) `python -m scripts.run_e4_similarity_pc_ridge --config config/e4/final_e4_similarity_pc_ridge_config.json` | `config/e4/` | `e4_similarity_pc_ridge/` |
| E5 — Island Shapley | `src/SLURM/final_e5_shapley_islands_pc_ridge.slurm` (array, one trait per task) | `config/e5/` | `e5_shapley_islands_pc_ridge/` |
| E6 — PC ridge weighted | `src/SLURM/final_e6_pc_ridge_nested_loio.slurm` (one sbatch per config) | `config/e6/` | `e6_pc_ridge_avggrm/`, `e6_pc_ridge_importance/` |
| E6 importance-from-ridge BPCRR | `src/SLURM/final_bpcrr_inla_loio_v3_importance_from_e6_ridge_m5.slurm` (depends on E6 importance output) | `config/bpcrr/` | `bpcrr_inla_loio_v3_importance_from_e6_ridge_m5/` |
| E8 — MLP / PC-MLP | `src/SLURM/final_e8_mlp_nested_loio.slurm` (one sbatch per config) | `config/e8/` | `e8_mlp_uniform/`, `e8_mlp_importance/`, `e8_pc_mlp_uniform/` |
| Within-population subsets | `within_pop_pevmean_subsets.slurm`, `within_pop_avggrm_random_subsets.slurm`, `within_pop_pc_ridge_10fold.slurm` | `config/within_pop/` | `within_pop_subset_pevmean/`, `within_pop_subset_avggrm_random/`, `within_pop_pc_ridge_10fold/` |
| E6 importance reselection | `src/SLURM/posthoc_reselect_importance.slurm` (depends on E6 importance output) | `config/e6/` | augments `e6_pc_ridge_importance/` |

Bulk-submit all E6 jobs:

```bash
for cfg in config/e6/final_e6_pc_ridge_*_config.json; do
  sbatch --export=ALL,CONFIG_PATH=$cfg src/SLURM/final_e6_pc_ridge_nested_loio.slurm
done
```

Same pattern for E8 (`config/e8/`) and BPCRR (`config/bpcrr/`).

To regenerate the E6 / E8 configs after editing search-space defaults:

```bash
python scripts/_gen_e6_e8_configs.py
```

---

## Building the thesis figures

After all `outputs/final_results/` are populated, open
`notebooks/thesis_figures.ipynb` and run all cells. The notebook writes PDFs
and PNGs to the LaTeX source directory next to this repo (path resolved by
`find_repo_root`).

Diagnostic CSVs (per-fold tables, summary tables) used by the notebook are
produced by these auxiliary scripts:

```
scripts/plot_e1_pc_ridge_diagnostics.py        -> E1 diagnostics
scripts/plot_e6_avggrm_thesis_figures.py       -> E6 avgGRM diagnostics
scripts/plot_e6_importance_thesis_figures.py   -> E6 importance diagnostics
scripts/plot_e8_pc_mlp_uniform_thesis_figures.py -> E8 PC-MLP uniform diagnostics
scripts/plot_e5_shapley_thesis_figures.py      -> E5 Shapley figures + tables
scripts/plot_avggrm_diversity_ga_sa.py         -> diversity GA-vs-SA runtime
scripts/plot_across_pop_diagnostics.py         -> across-population summaries
scripts/plot_descriptive_results.py            -> phenotype descriptives
scripts/plot_selection_diagnostics.py          -> generic selection plots
```

Each writes into the relevant `outputs/final_results/<experiment>/` directory.

---

## Library overview (`src/`)

Cross-cutting modules:

* `data.py` — load SNP NPZ + GRM + phenotype, apply min-count and island filters.
* `utils.py` — RNG seeding, Pearson, decode helpers, SNP top-k by abs-corr.
* `cv_utils.py` — island label resolution, inner LOIO split planner.
* `pc_feature.py` — common "fit PCA once, use top-k PCs as features" helper used
  by E6 / E8.
* `models.py` / `training.py` — MLP architecture + training loop (uniform and
  sample-weighted variants).
* `hyperparams.py` — Optuna suggest helpers for the MLP search space.

Method-specific:

* `avggrm_weighting.py` — avgGRM scoring, ranking, weight schemes, greedy
  avgGRM-diversity selection.
* `importance_weighting.py` — pc-logistic covariate-shift importance weights,
  effective sample size, Optuna suggest helpers.
* `removal_curve.py` — target-island split helpers used by E5 and the
  TMC-Shapley driver.
* `tmc_shapley_islands.py` — island-level Truncated-Monte-Carlo Shapley
  (utility on calibration set, add/remove curves on held-out test).
* `nested_cv_avggrm_weighted.py` — weighted training loop helpers shared by
  unified and MLP runners.
* `nested_cv_avggrm_weighted_unified.py` — unified nested-CV runner for ridge /
  PC ridge / BPCRR with avgGRM-scheme weighting.
* `nested_cv_avggrm_weighted_ridge.py` — thin wrapper that forces
  `model.type='ridge'` for E6 avgGRM runs.
* `nested_cv_importance_weighted_ridge.py` — nested-CV ridge / PC ridge runner
  with covariate-shift importance weighting.
* `nested_cv_importance_weighted_mlp.py` — nested-CV MLP / PC-MLP runner with
  three `weighting_mode` choices (uniform / avggrm / importance) and a
  `feature_pca` knob.
* `nested_cv_importance_weighted_bpcrr_from_ridge.py` — BPCRR using a fixed
  importance-weight prescription transferred from the E6 ridge importance run.
* `training_set_optimization/` — PEVmean (marker-form) and GA scaffolding used
  by E3 and E4.

---

## Reproducibility notes

* Every random seed is set from `base_train.seed` (or `seed` at the top of the
  config). Outer-fold-specific seeds are derived deterministically.
* PCA on the full filtered genotype matrix is *unsupervised* and therefore
  carries no phenotype leakage; all PC-based experiments (E1 fixed, E3, E4, E5,
  E6 PC-ridge, E8 PC-MLP) use this convention. E1 nested LOIO refits PCA per
  outer fold for the strictest setting.
* Results are written incrementally — the nested-CV runners persist the
  partial results JSON after each completed outer fold so a job that hits the
  wall-clock limit still saves what it finished.

---

## Figure style

A single style module is used everywhere:

```python
from scripts.thesis_style import (
    configure_thesis_style, style_axes, TRAIT_COLORS, SEMANTIC_COLORS, PALETTE
)
```

The thesis notebook applies this style once at the top. Conventions:

* Font: serif (Times New Roman) with STIX math.
* Axes: seaborn whitegrid, no top/right spines, faint grid.
* Trait colours (fixed across all figures): body mass `#4C78A8`,
  tarsus length `#59A14F`, wing length `#F28E2B`.
* Two-series accent (e.g. observed vs adjusted): blue `#4C78A8` / red `#E45756`.
* PDF export at 600 dpi; full-width figures are 6.7 in wide.
