# Thesis Results Plan
**Deadline: ~June 14, 2026 (1 month)**

---

## Timeline at a glance

| Week | Focus |
|------|-------|
| Week 1 (now) | Launch all slow jobs (Shapley, PEVmean-GA). Run fast baselines. |
| Week 2 | Launch medium jobs (nested CV, avggrm, pca-distance, weighting). Monitor Shapley. |
| Week 3 | All experiments done. Generate all figures. |
| Week 4 | Write Results + Discussion sections. Final polish. |

---

## PART 1 — EXPERIMENTS TO RUN

Trait abbreviations used throughout: **bm** = body_mass, **ta** = thr_tarsus, **wi** = thr_wing.
Island indexing: internal codes 0–14 (15 islands).

### 1.1  Baselines (FAST — do first)

- [ ] **Ridge LOIO baseline** — all 3 traits, all islands, full 65k SNPs
  - Script: `python -m scripts.run_ridge_loio --config config/ridge_loio_config.json`
  - Output: `outputs/nested_cv/ridge_*.csv`
  - Gives the denominator for all Δr comparisons; table + per-island bar chart.

- [ ] **Ridge LOIO baseline at 20k SNPs** — body_mass only
  - Same script, add `--num_snps 20000` or duplicate config with SNP selection on.
  - Needed for a fair baseline in the Shapley comparison figure.

### 1.2  Data Shapley — individual-level (SLOW — START NOW)

TMC is the bottleneck of the whole thesis. Start before anything else.

- [ ] **Shapley individuals — body_mass, all 15 islands**
  - Config: `config/tmc_shapley_individuals_grouped.json` (300 perms, group_size=10, 20k SNPs)
  - Script: `python -m scripts.run_tmc_shapley_individuals --config config/tmc_shapley_individuals_grouped.json`
  - Resume flag already set; can be interrupted and continued.
  - Output: `outputs/tmc_shapley_individuals_grouped/island_*/repeat_*/`

- [ ] **Shapley individuals — thr_tarsus** (if time allows; duplicate config, change npz/pheno paths)
  - At minimum get thr_tarsus since it has the highest heritability (h²=0.475) and will show clearest signal.
  - Use same 300 perms / 20k SNPs setup.

- [ ] **Shapley top-k evaluation** (run AFTER each island's Shapley output is ready)
  - Script: `python -m scripts.run_tmc_individual_topk_eval --config config/tmc_shapley_individuals_grouped.json --k_values 50 100 250 500 1000 1500 2000 2500 3000 --n_random_orders 5`
  - Output: `outputs/tmc_shapley_individuals_grouped/tmc_individual_topk_results.csv`
  - Also computes per-source-island average Shapley stats.

- [ ] **Positive-Shapley subset** — derive from top-k eval output
  - Filter to individuals with φ > 0; measure performance for this automatic cutoff.
  - No extra run needed — just a post-processing step on the top-k CSV.

### 1.3  PEVmean + Genetic Algorithm (MEDIUM-SLOW — START NOW)

- [ ] **PEVmean-GA — body_mass, all islands, multiple subset sizes**
  - Config: `config/pevmean_ga_config.json` (sizes 500–4500, 80 generations, pop=30)
  - Script: `python -m scripts.run_pevmean_ga --config config/pevmean_ga_config.json`
  - Output: `outputs/ridge_pev/`

- [ ] **PEVmean-GA — thr_tarsus** (duplicate config, change trait npz path)
  - Same sizes and GA settings.

- [ ] **PEVmean-GA at 20k SNPs — body_mass** (for fair Shapley comparison figure)
  - Duplicate config, enable `use_snp_selection: true, num_snps: 20000`.

### 1.4  AvgGRM top-k and diversity (FAST)

Run for all 3 traits once configs are in place. These are cheap and give the "simple baseline" story.

- [ ] **AvgGRM top-k — body_mass, all islands**
  - Config: `config/avggrm_rank_weight_config.json` (update target_islands to 0–14)
  - Script: `python -m scripts.run_avggrm_rank_weight --config config/avggrm_rank_weight_config.json`
  - Sizes: 50–4500 (already in config). 10 random repeats.
  - Output: `outputs/avggrm_rank_weight*/`

- [ ] **AvgGRM top-k — thr_tarsus and thr_wing** (update npz/pheno in duplicate configs)

- [ ] **AvgGRM diversity greedy — body_mass, all islands**
  - Config: `config/avggrm_diversity_greedy_config.json`
  - Script: `python -m scripts.run_avggrm_diversity_greedy --config config/avggrm_diversity_greedy_config.json`
  - This is the "top-k avggrm with diversity" strategy for the main comparison figure.

- [ ] **AvgGRM diversity greedy — thr_tarsus, thr_wing**

### 1.5  PCA-distance top-k (FAST)

- [ ] **PCA-distance (source PCA) — body_mass, all islands**
  - Config: `config/pca_source_rank_weight_config.json`
  - Script: `python -m scripts.run_pca_source_rank_weight --config config/pca_source_rank_weight_config.json`

- [ ] **PCA-distance (target PCA) — body_mass, all islands**
  - Config: `config/pca_target_rank_weight_config.json`
  - Choose one variant as the canonical one for the main figure; the other goes to appendix/robustness.

- [ ] **PCA-distance — thr_tarsus, thr_wing** (for both variants)

### 1.6  Nested CV — weighting schemes (MEDIUM — Ridge fast, MLP slow)

These produce the weighting-scheme boxplot and the "does weighted training help" question.

- [ ] **Nested CV — AvgGRM-weighted Ridge — body_mass**
  - Config: `config/nested_cv_avggrm_weighted_ridge_config.json`
  - Script: `python -m scripts.run_model_compare --config config/nested_cv_avggrm_weighted_ridge_config.json`
  - Tunes weighting scheme + Ridge alpha + optional SNP selection jointly.
  - 100 Optuna trials, full LOIO.

- [ ] **Nested CV — importance-weighted Ridge — body_mass**
  - Config: `config/nested_cv_importance_weighted_ridge_config.json`

- [ ] **Nested CV — importance-weighted MLP — body_mass**
  - Config: `config/nested_cv_importance_weighted_mlp_config.json`
  - Slowest of the nested CV jobs. Start early.

- [ ] **Nested CV — importance-weighted MLP (uniform baseline) — body_mass**
  - Config: `config/nested_cv_importance_weighted_mlp_uniform_config.json`

- [ ] **Nested CV — Ridge (unweighted baseline) — all 3 traits**
  - Config: `config/nested_cv_avggrm_weighted_ridge_config.json` with scheme fixed to "uniform"
  - Or use the plain ridge_loio output if the alpha tuning is equivalent.

- [ ] **Nested CV — thr_tarsus, thr_wing** for weighted Ridge (if compute allows)
  - At minimum body_mass must be complete; the other traits are bonus.

### 1.7  Island-level Shapley (OPTIONAL, lower priority)

- [ ] **TMC Shapley islands — body_mass**
  - Config: `config/tmc_shapley_islands.json`
  - Script: `python -m scripts.run_tmc_shapley_islands --config config/tmc_shapley_islands.json`
  - Produces island removal/addition curves; useful for the "island-level analysis" narrative.
  - Skip if time is tight — individual Shapley covers the main story.

### 1.8  TracIn (OPTIONAL — likely skip)

TracIn has unclear score distributions (noted in notebooks). Unless it stabilises, do not include in Results. If included, show only as a comparison to Shapley in the supplementary.

---

## PART 2 — FIGURES TO MAKE

Figures are grouped by Results subsection.

### 2.1  Data & Baseline Diagnostics

| # | Figure | Data source |
|---|--------|-------------|
| D1 | Geographic map of Helgeland islands (color = island ID) | Static — already in thesis figures folder |
| D2 | PCA of all individuals (color = island), PC1 vs PC2 | Compute from full genotype matrix |
| D3 | GRM relatedness heatmap (rows/cols = individuals, sorted by island) | GRM_vanraden.rds |
| D4 | Phenotype distributions: raw vs adjusted, all 3 traits (violin or density plots side-by-side) | pheno CSVs |
| D5 | Corr(y_adj, y_raw) per island per trait (scatter with identity line or bar of r values) | pheno CSVs |
| D6 | Sample sizes per island per trait (bar chart) | pheno CSVs |
| D7 | Ridge LOIO baseline correlation per island per trait (grouped bar chart, all 3 traits) | outputs/nested_cv/ridge_*.csv |
| D8 | Distribution of AvgGRM scores from source pool to each target island (one panel per target island, shows which source islands are most related) | GRM + island labels |

### 2.2  Main Comparison Figure (training set size vs. correlation)

| # | Figure | Data source |
|---|--------|-------------|
| M1 | **Learning curve** — x: training set size k, y: mean correlation — lines for: (a) random, (b) AvgGRM top-k diversity, (c) PCA-distance top-k, (d) PEVmean-GA, (e) Shapley top-k; averaged over target islands; **body_mass at 20k SNPs** | topk eval CSVs, pevmean output, avggrm output |
| M2 | Same as M1 but for thr_tarsus (shows whether pattern holds for more heritable trait) | same |
| M3 | Δr version of M1 (subtract full-source baseline at each k) | derived from M1 data |

### 2.3  Weighting Schemes

| # | Figure | Data source |
|---|--------|-------------|
| W1 | **Boxplot — weighting scheme comparison** (x: scheme = uniform / linear / minmax / exponential / top-heavy; y: correlation; one box per scheme across all islands) — Ridge | nested_cv/ridge_avggrm_weighted_nested.csv |
| W2 | Same for MLP | nested_cv/mlp_importance_weighted_nested.csv |
| W3 | Kish's effective sample size (n_eff) vs weight scheme (boxplot across islands) — shows cost of aggressive weighting | computed from weight outputs |
| W4 | Δr from weighting: weighted Ridge vs unweighted Ridge, per island (heatmap or bar) | nested CV outputs |

### 2.4  Per-Island Heterogeneity

| # | Figure | Data source |
|---|--------|-------------|
| H1 | **Δr heatmap** — rows = target islands, columns = methods; color = Δr; clearly shows which island benefits from which method | all method CSVs |
| H2 | Relationship between AvgGRM score of best source island and Δr (scatter) — does more relatedness → bigger gain? | avggrm + eval CSVs |
| H3 | Number of selected individuals per source island (stacked bar) for AvgGRM, PCA-distance, Shapley for 2-3 example target islands | selection CSVs |

### 2.5  Shapley-Specific Diagnostics

| # | Figure | Data source |
|---|--------|-------------|
| S1 | Distribution of Shapley values per source island (violin plot, one panel per target island) | tmc_individual_shap_per_source_island.csv |
| S2 | Average Shapley value per source island for each target island (heatmap: rows=target, cols=source) | same |
| S3 | Shapley add-curve vs random add-curve for 2–3 example target islands (x=k, y=correlation) | tmc_individual_topk_results.csv |
| S4 | Fraction of individuals with φ > 0 per target island and trait | Shapley CSVs |

### 2.6  Robustness Checks

| # | Figure | Data source |
|---|--------|-------------|
| R1 | k-sensitivity: AvgGRM top-k performance vs k (already in M1 but zoom in, CI shading) | avggrm output |
| R2 | Performance with 20k SNPs vs full 65k SNPs for AvgGRM and PCA-distance (paired bar or scatter) | run both SNP settings |
| R3 | Ranking consistency across target-split repeats (Spearman rank correlation of Shapley rankings, heatmap of repeat pairs) | repeat CSVs from Shapley |

---

## PART 3 — THESIS SECTIONS STATUS

### Results section (Sections/Results.tex)

- [ ] **2.1 Adjusted phenotype diagnostics** — fill with figures D4, D5, D6 and 2–3 sentences per trait
- [ ] **2.2 Baseline LOIO performance** — fill with figure D7, table of r per island/trait
- [ ] **2.3 Data Shapley results** — fill with figures S1–S4, M1 (Shapley line), discuss stability and cost
- [ ] **2.4 PEVmean-GA results** — fill with M1 (PEV line), discuss objective vs predictive gains
- [ ] **2.5 AvgGRM + PCA-distance results** — fill with M1 (AvgGRM + PCA lines), R1, R2
- [ ] **2.6 Weighting schemes** — fill with W1–W4
- [ ] **2.7 Comparative summary** — fill with H1, H2, M1 combined; table of best-method per island
- [ ] **2.8 Within-population** — brief: does selection still help when train/test from the same island mix?

### Methods section (Sections/Methods_v2.tex)

- [ ] Ridge implementation details (alpha search range, standardisation, weighted fit formula)
- [ ] MLP architecture (layer sizes, activation, dropout, batch norm, optimizer, Optuna search space)
- [ ] BPCRR — keep description but note not included in final evaluation (too slow)
- [ ] Weighting scheme parameter table (all 5 schemes with formulas and search ranges)
- [ ] SNP pre-selection procedure (random 20k, justification)
- [ ] Nested CV diagram or pseudocode (Figures/CV.tex already exists)

### Discussion section

- [ ] Which method family is most reliable across traits and islands?
- [ ] When does selection help vs. when is full pool better?
- [ ] Trade-off: computation cost vs. gain (Shapley expensive, AvgGRM cheap, similar gain?)
- [ ] Limitations: Shapley variance, GA non-determinism, phenotype adjustment sensitivity
- [ ] Future work: hybrid selectors, adaptive k, multi-trait joint optimisation

---

## PART 4 — COMPUTE PRIORITY QUEUE

Order to launch jobs given a single machine:

```
1.  NOW — run_tmc_shapley_individuals (grouped, body_mass) — will run for days
2.  NOW — run_pevmean_ga (body_mass, full 65k)
3.  Day 2 — run_ridge_loio (all 3 traits, full 65k) — fast
4.  Day 2 — run_avggrm_rank_weight (body_mass, all islands)
5.  Day 2 — run_avggrm_diversity_greedy (body_mass)
6.  Day 2 — run_pca_source_rank_weight (body_mass)
7.  Day 3 — run_model_compare / nested CV Ridge (body_mass, 100 trials)
8.  Day 3 — run_tmc_shapley_individuals (thr_tarsus) — if Shapley job 1 > 30% done
9.  Day 4 — run_pevmean_ga (thr_tarsus)
10. Day 4 — run_avggrm/pca configs for thr_tarsus + thr_wing
11. Day 5 — nested CV MLP (body_mass) — start early, takes longest of nested CV jobs
12. Day 6 — run_tmc_individual_topk_eval as soon as any island output is ready
13. Later — island-level Shapley (TMC islands) — only if time remains
```

---

## PART 5 — QUICK CONFIG GAPS TO FIX

Before running, check and fix these:

- [ ] Create `config/tmc_shapley_individuals_tarsus.json` — copy grouped config, update `npz` and `pheno` paths to thr_tarsus, update `output_dir`
- [ ] Create `config/tmc_shapley_individuals_wing.json` — same for thr_wing
- [ ] Update `config/pevmean_ga_config.json` — add thr_tarsus and thr_wing to `traits` list (or create separate configs)
- [ ] Update `config/avggrm_rank_weight_config.json` — ensure `target_islands` covers all 15 islands (0–14); add thr_tarsus, thr_wing variants
- [ ] Update `config/avggrm_diversity_greedy_config.json` — same island + trait coverage
- [ ] Update `config/pca_source_rank_weight_config.json` — all 15 islands, all 3 traits
- [ ] Create `config/ridge_loio_20k_config.json` — duplicate ridge_loio with `use_snp_selection: true, num_snps: 20000, trait: body_mass` for fair baseline
- [ ] Verify all 3 trait NPZ files exist: `Data/npz/snp_body_mass_ALL.npz`, `snp_thr_tarsus_ALL.npz`, `snp_thr_wing_ALL.npz`

---

## PART 6 — MINIMUM VIABLE RESULTS (if time runs out)

If forced to cut scope, keep these and drop the rest:

**Must have (non-negotiable):**
- Ridge LOIO baseline — all 3 traits
- Learning curve figure M1 — body_mass, 20k SNPs, all 5 methods
- Weighting boxplot W1 — Ridge, body_mass
- Δr heatmap H1 — body_mass
- Shapley diagnostics S1, S2 — body_mass

**Nice to have (add if time allows):**
- Same results for thr_tarsus and thr_wing
- MLP weighting (W2)
- Positive-Shapley subset analysis
- Island-level Shapley removal curves
- R2 (SNP count robustness)

**Drop if time is short:**
- TracIn results (unstable, not central to thesis)
- BPCRR in any nested CV (too slow, user already decided)
- Within-population secondary analysis (only if main across-pop story is complete)
