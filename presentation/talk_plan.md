# 10 minute talk plan

Working title: **Training set optimization for genomic prediction across island populations**

Audience assumption: statistics master students and supervisors. They know regression, regularization, CV, and prediction, but not genetics.

## Core message

Genomic prediction is high-dimensional regularized prediction. It becomes harder when the target population is not exchangeable with the training population. My thesis asks whether choosing or weighting the training set can improve prediction across island populations.

## Content and timing

| Time | Section | Purpose |
| --- | --- | --- |
| 0:00-0:30 | Title | Frame the thesis as target-aware prediction under population shift. |
| 0:30-1:40 | SNPs and phenotypes | Explain DNA variation, SNPs as 0/1/2 marker columns, and phenotypes as measured responses. |
| 1:40-3:00 | Genomic prediction | Move from the SNP design matrix to LMM language and marker-based ridge/BPCRR. |
| 3:00-4:00 | When genomic prediction gets hard | Explain prediction in the wild: across population, leave-one-island-out, and island structure. |
| 4:00-4:50 | Training set optimization | Define the target-aware training problem: choose or weight source individuals for a target island. |
| 4:50-5:40 | Data Shapley | Explain marginal contribution, why it is useful, and why it is expensive/calibration-dependent. |
| 5:40-6:50 | Top-k selection | Show performance over training set size. Main point: an optimum may exist, but it is hard to pick a priori. |
| 6:50-7:35 | Nested CV | Explain how target-aware rules are tuned without using the held-out island phenotype. |
| 7:35-8:35 | Importance weighting | Show the Bayes/domain-classifier density-ratio idea and summarize the nested-CV results. |
| 8:35-10:00 | Reflections | What worked, what did not, likely reasons, and future work. |

## Speaking notes

- Keep biology minimal: SNP = one genomic column; phenotype = measured trait response.
- Translate "island" to "population group" when addressing the statisticians.
- Say explicitly that Pearson correlation is the held-out-island prediction metric.
- Data Shapley is a diagnostic and partly oracle-like because it uses calibration target phenotypes.
- The honest current result: genotype similarity helps, top-k has promising structure, but strict nested weighting has not clearly beaten strong full-source baselines yet.

## One-sentence takeaway

The thesis is about replacing "use all available training data equally" with target-aware selection or weighting, and the preliminary evidence suggests useful signal in genetic similarity but no robust magic rule yet.
