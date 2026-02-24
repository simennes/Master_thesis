# TMC Data Shapley Summary (Target Island 0: Nesøy)

## Run metadata

| Field | Value |
|---|---:|
| target_island | 0 |
| target_island_name | Nesøy |
| v_full (all source islands) | 0.200503 |
| n_permutations | 50 |
| n_utility_evals | 609 |
| cache_hit_rate | 0.1891 |
| eps_trunc | 0.005 |
| use_truncation | false |
| min_prefix_islands | 3 |
| island_sampling_mode | mean |
| cap_per_island | null |
| n_train_seeds_per_eval | 1 |

## Global Shapley diagnostics

| Metric | Value |
|---|---:|
| Sum of island phi values | 0.200503 |
| Positive-contribution islands | 9 |
| Negative-contribution islands | 6 |
| Sum of positive phi | 0.405441 |
| Sum of negative phi | -0.204938 |

Interpretation: phi values sum to v_full, as expected.

## Island-level Shapley values (sorted by phi)

| Rank | Source island | n_island | phi | phi_per_individual |
|---:|---|---:|---:|---:|
| 1 | Hestmannøy | 1078 | 0.156573 | 0.00014524 |
| 2 | Gjerøy | 558 | 0.113988 | 0.00020428 |
| 3 | Southern 3 | 584 | 0.036806 | 0.00006302 |
| 4 | Sleneset | 166 | 0.032140 | 0.00019361 |
| 5 | Southern 2 | 397 | 0.026712 | 0.00006728 |
| 6 | Aldra | 173 | 0.016071 | 0.00009290 |
| 7 | Indre Kvarøy | 356 | 0.009220 | 0.00002590 |
| 8 | Selvær | 194 | 0.008372 | 0.00004315 |
| 9 | Southern 4 | 486 | 0.005559 | 0.00001144 |
| 10 | Myken | 92 | -0.008977 | -0.00009758 |
| 11 | Lovund | 158 | -0.016620 | -0.00010519 |
| 12 | Southern 5 | 202 | -0.018513 | -0.00009165 |
| 13 | Southern 1 | 583 | -0.041615 | -0.00007138 |
| 14 | Træna | 292 | -0.046887 | -0.00016057 |
| 15 | Onøy og Lurøy | 259 | -0.072325 | -0.00027925 |

## Add curve: Shapley order vs random baseline (aggregated over random order seeds)

Random baseline columns are mean/min/max correlation over random order seeds.

| n_islands included | shapley_corr | random_corr_mean | random_corr_min | random_corr_max | shapley_mse | random_mse_mean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.219070 | -0.039855 | -0.175862 | 0.219070 | 1.393396 | 1.407103 |
| 2 | 0.252411 | 0.030748 | -0.144071 | 0.232265 | 1.360300 | 1.423799 |
| 3 | 0.298813 | 0.020373 | -0.172852 | 0.176113 | 1.384397 | 1.440633 |
| 5 | 0.301374 | 0.114463 | -0.077495 | 0.238199 | 1.384918 | 1.450826 |
| 7 | 0.335871 | 0.114858 | -0.027252 | 0.240989 | 1.330106 | 1.447074 |
| 10 | 0.301097 | 0.168088 | 0.006542 | 0.273631 | 1.351928 | 1.442757 |
| 12 | 0.315272 | 0.200389 | 0.119872 | 0.274323 | 1.384418 | 1.464862 |
| 15 | 0.200503 | 0.200503 | 0.200503 | 0.200503 | 1.484637 | 1.484637 |

## Remove curve: Shapley removal vs random baseline (aggregated over random order seeds)

Random baseline columns are mean/min/max correlation over random order seeds.

| n_islands remaining | shapley_corr | random_corr_mean | random_corr_min | random_corr_max | shapley_mse | random_mse_mean |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.200503 | 0.200503 | 0.200503 | 0.200503 | 1.484637 | 1.484637 |
| 14 | 0.224110 | 0.203910 | 0.138875 | 0.251170 | 1.477444 | 1.477827 |
| 12 | 0.315272 | 0.211536 | 0.154275 | 0.275601 | 1.384418 | 1.474011 |
| 9 | 0.334331 | 0.172458 | -0.038644 | 0.304923 | 1.368872 | 1.485103 |
| 6 | 0.310226 | 0.150286 | -0.016582 | 0.293819 | 1.387654 | 1.471265 |
| 3 | 0.298813 | 0.065349 | -0.101897 | 0.231051 | 1.384397 | 1.439371 |
| 1 | 0.219070 | -0.054145 | -0.207540 | 0.240828 | 1.393396 | 1.410344 |