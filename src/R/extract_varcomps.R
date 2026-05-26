## Re-fit the LMM behind y_adjusted for each trait and print variance components.
## Reuses existing PLINK het output from Data/gnn/plink_het_subset.het.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(lme4)
})

setwd_to_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- normalizePath(sub("^--file=", "", file_arg[1]))
    setwd(dirname(script_path))
  }
}
setwd_to_script_dir()
source("adjust_support.R")

infile  <- "../../Data/AdultMorphology_20240201_fix.csv"
het_file <- "../../Data/gnn/plink_het_subset.het"

traits <- c("body_mass", "thr_tarsus", "thr_wing")

dd_full <- fread(infile, sep = ";", data.table = FALSE)

F_df <- read_plink_het(het_file = het_file, id_from = "FID",
                       center = TRUE, scale = FALSE)

results <- list()

for (phen in traits) {
  cat(sprintf("\n========== %s ==========\n", phen))
  dd <- dd_full[!is.na(dd_full[[phen]]), , drop = FALSE] |>
    mutate(
      ringnr     = as.character(ringnr),
      sex        = factor(ifelse(adult_sex == 1, "m",
                                 ifelse(adult_sex == 2, "f", NA))),
      month      = factor(as.integer(month)),
      locality   = factor(locality),
      hatch_year = as.integer(as.character(hatch_year)),
      max_year   = as.integer(as.character(max_year)),
      age        = max_year - hatch_year
    ) |>
    filter(!is.na(sex), !is.na(month), !is.na(age),
           !is.na(locality), !is.na(hatch_year)) |>
    left_join(F_df, by = "ringnr") |>
    filter(!is.na(F_hat))

  cat(sprintf("N observations: %d; N individuals: %d\n",
              nrow(dd), dplyr::n_distinct(dd$ringnr)))

  fit_res <- fit_lmm_and_adjust(dd, phenotype = phen, include_F = TRUE)
  fit <- fit_res$fit

  vc <- VarCorr(fit)
  sigma2_resid <- attr(vc, "sc")^2
  v_df <- as.data.frame(vc)
  v_df <- v_df[, c("grp", "vcov")]

  sigma2_ringnr     <- v_df$vcov[v_df$grp == "ringnr"]
  sigma2_locality   <- v_df$vcov[v_df$grp == "locality"]
  sigma2_hatch_year <- v_df$vcov[v_df$grp == "hatch_year"]

  total_random <- sigma2_ringnr + sigma2_locality + sigma2_hatch_year + sigma2_resid
  h2_narrow    <- sigma2_ringnr / total_random

  cat(sprintf("sigma2_ringnr  (V_A)     = %.6f\n", sigma2_ringnr))
  cat(sprintf("sigma2_locality          = %.6f\n", sigma2_locality))
  cat(sprintf("sigma2_hatch_year        = %.6f\n", sigma2_hatch_year))
  cat(sprintf("sigma2_resid   (sigma_e2)= %.6f\n", sigma2_resid))
  cat(sprintf("h2_narrow (V_A / V_P)    = %.4f\n", h2_narrow))

  results[[phen]] <- list(
    sigma2_ringnr     = sigma2_ringnr,
    sigma2_locality   = sigma2_locality,
    sigma2_hatch_year = sigma2_hatch_year,
    sigma2_resid      = sigma2_resid,
    h2_narrow         = h2_narrow,
    n_obs             = nrow(dd),
    n_id              = dplyr::n_distinct(dd$ringnr)
  )
}

cat("\n\n========== SUMMARY (for config files) ==========\n")
for (phen in traits) {
  r <- results[[phen]]
  cat(sprintf("%-12s  va_apriori = %.4f   sigma_e2_apriori = %.4f   (h2 = %.3f)\n",
              phen, r$sigma2_ringnr, r$sigma2_resid, r$h2_narrow))
}
