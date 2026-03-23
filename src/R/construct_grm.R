library(data.table)

# ---------- INPUT ----------
plink <- "\"C:/Users/Simen/OneDrive - NTNU/FYSMAT/INDMAT/25H/Prosjekt/PLINK/plink.exe\""
chrset <- 32
maf_min <- 0.01
geno_max <- 0.10
mind_max <- 0.05

pheno_file <- "Data/AdultMorphology_20240201_fix.csv"
bfile_raw  <- "Data/combined_200k_70k_sparrow_genotype_data/combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05"
trait_cols <- c("body_mass", "thr_tarsus", "thr_wing")

# Choose islands like in INLA example (set FALSE to keep all)
filter_isls <- FALSE
isls <- NA
n_min <- 20  # minimum samples per island after phenotype filtering

out_prefix <- "Data/GRM"
out_rds    <- "Data/GRM/GRM_vanraden.rds"
qc_subset_dir <- "Data/qc_trait_subset"

# ---------- LOAD PHENOS ----------
ph <- fread(pheno_file)
ph[, ringnr := as.character(ringnr)]
if (filter_isls) ph <- ph[locality %in% isls]

missing_trait_cols <- setdiff(trait_cols, names(ph))
if (length(missing_trait_cols) > 0) {
  stop("Missing trait columns in phenotype file: ", paste(missing_trait_cols, collapse = ", "))
}

# Keep individuals with at least one non-missing trait measurement.
has_any_trait <- ph[, rowSums(!is.na(.SD)) > 0, .SDcols = trait_cols]
ph <- ph[has_any_trait]

if (n_min > 1) {
  if (!"locality" %in% names(ph)) stop("Column 'locality' is required when n_min > 1.")
  isl_counts <- ph[, .N, by = locality]
  keep_localities <- isl_counts[N >= n_min, locality]
  ph <- ph[locality %in% keep_localities]
}

if (nrow(ph) == 0) {
  stop("No phenotyped individuals remain after applying trait/island filters.")
}

# ---------- PREP FAM (mirror INLA behavior) ----------
fam <- fread(paste0(bfile_raw, ".fam"), header = FALSE)
setnames(fam, c("V1","V2"), c("FID","IID"))   # FID=ringnr, IID=DNA ID (encoded)
fam[, FID := as.character(FID)]
fam[, IID := as.character(IID)]

# drop flagged samples
fam_keep <- fam[!grepl("HIGHHET|MISSEX", IID)]

# keep the *last* DNA ID per ringnr (as in the INLA code)
fam_keep <- fam_keep[!duplicated(fam_keep$FID, fromLast = TRUE)]

# ---------- 1) Overall QC ----------
dir.create("Data/qc_overall_", showWarnings = FALSE)
fwrite(fam_keep[, .(FID, IID)], "Data/qc_overall_/keep.txt",
       col.names = FALSE, sep = "\t")

cmd1 <- paste(
  plink,
  "--bfile", bfile_raw,
  "--keep Data/qc_overall_/keep.txt",
  "--maf", maf_min,
  "--geno", geno_max,
  "--mind", mind_max,
  "--chr-set", chrset,
  "--make-bed --freq",
  "--threads 8 --memory 48000",
  "--out Data/qc_overall_/qc"
)
stopifnot(system(cmd1) == 0)
bfile_qc_all <- "Data/qc_overall_/qc"

# ---------- 2) Trait/system subset QC ----------
dir.create(qc_subset_dir, showWarnings = FALSE)

# Keep only phenotyped ringnr among *cleaned, deduplicated* fam_keep
keep_pairs <- fam_keep[FID %in% unique(ph$ringnr), .(FID, IID)]
fwrite(unique(keep_pairs), file.path(qc_subset_dir, "keep.txt"),
       col.names = FALSE, sep = "\t")

cmd2 <- paste(
  plink,
  "--bfile", bfile_qc_all,
  "--keep", file.path(qc_subset_dir, "keep.txt"),
  "--maf", maf_min,
  "--geno", geno_max,
  "--mind", mind_max,
  "--chr-set", chrset,
  "--make-bed --freq",
  "--threads 8 --memory 48000",
  "--out", file.path(qc_subset_dir, "qc")
)
stopifnot(system(cmd2) == 0)
bfile_qc_sub <- file.path(qc_subset_dir, "qc")
frq_file     <- file.path(qc_subset_dir, "qc.frq")

# ---------- 3) Make GRM ----------
dir.create(dirname(out_prefix), recursive = TRUE, showWarnings = FALSE)
cmd3 <- paste(
  plink,
  "--bfile", bfile_qc_sub,
  "--keep", file.path(qc_subset_dir, "keep.txt"),
  "--maf", maf_min,
  "--geno", geno_max,
  "--mind", mind_max,
  "--chr-set", chrset,
  "--read-freq", frq_file,           # use the same freqs
  "--make-rel square bin cov",
  "--make-just-bim",                 # SNP list actually used
  "--threads 8 --memory 48000",
  "--out", out_prefix
)
stopifnot(system(cmd3) == 0)

# ---------- 4) Read + VanRaden scale ----------
rel_bin <- paste0(out_prefix, ".rel.bin")
rel_id  <- paste0(out_prefix, ".rel.id")
bim     <- paste0(out_prefix, ".bim")

id_tab <- fread(rel_id, header = FALSE)
ids <- id_tab$V1   # just ringnr (FID)
n <- length(ids)

M <- matrix(readBin(rel_bin, "numeric", n^2), nrow = n, byrow = TRUE)
rownames(M) <- ids
colnames(M) <- ids


snps   <- fread(bim, select = 2)
frq    <- fread(frq_file)
frq_inc <- frq[SNP %in% snps$V2]
stopifnot(nrow(frq_inc) > 0)

GRM <- M * (nrow(frq_inc) - 1) / (2 * sum(frq_inc$MAF * (1 - frq_inc$MAF)))

# ---------- SAVE ----------
dir.create(dirname(out_rds), recursive = TRUE, showWarnings = FALSE)
saveRDS(GRM, out_rds)
cat("Saved GRM with", n, "individuals to", out_rds, "\n")
