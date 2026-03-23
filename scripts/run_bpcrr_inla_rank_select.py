#!/usr/bin/env python
"""
Run BPCRR (Bayesian PCRR) experiments using R-INLA.

This script mirrors the existing rank/select experiment structure:
  1) Load genotype + phenotype + locality data.
  2) For each target island, define source candidates.
  3) Fit PCA once on the full candidate source set for that target.
  4) Reuse the same PC basis for all top-k subsets.
  5) Fit BPCRR with R-INLA on each subset and evaluate on target island.

Supported top-k selectors:
  - avggrm: rank by mean GRM similarity to target individuals (descending)
  - pc_distance: rank by Euclidean distance to target centroid in PC space (ascending)

The BPCRR model uses INLA's default prior for the PC-effect precision by not
overriding the latent-model hyper prior.

Usage
-----
Preflight (recommended on cluster before worker jobs):
    python -m scripts.run_bpcrr_inla_rank_select --mode preflight --config config/bpcrr_inla_config.json

Worker:
  python -m scripts.run_bpcrr_inla_rank_select --mode worker --config config/bpcrr_inla_config.json

Merge shards:
  python -m scripts.run_bpcrr_inla_rank_select --mode merge --config config/bpcrr_inla_config.json
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("rpy2").setLevel(logging.WARNING)

_RPY2_INIT_DONE = False


def _pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Pearson r; if either input has zero variance, return 0.0."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = np.sqrt((yt * yt).sum()) * np.sqrt((yp * yp).sum())
    if denom == 0.0:
        return 0.0
    return float((yt * yp).sum() / denom)


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))


def _configure_rpy2_startup() -> None:
    """Initialize rpy2/R with a clean startup to avoid workspace/profile side effects."""
    global _RPY2_INIT_DONE
    if _RPY2_INIT_DONE:
        return

    # Ensure R DLL directories are on PATH so embedded R can load base package DLLs.
    r_home = os.environ.get("R_HOME")
    if not r_home:
        try:
            from rpy2.situation import get_r_home

            r_home = get_r_home()
        except Exception:
            r_home = None

    if r_home:
        path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
        lower_entries = {p.lower() for p in path_entries}
        prepend_entries: List[str] = []
        for candidate in (Path(r_home) / "bin" / "x64", Path(r_home) / "bin"):
            candidate_str = str(candidate)
            if candidate.exists() and candidate_str.lower() not in lower_entries:
                prepend_entries.append(candidate_str)

        if prepend_entries:
            os.environ["PATH"] = os.pathsep.join(prepend_entries + path_entries)

    # Avoid loading user/site startup files and default .RData/.Rhistory state.
    # Keep R library path resolution intact; only block project/user profile scripts.
    os.environ.setdefault("R_PROFILE_USER", "")

    from rpy2.rinterface_lib import embedded

    if not embedded.isinitialized():
        # --vanilla guarantees no restore and no init/environ files.
        try:
            embedded.set_initoptions(("rpy2", "--vanilla", "--quiet"))
        except Exception as exc:
            raise RuntimeError(
                "Failed to configure embedded R startup options for rpy2. "
                "Cannot safely continue without clean R initialization."
            ) from exc

    _RPY2_INIT_DONE = True


def resolve_island_code(target_island, code_to_label: dict, present_codes: set) -> int:
    try:
        val = int(target_island)
    except (ValueError, TypeError):
        val = None

    if val is not None:
        if val in present_codes:
            return val
        for code, label in code_to_label.items():
            if int(label) == val:
                return code
        raise ValueError(f"Island {val} not found")

    if isinstance(target_island, str):
        target_lower = target_island.lower()
        for orig_label, name in ISLAND_ID_TO_NAME.items():
            if name.lower() == target_lower:
                for code, lbl in code_to_label.items():
                    if int(lbl) == orig_label:
                        return code
        for code, label in code_to_label.items():
            if str(label).lower() == target_lower:
                return code

    raise ValueError(f"Could not resolve target island: {target_island!r}")


def _resolve_training_islands(
    training_islands_config: Optional[List[Any]],
    code_to_label: dict,
    present_codes: set,
    target_code: int,
) -> Optional[List[int]]:
    if training_islands_config is None:
        return None
    result = []
    for island_ref in training_islands_config:
        code = resolve_island_code(island_ref, code_to_label, present_codes)
        if code != target_code:
            result.append(code)
    return result if result else None


def _build_trait_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    traits_cfg = cfg.get("traits", None)
    if traits_cfg is None:
        paths = dict(cfg["paths"])
        return [{
            "name": "default",
            "paths": paths,
            "target_column": cfg.get("target_column", "y_adjusted"),
            "eval_target_column": cfg.get("eval_target_column", "y_mean"),
            "standardize_features": cfg.get("standardize_features", True),
            "min_count": cfg.get("min_count", 20),
        }]

    specs: List[Dict[str, Any]] = []
    for t in traits_cfg:
        if not isinstance(t, dict):
            raise ValueError("traits must be a list of objects")
        if "name" not in t or "npz" not in t:
            raise ValueError("each trait must define 'name' and 'npz'")

        paths = dict(cfg["paths"])
        paths["npz"] = t["npz"]
        specs.append({
            "name": str(t["name"]),
            "paths": paths,
            "target_column": t.get("target_column", cfg.get("target_column", "y_adjusted")),
            "eval_target_column": t.get("eval_target_column", cfg.get("eval_target_column", "y_mean")),
            "standardize_features": t.get("standardize_features", cfg.get("standardize_features", True)),
            "min_count": int(t.get("min_count", cfg.get("min_count", 20))),
        })
    return specs


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def _assign_jobs_weighted(jobs: List[Dict[str, Any]], num_shards: int) -> List[List[Dict[str, Any]]]:
    if num_shards <= 1:
        return [jobs]

    bins: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
    loads = np.zeros(num_shards, dtype=np.float64)
    sorted_jobs = sorted(jobs, key=lambda j: float(j.get("weight", 1.0)), reverse=True)

    for job in sorted_jobs:
        tgt = int(np.argmin(loads))
        bins[tgt].append(job)
        loads[tgt] += float(job.get("weight", 1.0))

    return bins


def _make_repeat_seed(global_seed: int, target_code: int, repeat_idx: int) -> int:
    token = f"bpcrr_inla|{int(global_seed)}|{int(target_code)}|{int(repeat_idx)}"
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % 2_147_483_647


def _parse_n_components_values(exp_cfg: Dict[str, Any], key: str, default_value: int) -> List[int]:
    raw = exp_cfg.get(key, default_value)
    if raw is None:
        raw = default_value

    if isinstance(raw, (list, tuple)):
        values = [int(v) for v in raw]
    else:
        values = [int(raw)]

    values = sorted(set(v for v in values if v > 0))
    if not values:
        raise ValueError(f"bpcrr_inla_experiment.{key} must contain at least one positive integer")
    return values


def _parse_selection_methods(exp_cfg: Dict[str, Any]) -> List[str]:
    raw = exp_cfg.get("selection_methods", ["avggrm", "pc_distance"])
    if isinstance(raw, str):
        methods = [raw.lower()]
    else:
        methods = [str(x).lower() for x in raw]

    valid = {"avggrm", "pc_distance"}
    bad = [m for m in methods if m not in valid]
    if bad:
        raise ValueError(f"Unsupported selection methods: {bad}. Allowed: {sorted(valid)}")
    if not methods:
        raise ValueError("At least one selection method is required")
    return sorted(set(methods))


def _inla_bpcrr_predict(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
) -> np.ndarray:
    """Fit BPCRR in R-INLA and return test-set posterior mean predictions."""
    _configure_rpy2_startup()

    # Extra guard: if embedded R is first initialized while cwd contains .RData,
    # some Windows/rpy2 setups may still restore state. Initialize from a clean dir.
    try:
        from rpy2.rinterface_lib import embedded as _embedded_state
        need_clean_boot = not _embedded_state.isinitialized()
    except Exception:
        need_clean_boot = False

    old_cwd = os.getcwd()
    if need_clean_boot:
        os.chdir(tempfile.gettempdir())

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
    except Exception as exc:
        if need_clean_boot:
            os.chdir(old_cwd)
        raise ImportError(
            "rpy2 is required for BPCRR-INLA. Install it and ensure R + INLA are available."
        ) from exc

    if need_clean_boot:
        os.chdir(old_cwd)

    if "bpcrr_predict_inla" not in ro.globalenv:
        ro.r(
            """
                        configure_inla_call <- function() {
                            # On HPC systems, INLA's default wrapper may prioritize bundled
                            # runtime libs that are older/newer than the active conda toolchain.
                            # Build a small launcher that puts conda libs first.
                            use_compat <- Sys.getenv("INLA_USE_CONDA_COMPAT", unset = "1")
                            if (!(use_compat %in% c("1", "true", "TRUE", "yes", "YES"))) {
                                return(invisible(NULL))
                            }

                            conda_prefix <- Sys.getenv("CONDA_PREFIX", unset = "")
                            if (!nzchar(conda_prefix)) {
                                return(invisible(NULL))
                            }

                            inla_bin <- file.path(conda_prefix, "lib", "R", "library", "INLA", "bin", "linux", "64bit")
                            inla_exec <- file.path(inla_bin, "inla.mkl")
                            if (!file.exists(inla_exec)) {
                                return(invisible(NULL))
                            }

                            wrapper <- file.path(tempdir(), "inla_conda_compat.sh")
                            script_lines <- c(
                                "#!/bin/bash",
                                "set -e",
                                sprintf("INLA_DIR=\\\"%s\\\"", inla_bin),
                                sprintf(
                                    "export LD_LIBRARY_PATH=\\\"%s/lib:%s/lib/R/lib:$INLA_DIR:$INLA_DIR/first:${LD_LIBRARY_PATH:-}\\\"",
                                    conda_prefix,
                                    conda_prefix
                                ),
                                "exec \\\"$INLA_DIR/inla.mkl\\\" \\\"$@\\\""
                            )
                            writeLines(script_lines, con = wrapper)
                            Sys.chmod(wrapper, mode = "0755")
                            INLA::inla.setOption(inla.call = wrapper)
                        }

            bpcrr_predict_inla <- function(Z_train, y_train, Z_test) {
              if (!requireNamespace("INLA", quietly = TRUE)) {
                stop("R package 'INLA' is not installed. Install via install.packages('INLA', repos='https://inla.r-inla-download.org/R/stable').")
              }

              configure_inla_call()

              n_train <- nrow(Z_train)
              n_test <- nrow(Z_test)

              if (n_train < 2) {
                stop("Need at least 2 training samples for BPCRR-INLA.")
              }

              Z_all <- rbind(Z_train, Z_test)
              y_all <- c(as.numeric(y_train), rep(NA_real_, n_test))
              idx <- seq_len(nrow(Z_all))

              data_df <- data.frame(y = y_all, idx = idx)

                            thread_spec <- Sys.getenv("INLA_NUM_THREADS", unset = "")
                            if (!nzchar(thread_spec)) {
                                omp_threads <- Sys.getenv("OMP_NUM_THREADS", unset = "")
                                if (nzchar(omp_threads)) {
                                    thread_spec <- paste0(omp_threads, ":1")
                                }
                            }

                            if (nzchar(thread_spec)) {
                                fit <- INLA::inla(
                                    y ~ 1 + f(idx, model = "z", Z = Z_all),
                                    family = "gaussian",
                                    data = data_df,
                                    num.threads = thread_spec,
                                    control.predictor = list(compute = TRUE),
                                    control.compute = list(config = FALSE),
                                    verbose = FALSE
                                )
                            } else {
                                fit <- INLA::inla(
                                    y ~ 1 + f(idx, model = "z", Z = Z_all),
                                    family = "gaussian",
                                    data = data_df,
                                    control.predictor = list(compute = TRUE),
                                    control.compute = list(config = FALSE),
                                    verbose = FALSE
                                )
                            }

              pred_mean <- fit$summary.fitted.values$mean
              test_idx <- (n_train + 1):(n_train + n_test)

              list(
                test_pred = as.numeric(pred_mean[test_idx])
              )
            }
            """
        )

    fn = ro.globalenv["bpcrr_predict_inla"]
    Z_train = np.asarray(Z_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    Z_test = np.asarray(Z_test, dtype=np.float64)

    with localconverter(ro.default_converter + numpy2ri.converter):
        r_Z_train = ro.conversion.py2rpy(Z_train)
        r_y_train = ro.conversion.py2rpy(y_train)
        r_Z_test = ro.conversion.py2rpy(Z_test)

    res = fn(r_Z_train, r_y_train, r_Z_test)
    test_pred = np.asarray(res.rx2("test_pred"), dtype=np.float64)

    return test_pred


def _resolve_existing_path(config_path: Path, value: Optional[str], *, required: bool) -> Optional[Path]:
    if value is None:
        if required:
            raise FileNotFoundError("Required path is missing in config")
        return None

    p = Path(value)
    candidates = [
        p,
        config_path.parent / p,
        config_path.parent.parent / p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if required:
        tried = ", ".join(str(x) for x in candidates)
        raise FileNotFoundError(f"Path from config not found: {value}. Tried: {tried}")
    return None


def run_preflight(config_path: Path, smoke_test: bool = True) -> None:
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    exp_cfg = cfg.get("bpcrr_inla_experiment", {})
    selection_methods = _parse_selection_methods(exp_cfg)
    output_dir = Path(exp_cfg.get("output_dir", cfg["paths"].get("output_dir", "outputs/bpcrr_inla")))
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = _resolve_existing_path(config_path, cfg.get("paths", {}).get("npz"), required=True)
    need_grm = "avggrm" in selection_methods
    grm_path = _resolve_existing_path(config_path, cfg.get("paths", {}).get("grm_rds"), required=need_grm)

    logger.info("=== BPCRR-INLA preflight ===")
    logger.info("Config: %s", config_path)
    logger.info("NPZ: %s", npz_path)
    logger.info("GRM: %s", grm_path if grm_path is not None else "<not set>")
    logger.info("Output dir: %s", output_dir)
    logger.info("Python executable: %s", sys.executable)
    logger.info("R_HOME (env): %s", os.environ.get("R_HOME", "<unset>"))
    logger.info("R_LIBS_USER (env): %s", os.environ.get("R_LIBS_USER", "<unset>"))

    _configure_rpy2_startup()

    try:
        import rpy2.robjects as ro
    except Exception as exc:
        raise RuntimeError(
            "Failed to import rpy2.robjects in preflight. "
            "Ensure your Python environment has rpy2 and can find a working R installation."
        ) from exc

    def _r_scalar_str(expr: str) -> str:
        out = ro.r(expr)
        if out is None or len(out) == 0:
            raise RuntimeError(f"R expression returned empty result in preflight: {expr}")
        return str(out[0])

    def _r_scalar_bool(expr: str) -> bool:
        out = ro.r(expr)
        if out is None or len(out) == 0:
            return False
        return bool(out[0])

    r_home = _r_scalar_str("R.home()")
    r_version = _r_scalar_str("R.version.string")
    logger.info("Embedded R home: %s", r_home)
    logger.info("Embedded R version: %s", r_version)

    inla_ok = _r_scalar_bool('isTRUE(requireNamespace("INLA", quietly=TRUE))')
    if not inla_ok:
        raise RuntimeError(
            "R package 'INLA' is not available in the embedded R library path. "
            "Install INLA in the same R environment used by rpy2, e.g.:\n"
            "R -q -e 'install.packages(\"INLA\", repos=\"https://inla.r-inla-download.org/R/stable\")'"
        )

    inla_version = _r_scalar_str('as.character(packageVersion("INLA"))')
    logger.info("INLA version: %s", inla_version)

    if need_grm:
        try:
            import pyreadr  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Selection method 'avggrm' requires pyreadr and a readable GRM .rds file. "
                "Install pyreadr in the same Python env used by this job."
            ) from exc
        logger.info("pyreadr import check passed (avggrm enabled)")

    if smoke_test:
        rng = np.random.default_rng(1234)
        Z_train = rng.normal(size=(24, 6))
        beta = np.array([0.8, -0.4, 0.3, 0.1, -0.2, 0.5], dtype=np.float64)
        y_train = Z_train @ beta + rng.normal(scale=0.3, size=24)
        Z_test = rng.normal(size=(5, 6))
        pred = _inla_bpcrr_predict(Z_train=Z_train, y_train=y_train, Z_test=Z_test)

        if pred.shape[0] != Z_test.shape[0]:
            raise RuntimeError(
                f"INLA smoke test returned wrong prediction length: {pred.shape[0]} != {Z_test.shape[0]}"
            )
        if not np.all(np.isfinite(pred)):
            raise RuntimeError("INLA smoke test produced non-finite predictions")

        logger.info("INLA smoke test passed (n_test=%d)", pred.shape[0])

    logger.info("Preflight complete. Environment looks ready for cluster execution.")


def _evaluate_bpcrr_subset(
    train_idx: np.ndarray,
    Z_source: np.ndarray,
    y_source: np.ndarray,
    Z_target: np.ndarray,
    y_target: np.ndarray,
    y_eval_target: np.ndarray,
) -> Dict[str, float]:
    if len(train_idx) < 2:
        return {"corr_eval": 0.0, "mse_adj": float("inf")}

    pred = _inla_bpcrr_predict(
        Z_train=Z_source[train_idx],
        y_train=y_source[train_idx],
        Z_test=Z_target,
    )
    corr_eval = float(_pearson_corr(pred, y_eval_target))
    if not np.isfinite(corr_eval):
        corr_eval = 0.0
    mse_adj = float(np.mean((pred - y_target) ** 2))
    return {"corr_eval": corr_eval, "mse_adj": mse_adj}


def run_merge(config_path: Path) -> None:
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    exp_cfg = cfg.get("bpcrr_inla_experiment", {})
    output_dir = Path(exp_cfg.get("output_dir", cfg["paths"].get("output_dir", "outputs/bpcrr_inla")))
    trait_specs = _build_trait_specs(cfg)
    shards_root = output_dir / "shards"

    if not shards_root.exists():
        raise FileNotFoundError(f"No shard directory found: {shards_root}")

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        trait_output = output_dir / trait_name if len(trait_specs) > 1 else output_dir
        trait_output.mkdir(parents=True, exist_ok=True)

        merge_specs = [
            ("bpcrr_inla_rank_select_results.csv", True),
            ("bpcrr_inla_ranked_selected_individuals.csv", False),
        ]

        merged_results: Optional[pd.DataFrame] = None
        for filename, is_results in merge_specs:
            parts: List[pd.DataFrame] = []
            for csv_path in sorted(shards_root.glob(f"shard_*/{trait_name}/{filename}")):
                if csv_path.exists():
                    parts.append(pd.read_csv(csv_path))
            if len(parts) == 0:
                logger.warning("No shard files found for trait '%s': %s", trait_name, filename)
                continue

            merged = pd.concat(parts, ignore_index=True)
            out_path = trait_output / filename
            merged.to_csv(out_path, index=False)
            logger.info("Merged %d shards into %s (%d rows)", len(parts), out_path, len(merged))

            if is_results:
                merged_results = merged

        if merged_results is not None and len(merged_results) > 0:
            summary = (
                merged_results.groupby(
                    [
                        "trait",
                        "target_island",
                        "target_island_name",
                        "analysis",
                        "method",
                        "selection_method",
                        "n_components",
                        "selection_n_components",
                        "n_individuals",
                    ],
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    corr_mean=("corr_eval", "mean"),
                    corr_std=("corr_eval", "std"),
                    mse_mean=("mse_adj", "mean"),
                    n_rows=("corr_eval", "size"),
                )
            )
            summary_csv = trait_output / "bpcrr_inla_rank_select_summary.csv"
            summary.to_csv(summary_csv, index=False)
            logger.info("Wrote summary for trait '%s'", trait_name)

    logger.info("Merge complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="BPCRR-INLA ranked training-set analysis")
    parser.add_argument("--mode", choices=["worker", "merge", "preflight"], default="worker")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--target_islands", nargs="+", default=None)
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--skip_smoke_test", action="store_true", help="Skip tiny INLA fit during preflight")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    import json

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.mode == "merge":
        run_merge(config_path)
        return

    if args.mode == "preflight":
        run_preflight(config_path=config_path, smoke_test=not args.skip_smoke_test)
        return

    exp_cfg = cfg.get("bpcrr_inla_experiment", {})
    output_dir = Path(exp_cfg.get("output_dir", cfg["paths"].get("output_dir", "outputs/bpcrr_inla")))
    output_dir.mkdir(parents=True, exist_ok=True)

    global_seed = int(cfg.get("seed", 42))
    _set_seed(global_seed)

    n_repeats = int(exp_cfg.get("n_repeats", 1))
    if n_repeats < 1:
        raise ValueError("bpcrr_inla_experiment.n_repeats must be >= 1")

    n_random_reps = int(exp_cfg.get("n_random_reps", cfg.get("baselines", {}).get("n_random_orders", 5)))
    if n_random_reps < 0:
        raise ValueError("bpcrr_inla_experiment.n_random_reps must be >= 0")

    n_train_sizes_raw = exp_cfg.get("n_train_sizes", None)
    training_islands_raw = exp_cfg.get("training_islands", None)
    selection_methods = _parse_selection_methods(exp_cfg)
    bpcrr_n_components_values = _parse_n_components_values(
        exp_cfg,
        key="bpcrr_n_components",
        default_value=int(exp_cfg.get("n_components", 20) if exp_cfg.get("n_components", 20) is not None else 20),
    )
    pc_distance_n_components_values = _parse_n_components_values(
        exp_cfg,
        key="pc_distance_n_components",
        default_value=int(exp_cfg.get("n_components", 20) if exp_cfg.get("n_components", 20) is not None else 20),
    )

    trait_specs = _build_trait_specs(cfg)

    shard_index = args.shard_index
    if shard_index is None:
        shard_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    num_shards = args.num_shards
    if num_shards is None:
        num_shards = int(os.environ.get("SWEEP_NUM_SHARDS", "1"))

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}]")

    for trait_spec in trait_specs:
        trait_name = str(trait_spec["name"])
        if num_shards > 1:
            trait_output = output_dir / "shards" / f"shard_{shard_index:03d}" / trait_name
        else:
            trait_output = output_dir / trait_name if len(trait_specs) > 1 else output_dir
        trait_output.mkdir(parents=True, exist_ok=True)

        logger.info("Loading data for trait '%s'", trait_name)
        trait_paths = dict(trait_spec["paths"])
        trait_paths["npz"] = str(_resolve_existing_path(config_path, trait_paths.get("npz"), required=True))
        if "avggrm" in selection_methods:
            trait_paths["grm_rds"] = str(
                _resolve_existing_path(config_path, trait_paths.get("grm_rds"), required=True)
            )
        else:
            # Avoid unnecessary pyreadr dependency when avggrm is not selected.
            trait_paths.pop("grm_rds", None)

        X, y, ids, GRM_df, locality, code_to_label, y_eval = load_data(
            paths=trait_paths,
            target_column=trait_spec["target_column"],
            standardize_features=trait_spec["standardize_features"],
            return_locality=True,
            min_count=trait_spec["min_count"],
            return_eval=True,
            eval_target_column=trait_spec["eval_target_column"],
        )

        if "avggrm" in selection_methods and GRM_df is None:
            raise ValueError(
                "GRM is required for avggrm selection. Provide paths.grm_rds in config or remove 'avggrm' from selection_methods."
            )

        present_codes = set(int(c) for c in np.unique(locality))
        included_raw = cfg.get("included_islands", None)
        if included_raw is not None:
            included_island_codes = [resolve_island_code(v, code_to_label, present_codes) for v in included_raw]
        else:
            included_island_codes = sorted(present_codes)

        raw_targets = args.target_islands if args.target_islands else cfg.get("target_islands", [0])
        target_codes = [resolve_island_code(t, code_to_label, present_codes) for t in raw_targets]

        results_path = trait_output / "bpcrr_inla_rank_select_results.csv"
        selected_path = trait_output / "bpcrr_inla_ranked_selected_individuals.csv"
        for p in [results_path, selected_path]:
            if p.exists():
                p.unlink()

        jobs: List[Dict[str, Any]] = []
        steps_by_target: Dict[int, np.ndarray] = {}

        for target_code in target_codes:
            source_codes = [c for c in included_island_codes if c != target_code]
            if training_islands_raw is not None:
                training_codes = _resolve_training_islands(
                    training_islands_raw, code_to_label, present_codes, target_code
                )
                if training_codes is not None:
                    source_codes = [c for c in source_codes if c in training_codes]

            if len(source_codes) == 0:
                continue

            target_mask = locality == target_code
            source_mask = (~target_mask) & np.isin(locality, source_codes)
            n_source = int(np.sum(source_mask))
            if n_source < 2:
                continue

            if n_train_sizes_raw is not None:
                step_counts = np.array(sorted(int(x) for x in n_train_sizes_raw), dtype=np.int64)
            else:
                locality_source = locality[source_mask]
                n_per_island = np.array([(locality_source == c).sum() for c in source_codes], dtype=np.int64)
                step_counts = np.cumsum(np.sort(n_per_island)[::-1])

            step_counts = np.unique(np.clip(step_counts, 2, n_source))
            step_counts = step_counts[step_counts < n_source]
            steps_by_target[int(target_code)] = step_counts

            for repeat_idx in range(n_repeats):
                jobs.append({
                    "target_code": int(target_code),
                    "repeat_idx": int(repeat_idx),
                    "weight": float(
                        max(1, n_source)
                        * (
                            len(bpcrr_n_components_values)
                            * (1 + len(step_counts) * (len(selection_methods) + n_random_reps))
                        )
                    ),
                })

        shard_bins = _assign_jobs_weighted(jobs, num_shards)
        shard_jobs = shard_bins[shard_index] if num_shards > 1 else jobs
        assigned = {(int(j["target_code"]), int(j["repeat_idx"])) for j in shard_jobs}

        logger.info(
            "Shard %d/%d assigned %d/%d jobs",
            shard_index,
            num_shards,
            len(shard_jobs),
            len(jobs),
        )

        done = 0
        total = len(shard_jobs)

        for target_code in target_codes:
            target_name = island_label(target_code, code_to_label)
            source_codes = [c for c in included_island_codes if c != target_code]
            if training_islands_raw is not None:
                training_codes = _resolve_training_islands(
                    training_islands_raw, code_to_label, present_codes, target_code
                )
                if training_codes is not None:
                    source_codes = [c for c in source_codes if c in training_codes]

            if len(source_codes) == 0:
                continue

            target_mask = locality == target_code
            source_mask = (~target_mask) & np.isin(locality, source_codes)

            X_source = X[source_mask]
            y_source = y[source_mask]
            ids_source = ids[source_mask]
            locality_source = locality[source_mask]

            X_target = X[target_mask]
            y_target = y[target_mask]
            y_eval_target = y_eval[target_mask]
            ids_target = ids[target_mask]

            n_source = len(X_source)
            if n_source < 2 or len(X_target) == 0:
                continue

            # Fit PCA once on full source candidate set and reuse this basis for all subsets.
            max_bpcrr_requested = int(max(bpcrr_n_components_values))
            max_bpcrr_feasible = int(min(max_bpcrr_requested, X_source.shape[0], X_source.shape[1]))
            if max_bpcrr_feasible < 1:
                logger.warning("Skipping target %s due to infeasible PCA dimensionality", target_name)
                continue

            pca_bpcrr = PCA(n_components=max_bpcrr_feasible)
            Z_source_bpcrr_full = pca_bpcrr.fit_transform(X_source)
            Z_target_bpcrr_full = pca_bpcrr.transform(X_target)

            pc_distance_cache: Dict[int, np.ndarray] = {}
            if "pc_distance" in selection_methods:
                max_sel_requested = int(max(pc_distance_n_components_values))
                max_sel_feasible = int(min(max_sel_requested, X_source.shape[0], X_source.shape[1]))
                if max_sel_feasible < 1:
                    logger.warning("Skipping target %s due to infeasible PCA dimensionality for pc_distance", target_name)
                    continue
                pca_sel = PCA(n_components=max_sel_feasible)
                Z_source_sel_full = pca_sel.fit_transform(X_source)
                Z_target_sel_full = pca_sel.transform(X_target)
                for sel_n_comp_req in pc_distance_n_components_values:
                    sel_n_comp = int(min(int(sel_n_comp_req), Z_source_sel_full.shape[1]))
                    if sel_n_comp < 1:
                        continue
                    z_src = Z_source_sel_full[:, :sel_n_comp]
                    z_tgt = Z_target_sel_full[:, :sel_n_comp]
                    centroid = z_tgt.mean(axis=0)
                    pc_distance_cache[int(sel_n_comp)] = np.linalg.norm(z_src - centroid[None, :], axis=1)

            avg_grm = None
            if "avggrm" in selection_methods:
                grm_block = GRM_df.loc[ids_source, ids_target].to_numpy(dtype=float)
                avg_grm = np.asarray(grm_block.mean(axis=1), dtype=float)

            step_counts = steps_by_target.get(int(target_code), np.array([], dtype=np.int64)).copy()
            if len(step_counts) == 0:
                continue

            for repeat_idx in range(n_repeats):
                if (int(target_code), int(repeat_idx)) not in assigned:
                    continue

                done += 1
                repeat_seed = _make_repeat_seed(global_seed, int(target_code), int(repeat_idx))
                logger.info(
                    "Job %d/%d | trait=%s target=%s repeat=%d/%d",
                    done,
                    total,
                    trait_name,
                    target_code,
                    repeat_idx + 1,
                    n_repeats,
                )

                for n_comp_req in bpcrr_n_components_values:
                    n_comp = int(min(int(n_comp_req), Z_source_bpcrr_full.shape[1]))
                    if n_comp < 1:
                        continue

                    Z_source = Z_source_bpcrr_full[:, :n_comp]
                    Z_target = Z_target_bpcrr_full[:, :n_comp]

                    n_ranked_fit_evals = int(len(step_counts)) * (
                        int(n_random_reps)
                        + (1 if "avggrm" in selection_methods else 0)
                        + (len(pc_distance_cache) if "pc_distance" in selection_methods else 0)
                    )
                    n_fit_evals_total = 1 + n_ranked_fit_evals
                    fit_eval_done = 0
                    fit_eval_started_at = time.perf_counter()

                    def _log_fit_start(stage: str, n_train: int) -> float:
                        fit_idx = fit_eval_done + 1
                        logger.info(
                            "Fit %d/%d started | trait=%s target=%s repeat=%d n_comp=%d stage=%s n_train=%d",
                            fit_idx,
                            n_fit_evals_total,
                            trait_name,
                            target_code,
                            repeat_idx,
                            n_comp,
                            stage,
                            int(n_train),
                        )
                        return time.perf_counter()

                    def _log_fit_done(stage: str, start_ts: float) -> None:
                        nonlocal fit_eval_done
                        fit_eval_done += 1
                        elapsed = time.perf_counter() - start_ts
                        total_elapsed = time.perf_counter() - fit_eval_started_at
                        avg_time = total_elapsed / fit_eval_done
                        remaining = max(n_fit_evals_total - fit_eval_done, 0)
                        eta_sec = avg_time * remaining
                        logger.info(
                            "Fit %d/%d done | stage=%s elapsed=%.1fs total=%.1fs eta=%.1fs",
                            fit_eval_done,
                            n_fit_evals_total,
                            stage,
                            elapsed,
                            total_elapsed,
                            eta_sec,
                        )

                    full_idx = np.arange(n_source, dtype=np.int64)
                    fit_started = _log_fit_start("full_baseline", len(full_idx))
                    full_eval = _evaluate_bpcrr_subset(
                        train_idx=full_idx,
                        Z_source=Z_source,
                        y_source=y_source,
                        Z_target=Z_target,
                        y_target=y_target,
                        y_eval_target=y_eval_target,
                    )
                    _log_fit_done("full_baseline", fit_started)
                    full_row = {
                        "analysis": "full_baseline",
                        "method": "full_source_unweighted",
                        "selection_method": "none",
                        "order_seed": -2,
                        "weighted_fit_used": False,
                        "n_individuals": int(n_source),
                        "corr_eval": float(full_eval["corr_eval"]),
                        "mse_adj": float(full_eval["mse_adj"]),
                        "target_island": int(target_code),
                        "target_island_name": str(target_name),
                        "repeat": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "trait": trait_name,
                        "n_components": int(n_comp),
                        "selection_n_components": np.nan,
                        "avg_grm_obj": float(np.mean(avg_grm)) if avg_grm is not None else float("nan"),
                        "pca_dist_obj": float("nan"),
                    }
                    _append_csv(pd.DataFrame([full_row]), results_path)

                    for order_seed in range(n_random_reps):
                        rng = np.random.default_rng(repeat_seed + 500_000 + order_seed + n_comp)
                        shuffled = rng.permutation(n_source)
                        for k in step_counts:
                            n_train = int(min(int(k), n_source))
                            chosen = shuffled[:n_train]

                            stage = f"random_seed{int(order_seed)}_k{int(n_train)}"
                            fit_started = _log_fit_start(stage, n_train)
                            eval_result = _evaluate_bpcrr_subset(
                                train_idx=chosen,
                                Z_source=Z_source,
                                y_source=y_source,
                                Z_target=Z_target,
                                y_target=y_target,
                                y_eval_target=y_eval_target,
                            )
                            _log_fit_done(stage, fit_started)
                            rand_row = {
                                "analysis": "ranked_subset",
                                "method": "random_individual",
                                "selection_method": "random",
                                "order_seed": int(order_seed),
                                "weighted_fit_used": False,
                                "n_individuals": int(n_train),
                                "corr_eval": float(eval_result["corr_eval"]),
                                "mse_adj": float(eval_result["mse_adj"]),
                                "target_island": int(target_code),
                                "target_island_name": str(target_name),
                                "repeat": int(repeat_idx),
                                "repeat_seed": int(repeat_seed),
                                "trait": trait_name,
                                "n_components": int(n_comp),
                                "selection_n_components": np.nan,
                                "avg_grm_obj": float("nan"),
                                "pca_dist_obj": float("nan"),
                            }
                            _append_csv(pd.DataFrame([rand_row]), results_path)

                    for selection_method in selection_methods:
                        if selection_method == "avggrm":
                            scores = np.asarray(avg_grm, dtype=float)
                            order = np.argsort(-scores, kind="mergesort")
                            pca_distances = np.full(n_source, np.nan, dtype=float)
                            sel_n_comp_values = [np.nan]
                        else:
                            sel_n_comp_values = sorted(pc_distance_cache.keys())

                        for sel_n_comp in sel_n_comp_values:
                            if selection_method == "pc_distance":
                                pca_distances = pc_distance_cache[int(sel_n_comp)]
                                scores = -pca_distances
                                order = np.argsort(pca_distances, kind="mergesort")

                            ranks = np.empty_like(order)
                            ranks[order] = np.arange(1, len(order) + 1)

                            for k in step_counts:
                                n_train = int(min(int(k), n_source))
                                chosen = order[:n_train]

                                if selection_method == "pc_distance":
                                    stage = f"pc_distance_pc{int(sel_n_comp)}_k{int(n_train)}"
                                else:
                                    stage = f"avggrm_k{int(n_train)}"
                                fit_started = _log_fit_start(stage, n_train)
                                eval_result = _evaluate_bpcrr_subset(
                                    train_idx=chosen,
                                    Z_source=Z_source,
                                    y_source=y_source,
                                    Z_target=Z_target,
                                    y_target=y_target,
                                    y_eval_target=y_eval_target,
                                )
                                _log_fit_done(stage, fit_started)

                                row = {
                                    "analysis": "ranked_subset",
                                    "method": f"bpcrr_topk_{selection_method}",
                                    "selection_method": selection_method,
                                    "order_seed": -1,
                                    "weighted_fit_used": False,
                                    "n_individuals": int(n_train),
                                    "corr_eval": float(eval_result["corr_eval"]),
                                    "mse_adj": float(eval_result["mse_adj"]),
                                    "target_island": int(target_code),
                                    "target_island_name": str(target_name),
                                    "repeat": int(repeat_idx),
                                    "repeat_seed": int(repeat_seed),
                                    "trait": trait_name,
                                    "n_components": int(n_comp),
                                    "selection_n_components": float(sel_n_comp) if selection_method == "pc_distance" else np.nan,
                                    "avg_grm_obj": float(np.mean(scores[chosen])) if selection_method == "avggrm" else float("nan"),
                                    "pca_dist_obj": float(np.mean(pca_distances[chosen])) if selection_method == "pc_distance" else float("nan"),
                                }
                                _append_csv(pd.DataFrame([row]), results_path)

                                selected_df = pd.DataFrame({
                                    "trait": trait_name,
                                    "target_island": int(target_code),
                                    "target_island_name": str(target_name),
                                    "repeat": int(repeat_idx),
                                    "repeat_seed": int(repeat_seed),
                                    "n_train_size": int(n_train),
                                    "method": f"bpcrr_topk_{selection_method}",
                                    "selection_method": selection_method,
                                    "n_components": int(n_comp),
                                    "selection_n_components": float(sel_n_comp) if selection_method == "pc_distance" else np.nan,
                                    "ringnr": ids_source[chosen],
                                    "ringnumber": ids_source[chosen],
                                    "source_island": locality_source[chosen].astype(int),
                                    "source_island_name": [island_label(int(c), code_to_label) for c in locality_source[chosen]],
                                    "rank": ranks[chosen].astype(int),
                                    "avg_grm": np.asarray(avg_grm[chosen], dtype=float) if avg_grm is not None else np.full(n_train, np.nan),
                                    "pca_dist": pca_distances[chosen].astype(float),
                                })
                                _append_csv(selected_df, selected_path)

        if results_path.exists():
            all_results = pd.read_csv(results_path)
            summary = (
                all_results.groupby(
                    [
                        "trait",
                        "target_island",
                        "target_island_name",
                        "analysis",
                        "method",
                        "selection_method",
                        "n_components",
                        "selection_n_components",
                        "n_individuals",
                    ],
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    corr_mean=("corr_eval", "mean"),
                    corr_std=("corr_eval", "std"),
                    mse_mean=("mse_adj", "mean"),
                    n_rows=("corr_eval", "size"),
                )
            )
            summary_csv = trait_output / "bpcrr_inla_rank_select_summary.csv"
            summary.to_csv(summary_csv, index=False)

        logger.info("Trait '%s' complete. Output: %s", trait_name, trait_output)


if __name__ == "__main__":
    main()
