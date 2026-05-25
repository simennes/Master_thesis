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
    - bpcrr_pev_ga: GA subset search minimizing a BPCRR-space PEV surrogate

The BPCRR model can use either INLA's default latent-effect prior (default)
or a paper-style fixed precision prior derived from an a priori genetic
variance value (config: bpcrr_prior_mode, bpcrr_va_apriori).

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
import copy
import hashlib
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.cv_utils import ISLAND_ID_TO_NAME, island_label
from src.data import load_data
from src.training_set_optimization.ga_subset import GAConfig, run_ga
from src.training_set_optimization.pevmean import build_kernel, pev_mean

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
        if isinstance(t.get("paths", None), dict):
            paths.update(t["paths"])
        paths["npz"] = t["npz"]
        for path_key in ("phenotype_csv", "fhat_csv", "grm_rds"):
            if path_key in t:
                paths[path_key] = t[path_key]
        specs.append({
            "name": str(t["name"]),
            "paths": paths,
            "target_column": t.get("target_column", cfg.get("target_column", "y_adjusted")),
            "eval_target_column": t.get("eval_target_column", cfg.get("eval_target_column", "y_mean")),
            "standardize_features": t.get("standardize_features", cfg.get("standardize_features", True)),
            "min_count": int(t.get("min_count", cfg.get("min_count", 20))),
            "one_step": t.get("one_step", None),
            "bpcrr_prior_mode": t.get("bpcrr_prior_mode", None),
            "bpcrr_va_apriori": t.get("bpcrr_va_apriori", None),
            "bpcrr_inla_experiment": t.get("bpcrr_inla_experiment", None),
        })
    return specs


def _exp_cfg_for_trait(
    exp_cfg: Dict[str, Any],
    trait_spec: Dict[str, Any],
) -> Dict[str, Any]:
    exp_cfg_for_trait = copy.deepcopy(exp_cfg)
    trait_exp_cfg = trait_spec.get("bpcrr_inla_experiment", None)
    if isinstance(trait_exp_cfg, dict):
        exp_cfg_for_trait.update(trait_exp_cfg)

    for key in ("bpcrr_prior_mode", "bpcrr_va_apriori"):
        value = trait_spec.get(key, None)
        if value is not None:
            exp_cfg_for_trait[key] = value

    trait_one_step = trait_spec.get("one_step", None)
    if trait_one_step is not None:
        base_one_step = exp_cfg_for_trait.get("one_step", {})
        if isinstance(trait_one_step, bool):
            exp_cfg_for_trait["one_step"] = bool(trait_one_step)
        else:
            if isinstance(base_one_step, bool):
                base_one_step = {"enabled": bool(base_one_step)}
            merged_one_step = dict(base_one_step)
            merged_one_step.update(trait_one_step)
            exp_cfg_for_trait["one_step"] = merged_one_step

    return exp_cfg_for_trait


def _config_for_trait_one_step(
    cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
    trait_spec: Dict[str, Any],
    trait_paths: Dict[str, Any],
) -> Dict[str, Any]:
    cfg_for_one_step = copy.deepcopy(cfg)
    cfg_for_one_step["paths"] = dict(trait_paths)
    cfg_for_one_step["bpcrr_inla_experiment"] = _exp_cfg_for_trait(exp_cfg, trait_spec)
    return cfg_for_one_step


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        df.to_csv(f, header=write_header, index=False)
        f.flush()
        os.fsync(f.fileno())


_RESULT_KEY_COLUMNS = [
    "trait",
    "target_island",
    "repeat",
    "analysis",
    "method",
    "selection_method",
    "order_seed",
    "n_components",
    "selection_n_components",
    "n_individuals",
]

_SUMMARY_GROUP_COLUMNS = [
    "trait",
    "target_island",
    "target_island_name",
    "analysis",
    "method",
    "selection_method",
    "n_components",
    "selection_n_components",
    "n_individuals",
]


def _normalise_result_key_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def _result_key(row: Dict[str, Any]) -> tuple[Any, ...]:
    return tuple(_normalise_result_key_value(row.get(c, np.nan)) for c in _RESULT_KEY_COLUMNS)


def _load_completed_result_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    existing = pd.read_csv(path)
    missing = [c for c in _RESULT_KEY_COLUMNS if c not in existing.columns]
    if missing:
        logger.warning("Cannot resume from %s because columns are missing: %s", path, missing)
        return set()
    return {_result_key(row) for row in existing.to_dict(orient="records")}


def _write_results_summary(results_path: Path, summary_path: Path) -> None:
    if not results_path.exists():
        return
    all_results = pd.read_csv(results_path)
    if len(all_results) == 0:
        return
    missing = [c for c in _SUMMARY_GROUP_COLUMNS if c not in all_results.columns]
    if missing:
        logger.warning("Cannot write summary from %s because columns are missing: %s", results_path, missing)
        return

    summary = (
        all_results.groupby(
            _SUMMARY_GROUP_COLUMNS,
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
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = summary_path.with_name(f"{summary_path.name}.tmp")
    summary.to_csv(tmp_path, index=False)
    os.replace(tmp_path, summary_path)
    with open(summary_path, "rb") as f:
        os.fsync(f.fileno())


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


def _coerce_month_label(v: Any) -> str:
    if pd.isna(v):
        return "unk"
    try:
        return str(int(float(v)))
    except Exception:
        s = str(v).strip()
        return s if s else "unk"


def _infer_trait_column(paths_cfg: Dict[str, Any]) -> Optional[str]:
    npz_val = paths_cfg.get("npz", paths_cfg.get("npz_path", None))
    if npz_val is None:
        return None
    name = Path(str(npz_val)).stem
    if name.startswith("snp_") and name.endswith("_ALL"):
        return name[len("snp_"): -len("_ALL")]
    return None


def _prepare_one_step_covariates(
    config_path: Path,
    cfg: Dict[str, Any],
    ids: np.ndarray,
    locality_codes: np.ndarray,
    code_to_label: Dict[int, str],
) -> Optional[Dict[str, np.ndarray]]:
    """Build long-format per-record covariates for the one-step BPCRR-INLA model.

    Returns a dict whose value arrays all have length n_records (the total
    number of phenotype records across all genotyped individuals), plus an
    `ind_idx` array mapping each record to its index in `ids`. This mirrors
    the long-format model in Aase et al. (2025) and Aspheim et al. (2024),
    where individual-level (permanent environment) and session-level random
    effects absorb within-individual variation.
    """
    exp_cfg = cfg.get("bpcrr_inla_experiment", {})
    one_step_cfg = exp_cfg.get("one_step", {})
    if isinstance(one_step_cfg, bool):
        enabled = bool(one_step_cfg)
        one_step_cfg = {}
    else:
        enabled = bool(one_step_cfg.get("enabled", False))

    if not enabled:
        return None

    paths_cfg = cfg.get("paths", {})
    phen_rel = one_step_cfg.get("phenotype_csv", paths_cfg.get("phenotype_csv", None))
    phen_path = _resolve_existing_path(config_path, phen_rel, required=True)
    phen_sep = str(one_step_cfg.get("phenotype_sep", ";"))

    trait_column = one_step_cfg.get("trait_column") or _infer_trait_column(paths_cfg)
    if not trait_column:
        raise ValueError(
            "one_step requires 'trait_column' (e.g., 'body_mass') either in the "
            "one_step config block or inferable from the npz path."
        )

    phen = pd.read_csv(phen_path, sep=phen_sep)
    if "ringnr" not in phen.columns:
        raise ValueError("one_step requires 'ringnr' in phenotype_csv")
    if trait_column not in phen.columns:
        raise ValueError(
            f"one_step trait_column '{trait_column}' not found in phenotype_csv"
        )

    phen = phen.copy()
    phen["ringnr"] = phen["ringnr"].astype(str)

    # Restrict to records belonging to genotyped individuals.
    id_str = ids.astype(str) if isinstance(ids, np.ndarray) else np.asarray(ids).astype(str)
    id_set = set(id_str.tolist())
    phen = phen[phen["ringnr"].isin(id_set)].copy()

    # Drop records with no value for the trait of interest.
    trait_vals = pd.to_numeric(phen[trait_column], errors="coerce")
    finite_trait = np.isfinite(trait_vals.to_numpy(dtype=float))
    phen = phen.loc[finite_trait].copy()
    if phen.empty:
        raise ValueError(
            f"No phenotype records remain for trait '{trait_column}' after filtering."
        )

    # Every genotyped individual must contribute at least one record.
    present = set(phen["ringnr"].astype(str).unique().tolist())
    missing = [rid for rid in id_str.tolist() if rid not in present]
    if missing:
        raise ValueError(
            "one_step phenotype_csv has no records for some loaded ids; "
            f"missing={len(missing)} (e.g., {missing[:5]})"
        )

    n_rec = len(phen)
    ringnr_to_ind = {rid: i for i, rid in enumerate(id_str.tolist())}
    ringnr_arr = phen["ringnr"].astype(str).to_numpy(dtype=object)
    ind_idx = np.array([ringnr_to_ind[r] for r in ringnr_arr], dtype=np.int64)

    sex_raw = pd.to_numeric(phen.get("adult_sex", np.nan), errors="coerce")
    sex = np.where(
        sex_raw == 1,
        "m",
        np.where(sex_raw == 2, "f", "unk"),
    ).astype(str)

    month = np.array(
        [_coerce_month_label(v) for v in phen.get("month", pd.Series([np.nan] * n_rec))],
        dtype=object,
    )

    hatch_year_num = pd.to_numeric(phen.get("hatch_year", np.nan), errors="coerce")
    year_num = pd.to_numeric(phen.get("year", np.nan), errors="coerce")
    age = (year_num - hatch_year_num).to_numpy(dtype=float)
    if np.all(~np.isfinite(age)):
        age = np.zeros_like(age, dtype=float)
    else:
        age_fill = float(np.nanmedian(age[np.isfinite(age)]))
        age = np.where(np.isfinite(age), age, age_fill)

    if "locality" in phen.columns:
        locality_lbl = (
            phen["locality"].astype(str).replace({"nan": "unk"}).to_numpy(dtype=object)
        )
    else:
        locality_lbl = np.array(
            [str(code_to_label[int(locality_codes[i])]) for i in ind_idx],
            dtype=object,
        )

    if "hatch_year" in phen.columns:
        hatch_year = (
            phen["hatch_year"].astype(str).replace({"nan": "unk"}).to_numpy(dtype=object)
        )
    else:
        hatch_year = np.full(n_rec, "unk", dtype=object)

    include_f_hat = bool(one_step_cfg.get("include_f_hat", True))
    f_hat = None
    if include_f_hat:
        if "F_hat" in phen.columns:
            fhat_per_record = pd.to_numeric(phen["F_hat"], errors="coerce").to_numpy(
                dtype=float
            )
        else:
            fhat_rel = one_step_cfg.get("fhat_csv", paths_cfg.get("fhat_csv", None))
            if fhat_rel is None:
                raise ValueError(
                    "one_step.include_f_hat=true but no F_hat column or fhat_csv provided"
                )
            fhat_path = _resolve_existing_path(config_path, fhat_rel, required=True)
            fhat_df = pd.read_csv(fhat_path)
            if "ringnr" not in fhat_df.columns or "F_hat" not in fhat_df.columns:
                raise ValueError("fhat_csv must contain 'ringnr' and 'F_hat'")
            fhat_df = fhat_df.copy()
            fhat_df["ringnr"] = fhat_df["ringnr"].astype(str)
            fhat_df = fhat_df[~fhat_df["ringnr"].duplicated(keep="last")].set_index("ringnr")
            fhat_per_ind = pd.to_numeric(
                fhat_df.reindex(id_str)["F_hat"], errors="coerce"
            ).to_numpy(dtype=float)
            fhat_per_record = fhat_per_ind[ind_idx]

        finite_mask = np.isfinite(fhat_per_record)
        if not np.any(finite_mask):
            raise ValueError("one_step.include_f_hat=true but no finite F_hat values were found")
        if not np.all(finite_mask):
            n_missing = int(np.sum(~finite_mask))
            raise ValueError(
                f"one_step.include_f_hat=true but {n_missing} records are missing F_hat."
            )
        f_hat = fhat_per_record.astype(float)

    y_record = trait_vals.loc[phen.index].to_numpy(dtype=float)

    return {
        "sex": np.asarray(sex, dtype=object),
        "month": np.asarray(month, dtype=object),
        "age": np.asarray(age, dtype=float),
        "locality": np.asarray(locality_lbl, dtype=object),
        "hatch_year": np.asarray(hatch_year, dtype=object),
        "ringnr": np.asarray(ringnr_arr, dtype=object),
        "f_hat": None if f_hat is None else np.asarray(f_hat, dtype=float),
        "y": np.asarray(y_record, dtype=float),
        "ind_idx": np.asarray(ind_idx, dtype=np.int64),
    }


def _long_format_initial_slice(
    one_step_covars: Optional[Dict[str, np.ndarray]],
    selected_inds: np.ndarray,
    n_total_inds: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Slice global long-format one_step_covars (keyed by `ind_idx`) to records
    belonging to ``selected_inds`` (positions in the original `ids` array).
    Returns a new dict with `z_row` re-mapped to row positions in the new
    sub-selected Z matrix (0..len(selected_inds)-1)."""
    if one_step_covars is None:
        return None
    ind_idx_long = np.asarray(one_step_covars["ind_idx"], dtype=np.int64)
    selected_inds = np.asarray(selected_inds, dtype=np.int64)

    record_pos = np.where(np.isin(ind_idx_long, selected_inds))[0]
    old_to_new = -np.ones(int(n_total_inds), dtype=np.int64)
    old_to_new[selected_inds] = np.arange(len(selected_inds))
    z_row_new = old_to_new[ind_idx_long[record_pos]]

    out: Dict[str, np.ndarray] = {}
    for k, v in one_step_covars.items():
        if k == "ind_idx":
            continue
        if v is None:
            out[k] = None
        else:
            out[k] = np.asarray(v)[record_pos]
    out["z_row"] = z_row_new
    return out


def _long_format_subset(
    one_step_dict: Optional[Dict[str, np.ndarray]],
    selected_z_rows: np.ndarray,
    n_z_rows: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Subset a long-format dict (keyed by `z_row`) to records whose `z_row`
    points to ``selected_z_rows`` (positions in the parent Z matrix). Returns
    a new dict with `z_row` re-mapped into the new (sub-selected) Z space."""
    if one_step_dict is None:
        return None
    z_row_old = np.asarray(one_step_dict["z_row"], dtype=np.int64)
    selected_z_rows = np.asarray(selected_z_rows, dtype=np.int64)

    record_pos = np.where(np.isin(z_row_old, selected_z_rows))[0]
    old_to_new = -np.ones(int(n_z_rows), dtype=np.int64)
    old_to_new[selected_z_rows] = np.arange(len(selected_z_rows))
    z_row_new = old_to_new[z_row_old[record_pos]]

    out: Dict[str, np.ndarray] = {}
    for k, v in one_step_dict.items():
        if v is None:
            out[k] = None
        elif k == "z_row":
            out[k] = z_row_new
        else:
            out[k] = np.asarray(v)[record_pos]
    return out


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

    valid = {"avggrm", "pc_distance", "bpcrr_pev_ga"}
    bad = [m for m in methods if m not in valid]
    if bad:
        raise ValueError(f"Unsupported selection methods: {bad}. Allowed: {sorted(valid)}")
    if not methods:
        raise ValueError("At least one selection method is required")
    return sorted(set(methods))


def _estimate_sigma_e2(y: np.ndarray) -> float:
    """Estimate a positive residual-scale proxy from phenotypes when not provided."""
    vals = np.asarray(y, dtype=np.float64)
    if vals.size < 2:
        return 1.0
    sigma_e2 = float(np.var(vals, ddof=1))
    if not np.isfinite(sigma_e2) or sigma_e2 <= 0:
        sigma_e2 = float(np.var(vals, ddof=0))
    if not np.isfinite(sigma_e2) or sigma_e2 <= 0:
        sigma_e2 = 1.0
    return sigma_e2


def _sum_pc_variances(z_matrix: np.ndarray) -> float:
    """Return sum_j Var(PC_j) for the provided PC score matrix."""
    z = np.asarray(z_matrix, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1:
        return float("nan")
    v_sum = float(np.sum(np.var(z, axis=0, ddof=1)))
    if not np.isfinite(v_sum) or v_sum <= 0:
        v_sum = float(np.sum(np.var(z, axis=0, ddof=0)))
    return v_sum


def _parse_bpcrr_pev_lambda_cfg(exp_cfg: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Parse lambda configuration for BPCRR-space PEV GA.

    Supported modes:
      - fixed (default): use scalar bpcrr_pev_lambda
      - paper: lambda = sigma_e^2 / sigma_u*^2 with
               sigma_u*^2 = va_apriori / sum_j Var(PC_j)
    """
    prior_mode_hint = str(exp_cfg.get("bpcrr_prior_mode", "default")).strip().lower()
    default_mode = "paper" if prior_mode_hint == "fixed_va" else "fixed"
    mode = str(exp_cfg.get("bpcrr_pev_lambda_mode", default_mode)).strip().lower()
    valid_modes = {"fixed", "paper"}
    if mode not in valid_modes:
        raise ValueError(f"bpcrr_inla_experiment.bpcrr_pev_lambda_mode must be one of {sorted(valid_modes)}")

    lambda_fixed = float(exp_cfg.get("bpcrr_pev_lambda", 1.0))
    if not np.isfinite(lambda_fixed) or lambda_fixed <= 0:
        raise ValueError("bpcrr_inla_experiment.bpcrr_pev_lambda must be > 0")

    sigma_e2_raw = exp_cfg.get("bpcrr_sigma_e2_apriori", None)
    sigma_e2_apriori: Optional[float] = None
    if sigma_e2_raw is not None:
        sigma_e2_apriori = float(sigma_e2_raw)
        if not np.isfinite(sigma_e2_apriori) or sigma_e2_apriori <= 0:
            raise ValueError("bpcrr_inla_experiment.bpcrr_sigma_e2_apriori must be > 0 when provided")

    return {
        "mode": mode,
        "lambda_fixed": lambda_fixed,
        "sigma_e2_apriori": sigma_e2_apriori,
    }


def _parse_bpcrr_pev_ga_cfg(exp_cfg: Dict[str, Any], global_seed: int) -> GAConfig:
    """Parse GA hyperparameters for BPCRR-space PEV subset search."""
    ga_raw = exp_cfg.get("bpcrr_pev_ga", {})
    if ga_raw is None:
        ga_raw = {}

    return GAConfig(
        pop_size=int(ga_raw.get("pop_size", 30)),
        n_generations=int(ga_raw.get("n_generations", 80)),
        n_elite=int(ga_raw.get("n_elite", 2)),
        tournament_k=int(ga_raw.get("tournament_k", 3)),
        crossover_prob=float(ga_raw.get("crossover_prob", 0.9)),
        mutation_prob=float(ga_raw.get("mutation_prob", 0.35)),
        n_swaps_per_mut=int(ga_raw.get("n_swaps_per_mut", 2)),
        seed=int(ga_raw.get("seed", global_seed)),
        verbose=bool(ga_raw.get("verbose", False)),
        stagnation_limit=int(ga_raw.get("stagnation_limit", 10)),
    )


def _parse_bpcrr_prior_cfg(exp_cfg: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    """Parse BPCRR latent-effect prior mode and optional a priori variance."""
    mode = str(exp_cfg.get("bpcrr_prior_mode", "default")).strip().lower()
    valid_modes = {"default", "fixed_va"}
    if mode not in valid_modes:
        raise ValueError(f"bpcrr_inla_experiment.bpcrr_prior_mode must be one of {sorted(valid_modes)}")

    va_apriori_raw = exp_cfg.get("bpcrr_va_apriori", None)
    va_apriori: Optional[float] = None
    if va_apriori_raw is not None:
        va_apriori = float(va_apriori_raw)
        if va_apriori <= 0:
            raise ValueError("bpcrr_inla_experiment.bpcrr_va_apriori must be > 0 when provided")

    if mode == "fixed_va" and va_apriori is None:
        raise ValueError(
            "bpcrr_inla_experiment.bpcrr_prior_mode='fixed_va' requires bpcrr_va_apriori (>0)"
        )

    return mode, va_apriori


def _inla_bpcrr_predict(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    train_weights: Optional[np.ndarray] = None,
    one_step_train: Optional[Dict[str, np.ndarray]] = None,
    one_step_test: Optional[Dict[str, np.ndarray]] = None,
    rr_prior_mode: str = "default",
    rr_va_apriori: Optional[float] = None,
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

                        bpcrr_predict_inla <- function(
                            Z_train, y_train, Z_test,
                            train_weights = NULL,
                            sex_train = NULL, sex_test = NULL,
                            month_train = NULL, month_test = NULL,
                            age_train = NULL, age_test = NULL,
                            fhat_train = NULL, fhat_test = NULL,
                            locality_train = NULL, locality_test = NULL,
                            hatch_year_train = NULL, hatch_year_test = NULL,
                            ringnr_train = NULL, ringnr_test = NULL,
                            rr_prior_mode = NULL, rr_va_apriori = NULL,
                            z_var_sum_override = NULL
                        ) {
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
                            has_train_weights <- !is.null(train_weights)
                            if (has_train_weights) {
                                train_weights <- as.numeric(train_weights)
                                if (length(train_weights) != n_train) {
                                    stop("train_weights must have length n_train")
                                }
                                if (any(!is.finite(train_weights)) || any(train_weights <= 0)) {
                                    stop("train_weights must be finite and > 0")
                                }
                                # Some INLA builds expose an explicit gate for weights; others do not.
                                # Enable it only when available to avoid hard failures on newer releases.
                                inla_opt_names <- tryCatch(
                                    names(INLA::inla.getOption()),
                                    error = function(e) character(0)
                                )
                                if ("enable.inla.argument.weights" %in% inla_opt_names) {
                                    INLA::inla.setOption(enable.inla.argument.weights = TRUE)
                                }
                                weights_all <- c(train_weights, rep(1.0, n_test))
                            } else {
                                weights_all <- NULL
                            }

              data_df <- data.frame(y = y_all, idx = idx)

                            use_one_step <- !is.null(sex_train) && !is.null(sex_test) &&
                                !is.null(month_train) && !is.null(month_test) &&
                                !is.null(age_train) && !is.null(age_test) &&
                                !is.null(locality_train) && !is.null(locality_test) &&
                                !is.null(hatch_year_train) && !is.null(hatch_year_test)

                            rr_mode <- if (is.null(rr_prior_mode)) "default" else as.character(rr_prior_mode)[1]
                            if (is.na(rr_mode) || !nzchar(rr_mode)) {
                                rr_mode <- "default"
                            }

                            formula_str <- "y ~ 1 + f(idx, model = 'z', Z = Z_all)"
                            if (identical(rr_mode, "fixed_va")) {
                                va_apriori <- as.numeric(rr_va_apriori)[1]
                                if (!is.finite(va_apriori) || va_apriori <= 0) {
                                    stop("fixed_va prior mode requires rr_va_apriori > 0")
                                }

                                if (!is.null(z_var_sum_override)) {
                                    z_var_sum <- as.numeric(z_var_sum_override)[1]
                                } else {
                                    z_var_sum <- sum(diag(stats::var(Z_train)))
                                }
                                if (!is.finite(z_var_sum) || z_var_sum <= 0) {
                                    stop("Unable to compute positive PC variance sum for fixed_va prior mode")
                                }

                                rr_effect_var <- list(
                                    prec = list(
                                        fixed = TRUE,
                                        initial = log(1 / (va_apriori / z_var_sum))
                                    )
                                )
                                formula_str <- "y ~ 1 + f(idx, model = 'z', Z = Z_all, hyper = rr_effect_var)"
                            } else if (!identical(rr_mode, "default")) {
                                stop("Unknown rr_prior_mode. Expected 'default' or 'fixed_va'.")
                            }

                            if (use_one_step) {
                                data_df$sex <- factor(c(as.character(sex_train), as.character(sex_test)))
                                data_df$month <- factor(c(as.character(month_train), as.character(month_test)))
                                data_df$age <- as.numeric(c(age_train, age_test))
                                data_df$locality <- factor(c(as.character(locality_train), as.character(locality_test)))
                                data_df$hatch_year <- factor(c(as.character(hatch_year_train), as.character(hatch_year_test)))
                                if (!is.null(ringnr_train) && !is.null(ringnr_test)) {
                                    data_df$ringnr <- factor(c(as.character(ringnr_train), as.character(ringnr_test)))
                                }
                                if (!is.null(fhat_train) && !is.null(fhat_test)) {
                                    data_df$fhat <- as.numeric(c(fhat_train, fhat_test))
                                    formula_str <- paste0(formula_str, " + sex + month + age + fhat")
                                } else {
                                    formula_str <- paste0(formula_str, " + sex + month + age")
                                }
                                formula_str <- paste0(
                                    formula_str,
                                    " + f(locality, model='iid') + f(hatch_year, model='iid')"
                                )
                                if ("ringnr" %in% names(data_df)) {
                                    formula_str <- paste0(formula_str, " + f(ringnr, model='iid')")
                                }
                            }

                            model_formula <- stats::as.formula(formula_str)

                            thread_spec <- Sys.getenv("INLA_NUM_THREADS", unset = "")
                            if (!nzchar(thread_spec)) {
                                omp_threads <- Sys.getenv("OMP_NUM_THREADS", unset = "")
                                if (nzchar(omp_threads)) {
                                    thread_spec <- paste0(omp_threads, ":1")
                                }
                            }

                            # Empirical Bayes (use posterior mode of hyperparams instead of
                            # CCD/grid integration) gives a large speedup at minor cost in
                            # posterior-mean accuracy, which is what we need for genomic
                            # prediction. inla.mode='compact' pins the modern sparse backend.
                            inla_control_inla <- list(int.strategy = "eb")
                            # control.predictor compute=FALSE skips per-row marginal
                            # distributions; predictive means are still available via
                            # summary.linear.predictor (populated by default).
                            inla_control_predictor <- list(compute = FALSE)

                            if (nzchar(thread_spec)) {
                                fit <- INLA::inla(
                                    model_formula,
                                    family = "gaussian",
                                    data = data_df,
                                    weights = weights_all,
                                    num.threads = thread_spec,
                                    inla.mode = "compact",
                                    control.predictor = inla_control_predictor,
                                    control.inla = inla_control_inla,
                                    control.compute = list(config = FALSE),
                                    verbose = FALSE
                                )
                            } else {
                                fit <- INLA::inla(
                                    model_formula,
                                    family = "gaussian",
                                    data = data_df,
                                    weights = weights_all,
                                    inla.mode = "compact",
                                    control.predictor = inla_control_predictor,
                                    control.inla = inla_control_inla,
                                    control.compute = list(config = FALSE),
                                    verbose = FALSE
                                )
                            }

              pred_mean <- fit$summary.linear.predictor$mean
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

    # Detect long-format mode: per-record covariates with z_row mappings supplied.
    long_format = (
        one_step_train is not None
        and one_step_test is not None
        and one_step_train.get("y") is not None
        and one_step_train.get("z_row") is not None
        and one_step_test.get("z_row") is not None
    )

    # Compute the PC variance sum from the per-individual (un-replicated) Z_train
    # so the fixed-V_A prior calibration is independent of how many records each
    # individual contributes.
    z_var_sum_override: Optional[float] = None
    if rr_prior_mode == "fixed_va":
        try:
            z_var_sum_override = float(np.sum(np.var(Z_train, axis=0, ddof=1)))
        except Exception:
            z_var_sum_override = None

    train_weights_np = None if train_weights is None else np.asarray(train_weights, dtype=np.float64)
    if train_weights_np is not None and train_weights_np.shape[0] != Z_train.shape[0]:
        raise ValueError("train_weights length must match Z_train rows")

    z_row_test_np: Optional[np.ndarray] = None
    n_test_individuals = int(Z_test.shape[0])

    if long_format:
        z_row_train_np = np.asarray(one_step_train["z_row"], dtype=np.int64)
        z_row_test_np = np.asarray(one_step_test["z_row"], dtype=np.int64)
        Z_train_for_r = Z_train[z_row_train_np, :]
        Z_test_for_r = Z_test[z_row_test_np, :]
        y_train_for_r = np.asarray(one_step_train["y"], dtype=np.float64)
        weights_for_r = (
            None if train_weights_np is None else train_weights_np[z_row_train_np]
        )
        if y_train_for_r.shape[0] != Z_train_for_r.shape[0]:
            raise ValueError("Long-format y_train length must match number of train records.")
    else:
        Z_train_for_r = Z_train
        Z_test_for_r = Z_test
        y_train_for_r = y_train
        weights_for_r = train_weights_np

    with localconverter(ro.default_converter + numpy2ri.converter):
        r_Z_train = ro.conversion.py2rpy(Z_train_for_r)
        r_y_train = ro.conversion.py2rpy(y_train_for_r)
        r_Z_test = ro.conversion.py2rpy(Z_test_for_r)
        r_train_weights = ro.NULL if weights_for_r is None else ro.conversion.py2rpy(weights_for_r)

        def _cvt_opt(arr: Optional[np.ndarray]):
            if arr is None:
                return ro.NULL
            return ro.conversion.py2rpy(np.asarray(arr))

        sex_train = _cvt_opt(None if one_step_train is None else one_step_train.get("sex"))
        sex_test = _cvt_opt(None if one_step_test is None else one_step_test.get("sex"))
        month_train = _cvt_opt(None if one_step_train is None else one_step_train.get("month"))
        month_test = _cvt_opt(None if one_step_test is None else one_step_test.get("month"))
        age_train = _cvt_opt(None if one_step_train is None else one_step_train.get("age"))
        age_test = _cvt_opt(None if one_step_test is None else one_step_test.get("age"))
        fhat_train = _cvt_opt(None if one_step_train is None else one_step_train.get("f_hat"))
        fhat_test = _cvt_opt(None if one_step_test is None else one_step_test.get("f_hat"))
        locality_train = _cvt_opt(None if one_step_train is None else one_step_train.get("locality"))
        locality_test = _cvt_opt(None if one_step_test is None else one_step_test.get("locality"))
        hatch_year_train = _cvt_opt(None if one_step_train is None else one_step_train.get("hatch_year"))
        hatch_year_test = _cvt_opt(None if one_step_test is None else one_step_test.get("hatch_year"))
        ringnr_train = _cvt_opt(None if one_step_train is None else one_step_train.get("ringnr"))
        ringnr_test = _cvt_opt(None if one_step_test is None else one_step_test.get("ringnr"))
        r_rr_prior_mode = ro.StrVector([str(rr_prior_mode)])
        r_rr_va_apriori = ro.NULL if rr_va_apriori is None else ro.FloatVector([float(rr_va_apriori)])
        r_z_var_sum_override = (
            ro.NULL if z_var_sum_override is None else ro.FloatVector([float(z_var_sum_override)])
        )

    res = fn(
        r_Z_train,
        r_y_train,
        r_Z_test,
        r_train_weights,
        sex_train,
        sex_test,
        month_train,
        month_test,
        age_train,
        age_test,
        fhat_train,
        fhat_test,
        locality_train,
        locality_test,
        hatch_year_train,
        hatch_year_test,
        ringnr_train,
        ringnr_test,
        r_rr_prior_mode,
        r_rr_va_apriori,
        r_z_var_sum_override,
    )
    test_pred = np.asarray(res.rx2("test_pred"), dtype=np.float64)

    if long_format and z_row_test_np is not None:
        # R returned per-record fitted means for test rows. Aggregate to one
        # prediction per target individual by averaging across that individual's
        # records (matches Aase et al. 2025 evaluation against y-bar).
        sums = np.zeros(n_test_individuals, dtype=np.float64)
        counts = np.zeros(n_test_individuals, dtype=np.int64)
        for i, row in enumerate(z_row_test_np):
            sums[int(row)] += float(test_pred[i])
            counts[int(row)] += 1
        if np.any(counts == 0):
            missing = np.where(counts == 0)[0]
            raise ValueError(
                "Some target individuals have no test records in long-format BPCRR; "
                f"missing rows={missing.tolist()[:5]}"
            )
        return sums / counts

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
    baseline_only = bool(exp_cfg.get("baseline_only", False))
    selection_methods = [] if baseline_only else _parse_selection_methods(exp_cfg)
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
    one_step_source: Optional[Dict[str, np.ndarray]] = None,
    one_step_target: Optional[Dict[str, np.ndarray]] = None,
    rr_prior_mode: str = "default",
    rr_va_apriori: Optional[float] = None,
) -> Dict[str, float]:
    if len(train_idx) < 2:
        return {"corr_eval": 0.0, "mse_adj": float("inf")}

    one_step_train_subset = _long_format_subset(
        one_step_source,
        np.asarray(train_idx, dtype=np.int64),
        n_z_rows=Z_source.shape[0],
    )
    pred = _inla_bpcrr_predict(
        Z_train=Z_source[train_idx],
        y_train=y_source[train_idx],
        Z_test=Z_target,
        one_step_train=one_step_train_subset,
        one_step_test=one_step_target,
        rr_prior_mode=rr_prior_mode,
        rr_va_apriori=rr_va_apriori,
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
                        "bpcrr_prior_mode",
                        "bpcrr_va_apriori",
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
                    fit_time_mean_s=("fit_time_seconds", "mean"),
                    fit_time_total_s=("fit_time_seconds", "sum"),
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
    one_step_cfg = exp_cfg.get("one_step", {})
    if isinstance(one_step_cfg, bool):
        one_step_enabled = bool(one_step_cfg)
    else:
        one_step_enabled = bool(one_step_cfg.get("enabled", False))
    baseline_only = bool(exp_cfg.get("baseline_only", False))
    selection_methods = [] if baseline_only else _parse_selection_methods(exp_cfg)
    bpcrr_pev_lambda_cfg = _parse_bpcrr_pev_lambda_cfg(exp_cfg)
    bpcrr_pev_ga_cfg = _parse_bpcrr_pev_ga_cfg(exp_cfg, global_seed=global_seed)
    bpcrr_prior_mode, bpcrr_va_apriori = _parse_bpcrr_prior_cfg(exp_cfg)
    if str(bpcrr_pev_lambda_cfg["mode"]) == "paper" and bpcrr_prior_mode != "fixed_va":
        raise ValueError(
            "bpcrr_inla_experiment.bpcrr_pev_lambda_mode='paper' requires "
            "bpcrr_prior_mode='fixed_va' so sigma_u*^2 follows eq (4)"
        )
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
        trait_exp_cfg = _exp_cfg_for_trait(exp_cfg, trait_spec)
        bpcrr_prior_mode, bpcrr_va_apriori = _parse_bpcrr_prior_cfg(trait_exp_cfg)
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

        cfg_for_one_step = _config_for_trait_one_step(
            cfg=cfg,
            exp_cfg=exp_cfg,
            trait_spec=trait_spec,
            trait_paths=trait_paths,
        )
        one_step_covars = _prepare_one_step_covariates(
            config_path=config_path,
            cfg=cfg_for_one_step,
            ids=ids,
            locality_codes=locality,
            code_to_label=code_to_label,
        )
        if one_step_enabled and one_step_covars is None:
            raise RuntimeError("one_step is enabled but covariates could not be prepared")
        if one_step_enabled:
            logger.info("One-step BPCRR enabled: fixed+random effects will be included in INLA formula")
            if "adjusted" in str(trait_spec["target_column"]).lower():
                logger.warning(
                    "one_step is enabled but target_column is '%s'. "
                    "For strict one-step fitting, use an unadjusted phenotype target.",
                    trait_spec["target_column"],
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
        summary_path = trait_output / "bpcrr_inla_rank_select_summary.csv"
        completed_result_keys = _load_completed_result_keys(results_path)

        def _result_already_done(row: Dict[str, Any]) -> bool:
            return _result_key(row) in completed_result_keys

        def _record_result(row: Dict[str, Any]) -> None:
            key = _result_key(row)
            if key in completed_result_keys:
                logger.info(
                    "Skipping already completed result | trait=%s target=%s repeat=%s analysis=%s method=%s n_comp=%s n=%s",
                    row.get("trait"),
                    row.get("target_island"),
                    row.get("repeat"),
                    row.get("analysis"),
                    row.get("method"),
                    row.get("n_components"),
                    row.get("n_individuals"),
                )
                return
            _append_csv(pd.DataFrame([row]), results_path)
            completed_result_keys.add(key)
            _write_results_summary(results_path, summary_path)

        jobs: List[Dict[str, Any]] = []
        step_counts_by_repeat: Dict[tuple[int, int], np.ndarray] = {}

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

            if baseline_only:
                step_counts = np.array([], dtype=np.int64)
            elif n_train_sizes_raw is not None:
                step_counts = np.array(sorted(int(x) for x in n_train_sizes_raw), dtype=np.int64)
            else:
                locality_source = locality[source_mask]
                n_per_island = np.array([(locality_source == c).sum() for c in source_codes], dtype=np.int64)
                step_counts = np.cumsum(np.sort(n_per_island)[::-1])

            step_counts = np.unique(np.clip(step_counts, 2, n_source))
            step_counts = step_counts[step_counts < n_source]

            if len(step_counts) == 0 and not baseline_only:
                continue

            n_rank_tasks_per_k = (
                int(n_random_reps)
                + (1 if "avggrm" in selection_methods else 0)
                + (len(pc_distance_n_components_values) if "pc_distance" in selection_methods else 0)
                + (1 if "bpcrr_pev_ga" in selection_methods else 0)
            )
            n_comp_factor = max(1, len(bpcrr_n_components_values))

            for repeat_idx in range(n_repeats):
                repeat_key = (int(target_code), int(repeat_idx))
                step_counts_by_repeat[repeat_key] = step_counts.copy()

                # Baseline uses full source set once per n_components value.
                jobs.append({
                    "target_code": int(target_code),
                    "repeat_idx": int(repeat_idx),
                    "task": "baseline",
                    "k": -1,
                    "weight": float(max(2, n_source) * n_comp_factor),
                })

                if not baseline_only:
                    # Ranked subset tasks are weighted by train size (k), since runtime is roughly proportional to k.
                    per_k_weight_factor = float(n_comp_factor * max(1, n_rank_tasks_per_k))
                    for k in step_counts:
                        jobs.append({
                            "target_code": int(target_code),
                            "repeat_idx": int(repeat_idx),
                            "task": "k",
                            "k": int(k),
                            "weight": float(max(2, int(k)) * per_k_weight_factor),
                        })

        shard_bins = _assign_jobs_weighted(jobs, num_shards)
        shard_jobs = shard_bins[shard_index] if num_shards > 1 else jobs
        repeat_assignments: Dict[tuple[int, int], Dict[str, Any]] = {}
        for j in shard_jobs:
            key = (int(j["target_code"]), int(j["repeat_idx"]))
            if key not in repeat_assignments:
                repeat_assignments[key] = {"baseline": False, "step_counts": []}
            if str(j.get("task", "k")) == "baseline":
                repeat_assignments[key]["baseline"] = True
            else:
                repeat_assignments[key]["step_counts"].append(int(j.get("k", -1)))

        for key, assignment in repeat_assignments.items():
            if len(assignment["step_counts"]) > 0:
                assignment["step_counts"] = np.array(sorted(set(assignment["step_counts"])), dtype=np.int64)
            else:
                assignment["step_counts"] = np.array([], dtype=np.int64)

        logger.info(
            "Shard %d/%d assigned %d/%d jobs",
            shard_index,
            num_shards,
            len(shard_jobs),
            len(jobs),
        )

        assigned_repeat_keys = [
            key for key, a in repeat_assignments.items() if bool(a["baseline"]) or len(a["step_counts"]) > 0
        ]
        done = 0
        total = len(assigned_repeat_keys)

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

            one_step_source = None
            one_step_target = None
            if one_step_covars is not None:
                source_inds_global = np.where(source_mask)[0]
                target_inds_global = np.where(target_mask)[0]
                one_step_source = _long_format_initial_slice(
                    one_step_covars, source_inds_global, n_total_inds=len(ids),
                )
                one_step_target = _long_format_initial_slice(
                    one_step_covars, target_inds_global, n_total_inds=len(ids),
                )

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

            for repeat_idx in range(n_repeats):
                repeat_key = (int(target_code), int(repeat_idx))
                assignment = repeat_assignments.get(repeat_key)
                if assignment is None:
                    continue

                run_baseline = bool(assignment["baseline"])
                step_counts = np.asarray(assignment["step_counts"], dtype=np.int64)
                if (not run_baseline) and len(step_counts) == 0:
                    continue

                done += 1
                repeat_seed = _make_repeat_seed(global_seed, int(target_code), int(repeat_idx))
                logger.info(
                    "Job %d/%d | trait=%s target=%s repeat=%d/%d baseline=%s n_train_sizes=%d",
                    done,
                    total,
                    trait_name,
                    target_code,
                    repeat_idx + 1,
                    n_repeats,
                    str(run_baseline),
                    int(len(step_counts)),
                )

                for n_comp_req in bpcrr_n_components_values:
                    n_comp = int(min(int(n_comp_req), Z_source_bpcrr_full.shape[1]))
                    if n_comp < 1:
                        continue

                    Z_source = Z_source_bpcrr_full[:, :n_comp]
                    Z_target = Z_target_bpcrr_full[:, :n_comp]
                    eval_kwargs = {
                        "rr_prior_mode": bpcrr_prior_mode,
                        "rr_va_apriori": bpcrr_va_apriori,
                    }
                    sigma_e2_for_pev = (
                        bpcrr_pev_lambda_cfg["sigma_e2_apriori"]
                        if bpcrr_pev_lambda_cfg["sigma_e2_apriori"] is not None
                        else _estimate_sigma_e2(y_source)
                    )

                    n_ranked_fit_evals = int(len(step_counts)) * (
                        int(n_random_reps)
                        + (1 if "avggrm" in selection_methods else 0)
                        + (len(pc_distance_cache) if "pc_distance" in selection_methods else 0)
                        + (1 if "bpcrr_pev_ga" in selection_methods else 0)
                    )
                    n_fit_evals_total = (1 if run_baseline else 0) + n_ranked_fit_evals
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

                    def _log_fit_done(stage: str, start_ts: float) -> float:
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
                        return float(elapsed)

                    if run_baseline:
                        full_idx = np.arange(n_source, dtype=np.int64)
                        full_key_row = {
                            "analysis": "full_baseline",
                            "method": "full_source_unweighted",
                            "selection_method": "none",
                            "order_seed": -2,
                            "n_individuals": int(n_source),
                            "target_island": int(target_code),
                            "repeat": int(repeat_idx),
                            "trait": trait_name,
                            "n_components": int(n_comp),
                            "selection_n_components": np.nan,
                        }
                        if _result_already_done(full_key_row):
                            logger.info(
                                "Skipping completed fit | trait=%s target=%s repeat=%d n_comp=%d stage=full_baseline",
                                trait_name,
                                target_code,
                                repeat_idx,
                                n_comp,
                            )
                        else:
                            fit_started = _log_fit_start("full_baseline", len(full_idx))
                            full_eval = _evaluate_bpcrr_subset(
                                train_idx=full_idx,
                                Z_source=Z_source,
                                y_source=y_source,
                                Z_target=Z_target,
                                y_target=y_target,
                                y_eval_target=y_eval_target,
                                one_step_source=one_step_source,
                                one_step_target=one_step_target,
                                **eval_kwargs,
                            )
                            fit_time_seconds = _log_fit_done("full_baseline", fit_started)
                            full_row = {
                                **full_key_row,
                                "weighted_fit_used": False,
                                "fit_time_seconds": float(fit_time_seconds),
                                "corr_eval": float(full_eval["corr_eval"]),
                                "mse_adj": float(full_eval["mse_adj"]),
                                "target_island_name": str(target_name),
                                "repeat_seed": int(repeat_seed),
                                "bpcrr_prior_mode": str(bpcrr_prior_mode),
                                "bpcrr_va_apriori": float(bpcrr_va_apriori) if bpcrr_va_apriori is not None else np.nan,
                                "avg_grm_obj": float(np.mean(avg_grm)) if avg_grm is not None else float("nan"),
                                "pca_dist_obj": float("nan"),
                            }
                            _record_result(full_row)

                    for order_seed in range(n_random_reps):
                        rng = np.random.default_rng(repeat_seed + 500_000 + order_seed + n_comp)
                        shuffled = rng.permutation(n_source)
                        for k in step_counts:
                            n_train = int(min(int(k), n_source))
                            chosen = shuffled[:n_train]

                            stage = f"random_seed{int(order_seed)}_k{int(n_train)}"
                            rand_key_row = {
                                "analysis": "ranked_subset",
                                "method": "random_individual",
                                "selection_method": "random",
                                "order_seed": int(order_seed),
                                "n_individuals": int(n_train),
                                "target_island": int(target_code),
                                "repeat": int(repeat_idx),
                                "trait": trait_name,
                                "n_components": int(n_comp),
                                "selection_n_components": np.nan,
                            }
                            if _result_already_done(rand_key_row):
                                logger.info(
                                    "Skipping completed fit | trait=%s target=%s repeat=%d n_comp=%d stage=%s",
                                    trait_name,
                                    target_code,
                                    repeat_idx,
                                    n_comp,
                                    stage,
                                )
                                continue
                            fit_started = _log_fit_start(stage, n_train)
                            eval_result = _evaluate_bpcrr_subset(
                                train_idx=chosen,
                                Z_source=Z_source,
                                y_source=y_source,
                                Z_target=Z_target,
                                y_target=y_target,
                                y_eval_target=y_eval_target,
                                one_step_source=one_step_source,
                                one_step_target=one_step_target,
                                **eval_kwargs,
                            )
                            fit_time_seconds = _log_fit_done(stage, fit_started)
                            rand_row = {
                                **rand_key_row,
                                "weighted_fit_used": False,
                                "fit_time_seconds": float(fit_time_seconds),
                                "corr_eval": float(eval_result["corr_eval"]),
                                "mse_adj": float(eval_result["mse_adj"]),
                                "target_island_name": str(target_name),
                                "repeat_seed": int(repeat_seed),
                                "bpcrr_prior_mode": str(bpcrr_prior_mode),
                                "bpcrr_va_apriori": float(bpcrr_va_apriori) if bpcrr_va_apriori is not None else np.nan,
                                "avg_grm_obj": float("nan"),
                                "pca_dist_obj": float("nan"),
                            }
                            _record_result(rand_row)

                    for selection_method in selection_methods:
                        if selection_method == "avggrm":
                            scores = np.asarray(avg_grm, dtype=float)
                            order = np.argsort(-scores, kind="mergesort")
                            pca_distances = np.full(n_source, np.nan, dtype=float)
                            sel_n_comp_values = [np.nan]
                        elif selection_method == "pc_distance":
                            sel_n_comp_values = sorted(pc_distance_cache.keys())
                        elif selection_method == "bpcrr_pev_ga":
                            sel_n_comp_values = [float(n_comp)]
                            Z_joint = np.vstack([Z_source, Z_target])
                            K_joint, diag_joint = build_kernel(Z_joint)
                            target_idx_joint = np.arange(n_source, n_source + len(Z_target), dtype=np.int64)
                            pca_distances = np.full(n_source, np.nan, dtype=float)
                        else:
                            raise RuntimeError(f"Unhandled selection method: {selection_method}")

                        for sel_n_comp in sel_n_comp_values:
                            if selection_method == "pc_distance":
                                pca_distances = pc_distance_cache[int(sel_n_comp)]
                                scores = -pca_distances
                                order = np.argsort(pca_distances, kind="mergesort")

                            if selection_method in {"avggrm", "pc_distance"}:
                                ranks = np.empty_like(order)
                                ranks[order] = np.arange(1, len(order) + 1)

                            for k in step_counts:
                                n_train = int(min(int(k), n_source))
                                if selection_method == "bpcrr_pev_ga":
                                    def _fitness_fn(train_idx: np.ndarray) -> float:
                                        if str(bpcrr_pev_lambda_cfg["mode"]) == "paper":
                                            if bpcrr_va_apriori is None:
                                                return float("inf")
                                            z_var_sum = _sum_pc_variances(Z_source[np.asarray(train_idx, dtype=np.int64)])
                                            if not np.isfinite(z_var_sum) or z_var_sum <= 0:
                                                return float("inf")
                                            # paper-consistent lambda = sigma_e^2 / sigma_u*^2,
                                            # with sigma_u*^2 = va_apriori / sum_j Var(PC_j)
                                            lam_eff = float(sigma_e2_for_pev) * (float(z_var_sum) / float(bpcrr_va_apriori))
                                        else:
                                            lam_eff = float(bpcrr_pev_lambda_cfg["lambda_fixed"])
                                        return float(
                                            pev_mean(
                                                K=K_joint,
                                                diag_K=diag_joint,
                                                train_idx=np.asarray(train_idx, dtype=np.int64),
                                                target_idx=target_idx_joint,
                                                lam=lam_eff,
                                            )
                                        )

                                    ga_seed_token = (
                                        f"bpcrr_pev_ga|{int(repeat_seed)}|{int(target_code)}|"
                                        f"{int(n_comp)}|{int(n_train)}"
                                    )
                                    ga_seed = int.from_bytes(
                                        hashlib.blake2b(
                                            ga_seed_token.encode("utf-8"),
                                            digest_size=8,
                                        ).digest(),
                                        byteorder="little",
                                        signed=False,
                                    ) % 2_147_483_647

                                    ga_cfg = copy.deepcopy(bpcrr_pev_ga_cfg)
                                    ga_cfg.seed = ga_seed
                                    chosen, _, _ = run_ga(
                                        n_candidates=n_source,
                                        n_train=n_train,
                                        fitness_fn=_fitness_fn,
                                        cfg=ga_cfg,
                                    )
                                    chosen = np.asarray(chosen, dtype=np.int64)
                                    ranks = np.full(n_source, -1, dtype=np.int64)
                                    ranks[chosen] = np.arange(1, len(chosen) + 1)
                                    scores = np.full(n_source, np.nan, dtype=float)
                                    stage = f"bpcrr_pev_ga_k{int(n_train)}"
                                else:
                                    chosen = order[:n_train]
                                    if selection_method == "pc_distance":
                                        stage = f"pc_distance_pc{int(sel_n_comp)}_k{int(n_train)}"
                                    else:
                                        stage = f"avggrm_k{int(n_train)}"

                                if selection_method == "pc_distance":
                                    sel_n_components_value = float(sel_n_comp)
                                elif selection_method == "bpcrr_pev_ga":
                                    sel_n_components_value = float(n_comp)
                                else:
                                    sel_n_components_value = np.nan

                                key_row = {
                                    "analysis": "ranked_subset",
                                    "method": f"bpcrr_topk_{selection_method}",
                                    "selection_method": selection_method,
                                    "order_seed": -1,
                                    "n_individuals": int(n_train),
                                    "target_island": int(target_code),
                                    "repeat": int(repeat_idx),
                                    "trait": trait_name,
                                    "n_components": int(n_comp),
                                    "selection_n_components": sel_n_components_value,
                                }
                                if _result_already_done(key_row):
                                    logger.info(
                                        "Skipping completed fit | trait=%s target=%s repeat=%d n_comp=%d stage=%s",
                                        trait_name,
                                        target_code,
                                        repeat_idx,
                                        n_comp,
                                        stage,
                                    )
                                    continue
                                fit_started = _log_fit_start(stage, n_train)
                                eval_result = _evaluate_bpcrr_subset(
                                    train_idx=chosen,
                                    Z_source=Z_source,
                                    y_source=y_source,
                                    Z_target=Z_target,
                                    y_target=y_target,
                                    y_eval_target=y_eval_target,
                                    one_step_source=one_step_source,
                                    one_step_target=one_step_target,
                                    **eval_kwargs,
                                )
                                fit_time_seconds = _log_fit_done(stage, fit_started)

                                row = {
                                    **key_row,
                                    "weighted_fit_used": False,
                                    "fit_time_seconds": float(fit_time_seconds),
                                    "corr_eval": float(eval_result["corr_eval"]),
                                    "mse_adj": float(eval_result["mse_adj"]),
                                    "target_island_name": str(target_name),
                                    "repeat_seed": int(repeat_seed),
                                    "bpcrr_prior_mode": str(bpcrr_prior_mode),
                                    "bpcrr_va_apriori": float(bpcrr_va_apriori) if bpcrr_va_apriori is not None else np.nan,
                                    "avg_grm_obj": float(np.mean(scores[chosen])) if selection_method == "avggrm" else float("nan"),
                                    "pca_dist_obj": float(np.mean(pca_distances[chosen])) if selection_method == "pc_distance" else float("nan"),
                                }
                                _record_result(row)

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
                                    "selection_n_components": sel_n_components_value,
                                    "ringnr": ids_source[chosen],
                                    "ringnumber": ids_source[chosen],
                                    "source_island": locality_source[chosen].astype(int),
                                    "source_island_name": [island_label(int(c), code_to_label) for c in locality_source[chosen]],
                                    "rank": ranks[chosen].astype(int),
                                    "avg_grm": np.asarray(avg_grm[chosen], dtype=float) if avg_grm is not None else np.full(n_train, np.nan),
                                    "pca_dist": pca_distances[chosen].astype(float),
                                })
                                _append_csv(selected_df, selected_path)

        _write_results_summary(results_path, summary_path)

        logger.info("Trait '%s' complete. Output: %s", trait_name, trait_output)


if __name__ == "__main__":
    main()
