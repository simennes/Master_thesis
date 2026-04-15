from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import optuna
import torch

from src.cv_utils import island_label, make_inner_loio_splits
from src.data import load_data
from src.hyperparams import suggest_params
from src.models import TrainParams, make_model
from src.utils import (
    _optimizer,
    _pearson_corr,
    _select_top_snps_by_abs_corr,
    decode_choice,
    make_loss,
    set_seed,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _safe_minmax(x: np.ndarray) -> np.ndarray:
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def _normalize_mean_one(w: np.ndarray, floor: float = 1e-6, clip_max: Optional[float] = None) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, float(floor))
    mean_w = float(np.mean(w))
    if mean_w <= 0:
        w = np.ones_like(w, dtype=float)
    else:
        w = w / mean_w
    if clip_max is not None:
        w = np.minimum(w, float(clip_max))
        w = w / max(float(np.mean(w)), 1e-12)
    return w


def _weights_from_scheme(avg_grm: np.ndarray, ranks: np.ndarray, scheme_cfg: Dict[str, Any]) -> np.ndarray:
    name = str(scheme_cfg.get("name", "uniform")).lower()
    floor = float(scheme_cfg.get("floor", 1e-6))
    clip_max = scheme_cfg.get("clip_max", None)

    s = _safe_minmax(avg_grm)
    n = len(avg_grm)

    if name == "uniform":
        w = np.ones(n, dtype=float)
    elif name == "linear":
        min_weight = float(scheme_cfg.get("min_weight", 0.25))
        max_weight = float(scheme_cfg.get("max_weight", 1.75))
        w = min_weight + (max_weight - min_weight) * s
    elif name == "minmax":
        eps = float(scheme_cfg.get("eps", 0.05))
        w = eps + s
    elif name == "exponential":
        beta = float(scheme_cfg.get("beta", 3.0))
        w = np.exp(beta * s)
    elif name == "top-heavy":
        top_frac = float(scheme_cfg.get("top_frac", 0.2))
        high = float(scheme_cfg.get("high", 3.0))
        low = float(scheme_cfg.get("low", 1.0))
        top_n = max(1, int(np.ceil(top_frac * n)))
        w = np.full(n, low, dtype=float)
        order = np.argsort(ranks)
        w[order[:top_n]] = high
    else:
        raise ValueError(f"Unknown weight scheme: {name}")

    return _normalize_mean_one(w, floor=floor, clip_max=clip_max)


def _suggest_weighting_params(trial: optuna.Trial, weighting_space: Dict[str, Any]) -> Dict[str, Any]:
    cfg = weighting_space or {}

    raw_choices = cfg.get("scheme_choices", ["uniform", "linear", "minmax", "exponential", "top-heavy"])
    scheme_choices = [str(x).lower() for x in raw_choices]
    if not scheme_choices:
        raise ValueError("search_space.weighting.scheme_choices must contain at least one scheme")

    scheme = trial.suggest_categorical("weight_scheme", scheme_choices)

    floor_range = cfg.get("floor_range", None)
    if floor_range is not None:
        floor = trial.suggest_float(
            "weight_floor",
            float(floor_range[0]),
            float(floor_range[1]),
            log=bool(cfg.get("floor_log", True)),
        )
    else:
        floor = float(cfg.get("floor", 1e-6))

    clip_max = None
    if "clip_max_choices" in cfg:
        clip_max = trial.suggest_categorical("weight_clip_max", cfg.get("clip_max_choices"))
    elif "clip_max" in cfg:
        clip_max = cfg.get("clip_max")

    weight_spec: Dict[str, Any] = {
        "name": scheme,
        "floor": float(floor),
        "clip_max": None if clip_max is None else float(clip_max),
    }

    if scheme == "linear":
        linear_cfg = cfg.get("linear", {})
        min_range = linear_cfg.get("min_weight_range", [0.2, 1.0])
        max_range = linear_cfg.get("max_weight_range", [1.0, 3.0])

        min_w = trial.suggest_float("weight_linear_min_weight", float(min_range[0]), float(min_range[1]))
        max_lower = max(float(max_range[0]), float(min_w) + 1e-6)
        if max_lower >= float(max_range[1]):
            max_w = max_lower
        else:
            max_w = trial.suggest_float("weight_linear_max_weight", max_lower, float(max_range[1]))

        weight_spec["min_weight"] = float(min_w)
        weight_spec["max_weight"] = float(max_w)

    elif scheme == "minmax":
        mm_cfg = cfg.get("minmax", {})
        eps_range = mm_cfg.get("eps_range", [1e-4, 0.2])
        eps = trial.suggest_float(
            "weight_minmax_eps",
            float(eps_range[0]),
            float(eps_range[1]),
            log=bool(mm_cfg.get("eps_log", False)),
        )
        weight_spec["eps"] = float(eps)

    elif scheme == "exponential":
        exp_cfg = cfg.get("exponential", {})
        beta_range = exp_cfg.get("beta_range", [0.1, 6.0])
        beta = trial.suggest_float(
            "weight_exponential_beta",
            float(beta_range[0]),
            float(beta_range[1]),
            log=bool(exp_cfg.get("beta_log", False)),
        )
        weight_spec["beta"] = float(beta)

    elif scheme == "top-heavy":
        top_cfg = cfg.get("top_heavy", {})
        frac_range = top_cfg.get("top_frac_range", [0.05, 0.5])
        low_range = top_cfg.get("low_range", [0.5, 1.0])
        high_range = top_cfg.get("high_range", [1.0, 5.0])

        top_frac = trial.suggest_float("weight_topheavy_top_frac", float(frac_range[0]), float(frac_range[1]))
        low = trial.suggest_float("weight_topheavy_low", float(low_range[0]), float(low_range[1]))
        high_lower = max(float(high_range[0]), float(low) + 1e-6)
        if high_lower >= float(high_range[1]):
            high = high_lower
        else:
            high = trial.suggest_float("weight_topheavy_high", high_lower, float(high_range[1]))

        weight_spec["top_frac"] = float(top_frac)
        weight_spec["low"] = float(low)
        weight_spec["high"] = float(high)

    trial.set_user_attr("weight_spec", weight_spec)
    return weight_spec


def _avg_grm_train_to_target(grm_mat: np.ndarray, train_idx: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    if train_idx.size == 0:
        return np.array([], dtype=float)
    if target_idx.size == 0:
        return np.zeros(train_idx.size, dtype=float)
    block = grm_mat[np.ix_(train_idx, target_idx)]
    return np.asarray(block.mean(axis=1), dtype=float)


def _train_epochs_weighted(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    opt: torch.optim.Optimizer,
    loss_name: str,
    sample_weight: Optional[np.ndarray] = None,
):
    loss_name = (loss_name or "mse").lower()
    if loss_name not in {"mse", "mae"}:
        # Keep compatibility with existing config defaults in case new names are introduced later.
        loss_fn = make_loss(loss_name)
        for _ in range(int(epochs)):
            model.train()
            opt.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()
        return

    w_t: Optional[torch.Tensor] = None
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float32)
        w_t = torch.from_numpy(w).to(x.device)

    for _ in range(int(epochs)):
        model.train()
        opt.zero_grad()
        preds = model(x)
        if loss_name == "mae":
            per_sample = torch.abs(preds - y)
        else:
            per_sample = (preds - y) ** 2

        if w_t is not None:
            loss = (per_sample * w_t).sum() / torch.clamp(w_t.sum(), min=1e-12)
        else:
            loss = per_sample.mean()

        loss.backward()
        opt.step()


def _parse_selected_splits(raw_selected: Any) -> Optional[set[int]]:
    selected_splits: Optional[list[int]]

    if isinstance(raw_selected, (list, tuple, np.ndarray)):
        try:
            selected_splits = [int(x) for x in raw_selected]
        except Exception:
            selected_splits = None
    elif isinstance(raw_selected, str):
        s = raw_selected.strip().lower()
        if s in ("false", "none", "", "0"):
            selected_splits = None
        else:
            try:
                parsed = json.loads(raw_selected)
                if isinstance(parsed, list):
                    selected_splits = [int(x) for x in parsed]
                else:
                    selected_splits = None
            except Exception:
                try:
                    selected_splits = [int(x) for x in raw_selected.split(",") if x.strip()]
                except Exception:
                    selected_splits = None
    else:
        selected_splits = None

    return set(selected_splits) if selected_splits else None


def _apply_include_islands_filter(
    X: np.ndarray,
    y: np.ndarray,
    y_eval: np.ndarray,
    ids: Optional[np.ndarray],
    locality: np.ndarray,
    code_to_label: Optional[Dict[int, str]],
    grm_df,
    include_islands: Any,
):
    if not include_islands:
        return X, y, y_eval, ids, locality, grm_df

    if isinstance(include_islands, (list, tuple, set, np.ndarray)):
        include_list = list(include_islands)
    else:
        include_list = [include_islands]
    include_list = [x.item() if isinstance(x, np.generic) else x for x in include_list]

    label_to_code = {str(v): int(k) for k, v in (code_to_label or {}).items()}
    present_codes = set(np.unique(locality).astype(int).tolist())

    include_codes: set[int] = set()
    for val in include_list:
        sval = str(val)
        if sval in label_to_code:
            include_codes.add(int(label_to_code[sval]))
            continue
        try:
            ival = int(val)
            if ival in present_codes:
                include_codes.add(ival)
        except Exception:
            pass

    if not include_codes:
        available = [f"{c}:{(code_to_label or {}).get(int(c), '?')}" for c in sorted(present_codes)]
        raise ValueError(
            f"include_islands={include_islands} did not match any samples after mapping. "
            f"Available codes/labels: {available}"
        )

    mask = np.isin(locality.astype(int), np.fromiter(include_codes, dtype=int))
    idx = np.where(mask)[0]
    if idx.size == 0:
        available = [f"{c}:{(code_to_label or {}).get(int(c), '?')}" for c in sorted(present_codes)]
        raise ValueError(
            f"include_islands={include_islands} filtered out all samples. "
            f"Matched codes={sorted(include_codes)}. Available codes/labels: {available}"
        )

    X = X[idx]
    y = y[idx]
    y_eval = y_eval[idx]
    ids = ids[idx] if ids is not None else None
    locality = locality[idx]
    if grm_df is not None:
        grm_df = grm_df.iloc[idx, idx]

    kept_codes = sorted(set(locality.astype(int).tolist()))
    kept_labels = [(code_to_label or {}).get(int(c), str(c)) for c in kept_codes]
    logger.info(
        "Filtered to %d samples from islands (codes->labels): %s based on include_islands=%s",
        idx.size,
        ", ".join(f"{c}->{lbl}" for c, lbl in zip(kept_codes, kept_labels)),
        include_islands,
    )

    return X, y, y_eval, ids, locality, grm_df


def run_nested_cv_avggrm_weighted(config: Dict[str, Any]):
    base = config["base_train"]
    search_space = config.get("search_space", {})
    weighting_space = search_space.get("weighting", {})

    seed = int(base.get("seed", config.get("seed", 42)))
    set_seed(seed)

    X, y, ids, grm_df, locality, code_to_label, y_eval = load_data(
        base["paths"],
        target_column=base.get("target_column", config.get("target_column", "y_adjusted")),
        standardize_features=base.get("standardize_features", config.get("standardize_features", False)),
        return_locality=True,
        min_count=int(base.get("min_count", config.get("min_count", 20))),
        return_eval=True,
        eval_target_column=base.get("eval_target_column", config.get("eval_target_column", "y_mean")),
    )
    if y_eval is None:
        y_eval = y.copy()

    cv_cfg = config.get("cv", {})
    X, y, y_eval, ids, locality, grm_df = _apply_include_islands_filter(
        X=X,
        y=y,
        y_eval=y_eval,
        ids=ids,
        locality=locality,
        code_to_label=code_to_label,
        grm_df=grm_df,
        include_islands=cv_cfg.get("include_islands"),
    )

    # AvgGRM weighting requires a GRM matrix when non-uniform schemes are considered.
    scheme_choices = [
        str(x).lower()
        for x in weighting_space.get("scheme_choices", ["uniform", "linear", "minmax", "exponential", "top-heavy"])
    ]
    non_uniform = {s for s in scheme_choices if s != "uniform"}
    if non_uniform and grm_df is None:
        raise ValueError(
            "AvgGRM weighting requested but GRM was not loaded. "
            "Set base_train.paths.grm_rds (or paths.grm_rds) in your config, "
            "or limit search_space.weighting.scheme_choices to ['uniform']."
        )

    grm_mat = None
    if grm_df is not None:
        grm_mat = grm_df.to_numpy(dtype=np.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    strategy = str(cv_cfg.get("strategy", "leave_island_out")).lower()
    if strategy != "leave_island_out":
        raise ValueError(
            "This runner is LOIO-only. Set cv.strategy='leave_island_out'."
        )

    selected_raw = config.get("selected_splits", None)
    if selected_raw is None:
        selected_raw = cv_cfg.get("selected_splits", None)
    selected_set = _parse_selected_splits(selected_raw)
    if selected_set:
        logger.info("Running only selected outer splits: %s (1-based)", sorted(selected_set))

    n_trials = int(config.get("n_trials", 100))
    enable_pruning = bool(config.get("enable_pruning", True))
    pruner = (
        optuna.pruners.MedianPruner(n_warmup_steps=int(config.get("pruner_warmup_epochs", 5)))
        if enable_pruning
        else optuna.pruners.NopPruner()
    )

    outer_results: list[float] = []
    best_params_per_fold: list[dict[str, Any]] = []
    per_fold_metrics: list[dict[str, Any]] = []

    unique_islands = np.unique(locality)
    for outer_idx, isl in enumerate(unique_islands):
        tr_idx = np.where(locality != isl)[0]
        te_idx = np.where(locality == isl)[0]
        if selected_set and (outer_idx + 1) not in selected_set:
            continue

        isl_name = island_label(isl, code_to_label)
        logger.info("OUTER %d: test_size=%d island=%s (%s)", outer_idx + 1, len(te_idx), isl, isl_name)

        idx_outer_train = tr_idx
        idx_outer_test = te_idx

        inner_isls = np.unique(locality[idx_outer_train])
        inner_names = [island_label(int(i), code_to_label) for i in inner_isls]
        pairs = ", ".join(f"{int(i)}({n})" for i, n in zip(inner_isls, inner_names))
        logger.info("OUTER %d: inner LOIO validation islands: %s", outer_idx + 1, pairs)

        def objective(trial: optuna.Trial) -> float:
            tp = suggest_params(trial, search_space)
            weight_spec = _suggest_weighting_params(trial, weighting_space)

            hidden_repr = list(tp.hidden_dims) if tp.hidden_dims else None
            logger.info(
                "Trial %d | outer=%d | hidden=%s epochs=%s lr=%.3e wd=%.3e weight=%s",
                trial.number,
                outer_idx + 1,
                hidden_repr,
                tp.epochs,
                tp.lr,
                tp.weight_decay,
                weight_spec,
            )

            inner_plan = make_inner_loio_splits(locality, idx_outer_train)

            r_vals: list[float] = []
            for in_tr, in_va, in_isl in inner_plan:
                if in_tr.size < 2 or in_va.size == 0:
                    logger.warning(
                        "Skipping inner fold with train=%d val=%d (outer=%d, inner_island=%s)",
                        in_tr.size,
                        in_va.size,
                        outer_idx + 1,
                        in_isl,
                    )
                    continue

                cols = slice(None)
                if bool(trial.params.get("use_snp_selection", False)):
                    k = int(trial.params.get("num_snps", X.shape[1]))
                    cols = _select_top_snps_by_abs_corr(X[in_tr], y[in_tr], min(k, X.shape[1]))

                X_tr, X_va = X[in_tr][:, cols], X[in_va][:, cols]
                x_tr = torch.from_numpy(X_tr).to(device)
                y_tr_t = torch.from_numpy(y[in_tr]).to(device).float()
                x_va = torch.from_numpy(X_va).to(device)

                if weight_spec["name"] == "uniform":
                    train_weights = np.ones(len(in_tr), dtype=float)
                else:
                    if grm_mat is None:
                        raise RuntimeError("GRM matrix is required for non-uniform AvgGRM weighting")
                    avg_grm_tr = _avg_grm_train_to_target(grm_mat, in_tr, in_va)
                    order = np.argsort(-avg_grm_tr, kind="mergesort")
                    ranks = np.empty_like(order)
                    ranks[order] = np.arange(1, len(order) + 1)
                    train_weights = _weights_from_scheme(avg_grm_tr, ranks, weight_spec)

                model = make_model(in_dim=X_tr.shape[1], tp=tp).to(device)
                opt = _optimizer(tp.optimizer, model.parameters(), tp.lr, tp.weight_decay)

                _train_epochs_weighted(
                    model=model,
                    x=x_tr,
                    y=y_tr_t,
                    epochs=tp.epochs,
                    opt=opt,
                    loss_name=tp.loss_name,
                    sample_weight=train_weights,
                )

                model.eval()
                with torch.no_grad():
                    yhat_va = model(x_va).detach().cpu().numpy().ravel()
                r_vals.append(_pearson_corr(y_eval[in_va], yhat_va))

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            return float(np.mean(r_vals)) if r_vals else 0.0

        study = optuna.create_study(
            direction="maximize",
            study_name=f"inner_outer{outer_idx}",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=pruner,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=bool(config.get("show_progress_bar", True)),
        )

        best = study.best_params
        best_decoded = dict(best)
        if "hidden_dims" in best_decoded:
            try:
                best_decoded["hidden_dims"] = decode_choice(best_decoded["hidden_dims"])
            except Exception:
                pass

        best_weight_spec = dict(study.best_trial.user_attrs.get("weight_spec", {"name": "uniform"}))

        full_best = dict(best_decoded)
        full_best["weighting"] = best_weight_spec

        logger.info(
            "OUTER %d best (inner mean r=%.4f): model=%s weighting=%s",
            outer_idx + 1,
            study.best_value,
            {k: v for k, v in best_decoded.items() if k != "weighting"},
            best_weight_spec,
        )

        best_params_per_fold.append(
            {
                "fold": int(outer_idx + 1),
                "best_params": full_best,
                "mean_inner_r": float(study.best_value),
            }
        )

        tp_final = TrainParams(
            lr=best.get("lr"),
            weight_decay=best.get("weight_decay"),
            epochs=best.get("epochs"),
            loss_name=best.get("loss"),
            optimizer=best.get("optimizer"),
            hidden_dims=json.loads(best.get("hidden_dims"))
            if isinstance(best.get("hidden_dims"), str)
            else best.get("hidden_dims"),
            dropout=best.get("dropout"),
            batch_norm=bool(best.get("batch_norm")),
        )

        cols = slice(None)
        if bool(best.get("use_snp_selection", False)):
            k = int(best.get("num_snps", X.shape[1]))
            cols = _select_top_snps_by_abs_corr(X[idx_outer_train], y[idx_outer_train], min(k, X.shape[1]))

        X_tr, X_te = X[idx_outer_train][:, cols], X[idx_outer_test][:, cols]
        x_tr = torch.from_numpy(X_tr).to(device)
        y_tr_t = torch.from_numpy(y[idx_outer_train]).to(device).float()
        x_te = torch.from_numpy(X_te).to(device)

        if best_weight_spec.get("name", "uniform") == "uniform":
            final_train_weights = np.ones(len(idx_outer_train), dtype=float)
        else:
            if grm_mat is None:
                raise RuntimeError("GRM matrix is required for non-uniform AvgGRM weighting")
            avg_grm_outer = _avg_grm_train_to_target(grm_mat, idx_outer_train, idx_outer_test)
            order_outer = np.argsort(-avg_grm_outer, kind="mergesort")
            ranks_outer = np.empty_like(order_outer)
            ranks_outer[order_outer] = np.arange(1, len(order_outer) + 1)
            final_train_weights = _weights_from_scheme(avg_grm_outer, ranks_outer, best_weight_spec)

        model = make_model(in_dim=X_tr.shape[1], tp=tp_final).to(device)
        opt = _optimizer(tp_final.optimizer, model.parameters(), tp_final.lr, tp_final.weight_decay)
        _train_epochs_weighted(
            model=model,
            x=x_tr,
            y=y_tr_t,
            epochs=tp_final.epochs,
            opt=opt,
            loss_name=tp_final.loss_name,
            sample_weight=final_train_weights,
        )

        model.eval()
        with torch.no_grad():
            yhat_te = model(x_te).detach().cpu().numpy().ravel()
        r_test = _pearson_corr(y_eval[idx_outer_test], yhat_te)

        logger.info("OUTER %d TEST r = %.4f", outer_idx + 1, r_test)
        outer_results.append(float(r_test))
        per_fold_metrics.append(
            {
                "fold": int(outer_idx + 1),
                "test_corr": float(r_test),
                "test_size": int(len(idx_outer_test)),
                "test_island": None if isl is None else int(isl),
                "test_island_name": str(isl_name),
                "weighting": best_weight_spec,
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    out_dir = base["paths"].get("output_dir", "outputs/nested_cv")
    out_name = base["paths"].get("output_name", "nested_cv_avggrm_weighted")
    if selected_set:
        suffix = "splits_" + "_".join(str(i) for i in sorted(selected_set))
        out_name = f"{out_name}_{suffix}"

    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "cv_strategy": strategy,
        "outer_test_corr": outer_results,
        "outer_test_corr_mean": float(np.mean(outer_results)) if outer_results else None,
        "outer_test_corr_std": float(np.std(outer_results)) if outer_results else None,
        "inner_strategy": "leave_island_out",
        "outer_splits": int(len(selected_set)) if selected_set else int(len(unique_islands)),
        "selected_splits": sorted(selected_set) if selected_set else None,
        "best_params_per_fold": best_params_per_fold,
        "per_fold_metrics": per_fold_metrics,
        "weighting_scheme_choices": scheme_choices,
    }

    out_path = os.path.join(out_dir, f"{out_name}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    mean_r = summary["outer_test_corr_mean"]
    std_r = summary["outer_test_corr_std"]
    if mean_r is not None and std_r is not None:
        logger.info("DONE. Mean OUTER r = %.4f +- %.4f", mean_r, std_r)
    else:
        logger.info("DONE. No outer folds were evaluated or results are empty.")
    logger.info("Saved summary to: %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Nested CV (MLP) with AvgGRM-weight hyperparameter tuning")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--selected_splits",
        type=str,
        default=None,
        help="Optional: JSON list or comma-separated 1-based outer split indices to run (e.g., '[10,11]' or '10,11'). Use 'false' to disable.",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.selected_splits is not None:
        s = args.selected_splits.strip()
        if s.lower() not in ("false", "none", "", "0"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    cfg.setdefault("cv", {})["selected_splits"] = [int(x) for x in parsed]
                else:
                    vals = [int(x) for x in s.split(",") if x.strip()]
                    cfg.setdefault("cv", {})["selected_splits"] = vals
            except Exception:
                try:
                    vals = [int(x) for x in s.split(",") if x.strip()]
                    cfg.setdefault("cv", {})["selected_splits"] = vals
                except Exception as exc:
                    raise ValueError(
                        "--selected_splits must be a JSON list or comma-separated integers, or 'false'."
                    ) from exc

    run_nested_cv_avggrm_weighted(cfg)


if __name__ == "__main__":
    main()
