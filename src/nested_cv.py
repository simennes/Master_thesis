from __future__ import annotations
import argparse
import json
import logging
import os
import gc
from typing import Any, Dict

import numpy as np
import optuna
import torch


# ------------------- project imports -------------------
from src.data import load_data
from src.utils import (
    set_seed, _pearson_corr, _select_top_snps_by_abs_corr, 
    _optimizer, decode_choice, make_loss, train_epochs
)
from src.models import TrainParams, make_model
from src.cv_utils import make_outer_splits, make_inner_splits, make_inner_loio_splits, island_label
from src.hyperparams import suggest_params

# ---------------------------- logging ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------- Runner (nested) ----------------------------

def run_nested_cv(config: Dict[str, Any]):
    base = config["base_train"]
    search_space = config.get("search_space", {})

    seed = int(base.get("seed", 42))
    set_seed(seed)

    # ---- Load data
    if load_data is None:
        raise RuntimeError("load_data() not found. Please provide your project loader via src.data.load_data.")

    X, y, ids, GRM_df, locality, code_to_label, y_eval = load_data(
        base["paths"],
        target_column=base.get("target_column", "y_adjusted"),
        standardize_features=base.get("standardize_features", False),
        return_locality=True,
        min_count=20,
        return_eval=True,
        eval_target_column=base.get("eval_target_column", "y_mean"),
    )
    if y_eval is None:
        y_eval = y.copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Optional island inclusion filter (by original location number)
    cv_cfg = config.get("cv", {})
    include_islands = cv_cfg.get("include_islands")
    if include_islands:
        # Normalize input to a flat python list
        if isinstance(include_islands, (list, tuple, set, np.ndarray)):
            include_list = list(include_islands)
        else:
            include_list = [include_islands]
        # Convert numpy scalars to native python types
        include_list = [x.item() if isinstance(x, np.generic) else x for x in include_list]

        # Build label->code map from code_to_label (which is code->label)
        label_to_code = {str(v): int(k) for k, v in (code_to_label or {}).items()}
        present_codes = set(np.unique(locality).astype(int).tolist())

        # Resolve requested include list into encoded codes
        include_codes = set()
        for val in include_list:
            # Try matching by original label string
            sval = str(val)
            if sval in label_to_code:
                include_codes.add(int(label_to_code[sval]))
                continue
            # Else, if it's already an encoded code number
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

        # Apply filtering consistently across all aligned arrays
        X = X[idx]
        y = y[idx]
        y_eval = y_eval[idx]
        ids = ids[idx] if ids is not None else None
        locality = locality[idx]
        if GRM_df is not None:
            GRM_df = GRM_df.iloc[idx, idx]

        # Log human-readable info
        kept_codes = sorted(set(locality.astype(int).tolist()))
        kept_labels = [(code_to_label or {}).get(int(c), str(c)) for c in kept_codes]
        logger.info(
            "Filtered to %d samples from islands (codes->labels): %s based on include_islands=%s",
            idx.size,
            ", ".join(f"{c}->{lbl}" for c, lbl in zip(kept_codes, kept_labels)),
            include_islands,
        )

    # ---- CV config
    cv_cfg = config.get("cv", {})
    strategy = cv_cfg.get("strategy", "kfold").lower()  # "kfold" or "leave_island_out"
    outer_splits = int(cv_cfg.get("n_splits", 10))
    inner_splits = int(cv_cfg.get("inner_splits", 5))
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = int(cv_cfg.get("random_state", seed))
    
    # Load predefined folds if path is provided
    predefined_folds = None
    predefined_folds_path = cv_cfg.get("predefined_folds_path", None)
    if predefined_folds_path and strategy == "kfold":
        logger.info(f"Loading predefined folds from: {predefined_folds_path}")
        with open(predefined_folds_path, "r", encoding="utf-8") as f:
            predefined_folds = json.load(f)
        outer_splits = len(predefined_folds)
        logger.info(f"Loaded {outer_splits} predefined folds")

    # Optional: run only selected outer split indices (1-based)
    sel_from_cfg = config.get("selected_splits", None)
    sel_from_cv = cv_cfg.get("selected_splits", None)
    selected_splits = sel_from_cfg if sel_from_cfg is not None else sel_from_cv
    if isinstance(selected_splits, (list, tuple, np.ndarray)):
        try:
            selected_splits = [int(x) for x in selected_splits]
        except Exception:
            selected_splits = None
    elif isinstance(selected_splits, (str,)):
        s = selected_splits.strip().lower()
        if s in ("false", "none", "", "0"):
            selected_splits = None
        else:
            try:
                parsed = json.loads(selected_splits)
                if isinstance(parsed, list):
                    selected_splits = [int(x) for x in parsed]
                else:
                    selected_splits = None
            except Exception:
                # try comma-separated
                try:
                    selected_splits = [int(x) for x in selected_splits.split(",") if x.strip()]
                except Exception:
                    selected_splits = None
    else:
        selected_splits = None

    selected_set = set(selected_splits) if selected_splits else None
    if selected_set:
        logger.info("Running only selected outer splits: %s (1-based)", sorted(selected_set))

    # ---- Optuna global knobs
    n_trials = int(config.get("n_trials", 100))
    enable_pruning = bool(config.get("enable_pruning", True))
    pruner = (
        optuna.pruners.MedianPruner(n_warmup_steps=int(config.get("pruner_warmup_epochs", 5)))
        if enable_pruning else optuna.pruners.NopPruner()
    )

    outer_results = []
    best_params_per_fold = []

    # iterate OUTER splits
    for outer_idx, (tr_idx, te_idx, isl) in enumerate(make_outer_splits(strategy, locality, outer_splits, shuffle, random_state, n=len(X), 
                                                                         predefined_folds=predefined_folds, ids=ids)):
        # Filter by selected_splits if provided (1-based indices)
        if selected_set and (outer_idx + 1) not in selected_set:
            continue
        isl_name = island_label(isl, code_to_label)
        logger.info(f"OUTER {outer_idx+1}: test_size={len(te_idx)} island={isl} ({isl_name})")
        idx_outer_train = tr_idx
        idx_outer_test = te_idx

        if strategy == "leave_island_out":
            inner_isls = np.unique(locality[idx_outer_train])
            inner_names = [island_label(int(i), code_to_label) for i in inner_isls]
            pairs = ", ".join(f"{int(i)}({n})" for i, n in zip(inner_isls, inner_names))
            logger.info(f"OUTER {outer_idx+1}: inner LOIO validation islands: {pairs}")

        # ---------- Inner study (true nested) ----------
        def objective(trial: optuna.Trial) -> float:
            tp = suggest_params(trial, search_space)
            hidden_repr = list(tp.hidden_dims) if tp.hidden_dims else None
            logger.info(
                "Trial %d | outer=%d | hidden=%s epochs=%s lr=%.3e wd=%.3e",
                trial.number,
                outer_idx + 1,
                hidden_repr,
                tp.epochs,
                tp.lr,
                tp.weight_decay,
            )

            r_vals = []
            # iterate INNER folds on OUTER-TRAIN indices
            if strategy == "leave_island_out":
                inner_plan = make_inner_loio_splits(locality, idx_outer_train)
            else:
                inner_plan = [(tr, va, None) for (tr, va) in make_inner_splits(idx_outer_train, inner_splits, shuffle, random_state)]

            for in_tr, in_va, in_isl in inner_plan:
                # Feature selection on INNER-TRAIN only (avoids leakage)
                cols = slice(None)
                if bool(trial.params.get("use_snp_selection", False)):
                    k = int(trial.params.get("num_snps", X.shape[1]))
                    cols = _select_top_snps_by_abs_corr(X[in_tr], y[in_tr], min(k, X.shape[1]))

                X_tr, X_va = X[in_tr][:, cols], X[in_va][:, cols]
                x_tr = torch.from_numpy(X_tr).to(device)
                y_tr_t = torch.from_numpy(y[in_tr]).to(device).float()
                x_va = torch.from_numpy(X_va).to(device)

                model = make_model(in_dim=X_tr.shape[1], tp=tp).to(device)
                opt = _optimizer(tp.optimizer, model.parameters(), tp.lr, tp.weight_decay)
                loss_fn = make_loss(tp.loss_name)

                # Train the MLP
                train_epochs(model, x_tr, y_tr_t, tp.epochs, opt, loss_fn)

                model.eval()
                with torch.no_grad():
                    yhat_va = model(x_va).detach().cpu().numpy().ravel()
                r_vals.append(_pearson_corr(y_eval[in_va], yhat_va))

                # cleanup per inner fold
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            return float(np.mean(r_vals)) if r_vals else 0.0

        study = optuna.create_study(direction="maximize",
                                    study_name=f"inner_outer{outer_idx}",
                                    sampler=optuna.samplers.TPESampler(seed=seed),
                                    pruner=pruner)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=bool(config.get("show_progress_bar", True)))
        best = study.best_params
        # Decode complex params (e.g., hidden_dims)
        best_decoded = dict(best)
        if "hidden_dims" in best_decoded:
            try:
                best_decoded["hidden_dims"] = decode_choice(best_decoded["hidden_dims"])  # type: ignore[arg-type]
            except Exception:
                pass
        full_best = best_decoded
        logger.info(f"OUTER {outer_idx+1} best (inner mean r={study.best_value:.4f}): {full_best}")
        best_params_per_fold.append({
            "fold": int(outer_idx + 1),
            "best_params": full_best,
            "mean_inner_r": float(study.best_value),
        })

        # ---------- Final train on OUTER-TRAIN, evaluate on OUTER-TEST ----------
        tp_final = TrainParams(
            lr=best.get("lr"), weight_decay=best.get("weight_decay"), epochs=best.get("epochs"),
            loss_name=best.get("loss"), optimizer=best.get("optimizer"),
            hidden_dims=json.loads(best.get("hidden_dims")) if isinstance(best.get("hidden_dims"), str) else best.get("hidden_dims"),
            dropout=best.get("dropout"), batch_norm=bool(best.get("batch_norm")),
        )

        # FS refit on OUTER-TRAIN only if enabled (per best params)
        cols = slice(None)
        if bool(best.get("use_snp_selection", False)):
            k = int(best.get("num_snps", X.shape[1]))
            cols = _select_top_snps_by_abs_corr(X[idx_outer_train], y[idx_outer_train], min(k, X.shape[1]))

        X_tr, X_te = X[idx_outer_train][:, cols], X[idx_outer_test][:, cols]
        x_tr = torch.from_numpy(X_tr).to(device)
        y_tr_t = torch.from_numpy(y[idx_outer_train]).to(device).float()
        x_te = torch.from_numpy(X_te).to(device)

        model = make_model(in_dim=X_tr.shape[1], tp=tp_final).to(device)
        opt = _optimizer(tp_final.optimizer, model.parameters(), tp_final.lr, tp_final.weight_decay)
        loss_fn = make_loss(tp_final.loss_name)

        # Train the MLP on outer training set
        train_epochs(model, x_tr, y_tr_t, tp_final.epochs, opt, loss_fn)

        model.eval()
        with torch.no_grad():
            yhat_te = model(x_te).detach().cpu().numpy().ravel()
        r_test = _pearson_corr(y_eval[idx_outer_test], yhat_te)

        logger.info(f"OUTER {outer_idx+1} TEST r = {r_test:.4f}")
        outer_results.append(float(r_test))

        # cleanup outer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ---- save summary
    out_dir = base["paths"].get("output_dir", "outputs/nested_cv")
    out_name = base["paths"].get("output_name", "nested_cv_unified")
    if selected_set:
        # append suffix to indicate which outer splits were run
        suffix = "splits_" + "_".join(str(i) for i in sorted(selected_set))
        out_name = f"{out_name}_{suffix}"
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "cv_strategy": strategy,
        "outer_test_corr": outer_results,
        "outer_test_corr_mean": float(np.mean(outer_results)) if outer_results else None,
        "outer_test_corr_std": float(np.std(outer_results)) if outer_results else None,
        "inner_splits": inner_splits,
        "outer_splits": int(len(selected_set)) if selected_set else outer_splits,
        "selected_splits": sorted(selected_set) if selected_set else None,
        "best_params_per_fold": best_params_per_fold,
    }
    with open(os.path.join(out_dir, f"{out_name}_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    mean_r = summary["outer_test_corr_mean"]
    std_r = summary["outer_test_corr_std"]
    if mean_r is not None and std_r is not None:
        logger.info(f"DONE. Mean OUTER r = {mean_r:.4f} ± {std_r:.4f}")
    else:
        logger.info("DONE. No outer folds were evaluated or results are empty.")


# ------------------------------ CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Nested CV with MLP")
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument(
        "--selected_splits",
        type=str,
        default=None,
        help="Optional: JSON list or comma-separated 1-based outer split indices to run (e.g., '[10,11]' or '10,11'). Use 'false' to disable.",
    )
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # CLI override for selected_splits if provided
    if args.selected_splits is not None:
        s = args.selected_splits.strip()
        if s.lower() in ("false", "none", "", "0"):
            pass
        else:
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    cfg.setdefault("cv", {})["selected_splits"] = [int(x) for x in parsed]
                else:
                    # fallback to comma-separated
                    vals = [int(x) for x in s.split(",") if x.strip()]
                    cfg.setdefault("cv", {})["selected_splits"] = vals
            except Exception:
                try:
                    vals = [int(x) for x in s.split(",") if x.strip()]
                    cfg.setdefault("cv", {})["selected_splits"] = vals
                except Exception:
                    raise ValueError("--selected_splits must be a JSON list or comma-separated integers, or 'false'.")
    run_nested_cv(cfg)


if __name__ == "__main__":
    main()
