from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import optuna
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from src.avggrm_weighting import normalize_mean_one


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w**2))
    if denom <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / denom)


def _safe_standardize_against_source(
    X_source: np.ndarray,
    X_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X_source, axis=0, keepdims=True)
    std = np.std(X_source, axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return (X_source - mean) / std, (X_target - mean) / std


def suggest_importance_weighting_params(
    trial: optuna.Trial,
    weighting_space: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = weighting_space or {}

    raw_method_choices = cfg.get("method_choices", ["uniform", "pc_logistic"])
    method_choices = [str(x).lower() for x in raw_method_choices]
    if not method_choices:
        raise ValueError("search_space.importance_weighting.method_choices must contain at least one method")

    method = str(trial.suggest_categorical("iw_method", method_choices)).lower()

    weight_spec: Dict[str, Any] = {
        "name": method,
    }

    if method == "pc_logistic":
        floor_range = cfg.get("floor_range")
        if floor_range is not None:
            floor = float(
                trial.suggest_float(
                    "iw_floor",
                    float(floor_range[0]),
                    float(floor_range[1]),
                    log=bool(cfg.get("floor_log", True)),
                )
            )
        else:
            floor = float(cfg.get("floor", 1e-6))

        clip_max = None
        if "clip_max_choices" in cfg:
            clip_max = trial.suggest_categorical("iw_clip_max", cfg.get("clip_max_choices"))
        elif "clip_max" in cfg:
            clip_max = cfg.get("clip_max")

        prob_clip_range = cfg.get("prob_clip_range")
        if prob_clip_range is not None:
            prob_clip = float(
                trial.suggest_float(
                    "iw_prob_clip",
                    float(prob_clip_range[0]),
                    float(prob_clip_range[1]),
                    log=bool(cfg.get("prob_clip_log", True)),
                )
            )
        else:
            prob_clip = float(cfg.get("prob_clip", 1e-4))

        if "n_components_choices" in cfg:
            n_components = int(
                trial.suggest_categorical(
                    "iw_n_components",
                    [int(x) for x in cfg.get("n_components_choices", [])],
                )
            )
        else:
            comp_range = cfg.get("n_components_range", [5, 50])
            n_components = int(
                trial.suggest_int(
                    "iw_n_components",
                    int(comp_range[0]),
                    int(comp_range[1]),
                    step=int(cfg.get("n_components_step", 1)),
                )
            )

        if "logistic_c_choices" in cfg:
            logistic_c = float(
                trial.suggest_categorical(
                    "iw_logistic_c",
                    [float(x) for x in cfg.get("logistic_c_choices", [])],
                )
            )
        elif "logistic_c_range" in cfg:
            c_range = cfg.get("logistic_c_range", [1e-2, 1e2])
            logistic_c = float(
                trial.suggest_float(
                    "iw_logistic_c",
                    float(c_range[0]),
                    float(c_range[1]),
                    log=bool(cfg.get("logistic_c_log", True)),
                )
            )
        else:
            c_range = cfg.get("logistic_c_loguniform", [1e-2, 1e2])
            logistic_c = float(
                trial.suggest_float(
                    "iw_logistic_c",
                    float(c_range[0]),
                    float(c_range[1]),
                    log=True,
                )
            )

        pca_fit_choices = [str(x).lower() for x in cfg.get("pca_fit_choices", ["combined"])]
        pca_fit = str(trial.suggest_categorical("iw_pca_fit", pca_fit_choices)).lower()

        solver_choices = [str(x) for x in cfg.get("solver_choices", ["lbfgs"])]
        solver = str(trial.suggest_categorical("iw_solver", solver_choices))

        weight_spec.update(
            {
                "floor": float(floor),
                "clip_max": None if clip_max is None else float(clip_max),
                "prob_clip": float(prob_clip),
                "n_components": int(n_components),
                "logistic_c": float(logistic_c),
                "pca_fit": pca_fit,
                "solver": solver,
                "max_iter": int(cfg.get("max_iter", 2000)),
                "fit_intercept": bool(cfg.get("fit_intercept", True)),
                "standardize_with_source": bool(cfg.get("standardize_with_source", True)),
            }
        )

        if "rho_choices" in cfg:
            rho = float(
                trial.suggest_categorical(
                    "iw_rho",
                    [float(x) for x in cfg.get("rho_choices", [])],
                )
            )
        elif "rho_range" in cfg:
            rho_range = cfg.get("rho_range", [0.05, 1.0])
            rho = float(
                trial.suggest_float(
                    "iw_rho",
                    float(rho_range[0]),
                    float(rho_range[1]),
                    log=bool(cfg.get("rho_log", False)),
                )
            )
        else:
            rho = float(cfg.get("rho", 1.0))
        weight_spec["rho"] = float(np.clip(rho, 0.0, 1.0))
    elif method != "uniform":
        raise ValueError(f"Unknown importance-weighting method: {method}")

    trial.set_user_attr("weight_spec", weight_spec)
    return weight_spec


def compute_pc_logistic_importance_weights(
    X: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    weight_cfg: Dict[str, Any],
    *,
    feature_cols: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    method = str(weight_cfg.get("name", "uniform")).lower()
    n_train = int(len(train_idx))
    n_target = int(len(target_idx))

    if n_train == 0:
        return {
            "weights": np.array([], dtype=float),
            "raw_weights": np.array([], dtype=float),
            "target_prob_train": np.array([], dtype=float),
            "effective_sample_size": 0.0,
            "n_components_used": 0,
        }

    if method == "uniform" or n_target == 0:
        weights = np.ones(n_train, dtype=float)
        return {
            "weights": weights,
            "raw_weights": weights.copy(),
            "target_prob_train": np.full(n_train, 0.5, dtype=float),
            "effective_sample_size": effective_sample_size(weights),
            "n_components_used": 0,
        }

    if method != "pc_logistic":
        raise ValueError(f"Unknown importance-weighting method: {method}")

    X_train = np.asarray(X[train_idx], dtype=float)
    X_target = np.asarray(X[target_idx], dtype=float)
    if feature_cols is not None:
        X_train = X_train[:, feature_cols]
        X_target = X_target[:, feature_cols]

    if bool(weight_cfg.get("standardize_with_source", True)):
        X_train_proc, X_target_proc = _safe_standardize_against_source(X_train, X_target)
    else:
        X_train_proc = X_train
        X_target_proc = X_target

    pca_fit = str(weight_cfg.get("pca_fit", "combined")).lower()
    if pca_fit == "combined":
        X_pca_fit = np.vstack([X_train_proc, X_target_proc])
    elif pca_fit == "source":
        X_pca_fit = X_train_proc
    else:
        raise ValueError("importance weighting pca_fit must be one of ['combined', 'source'].")

    requested_components = int(weight_cfg.get("n_components", 10))
    max_feasible = int(min(requested_components, X_pca_fit.shape[0], X_pca_fit.shape[1]))
    if max_feasible < 1:
        weights = np.ones(n_train, dtype=float)
        return {
            "weights": weights,
            "raw_weights": weights.copy(),
            "target_prob_train": np.full(n_train, 0.5, dtype=float),
            "effective_sample_size": effective_sample_size(weights),
            "n_components_used": 0,
        }

    pca = PCA(n_components=max_feasible)
    pca.fit(X_pca_fit)
    Z_train = pca.transform(X_train_proc)
    Z_target = pca.transform(X_target_proc)

    Z_domain = np.vstack([Z_train, Z_target])
    domain_labels = np.concatenate(
        [
            np.zeros(n_train, dtype=int),
            np.ones(n_target, dtype=int),
        ]
    )

    clf = LogisticRegression(
        penalty="l2",
        C=max(float(weight_cfg.get("logistic_c", 1.0)), 1e-12),
        solver=str(weight_cfg.get("solver", "lbfgs")),
        fit_intercept=bool(weight_cfg.get("fit_intercept", True)),
        max_iter=int(weight_cfg.get("max_iter", 2000)),
    )
    clf.fit(Z_domain, domain_labels)

    target_prob_train = clf.predict_proba(Z_train)[:, 1]
    prob_clip = float(weight_cfg.get("prob_clip", 1e-4))
    target_prob_train = np.clip(target_prob_train, prob_clip, 1.0 - prob_clip)

    raw_weights = (float(n_train) / float(n_target)) * target_prob_train / (1.0 - target_prob_train)
    weights = normalize_mean_one(
        raw_weights,
        floor=float(weight_cfg.get("floor", 1e-6)),
        clip_max=weight_cfg.get("clip_max"),
    )
    pre_shrink_effective_sample_size = effective_sample_size(weights)
    rho = float(np.clip(float(weight_cfg.get("rho", 1.0)), 0.0, 1.0))
    if rho < 1.0:
        weights = (1.0 - rho) + rho * weights
        weights = normalize_mean_one(
            weights,
            floor=float(weight_cfg.get("floor", 1e-6)),
            clip_max=weight_cfg.get("clip_max"),
        )

    return {
        "weights": weights,
        "raw_weights": raw_weights,
        "target_prob_train": target_prob_train,
        "effective_sample_size": effective_sample_size(weights),
        "pre_shrink_effective_sample_size": pre_shrink_effective_sample_size,
        "n_components_used": int(max_feasible),
    }
