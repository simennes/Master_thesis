from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import optuna

from src.cv_utils import island_label


def safe_minmax(x: np.ndarray) -> np.ndarray:
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def normalize_mean_one(w: np.ndarray, floor: float = 1e-6, clip_max: Optional[float] = None) -> np.ndarray:
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


def ranks_from_desc_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def weights_from_scheme(avg_grm: np.ndarray, ranks: np.ndarray, scheme_cfg: Dict[str, Any]) -> np.ndarray:
    name = str(scheme_cfg.get("name", "uniform")).lower()
    floor = float(scheme_cfg.get("floor", 1e-6))
    clip_max = scheme_cfg.get("clip_max", None)

    s = safe_minmax(avg_grm)
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

    return normalize_mean_one(w, floor=floor, clip_max=clip_max)


def suggest_weighting_params(trial: optuna.Trial, weighting_space: Dict[str, Any]) -> Dict[str, Any]:
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


def avg_grm_train_to_target(grm_mat: np.ndarray, train_idx: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    if train_idx.size == 0:
        return np.array([], dtype=float)
    if target_idx.size == 0:
        return np.zeros(train_idx.size, dtype=float)
    block = grm_mat[np.ix_(train_idx, target_idx)]
    return np.asarray(block.mean(axis=1), dtype=float)


def compute_avggrm_weights(
    grm_mat: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    scheme_cfg: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    avg_grm = avg_grm_train_to_target(grm_mat, train_idx, target_idx)
    ranks = ranks_from_desc_scores(avg_grm)
    weights = weights_from_scheme(avg_grm, ranks, scheme_cfg)
    return avg_grm, ranks, weights


def parse_top_k_related_islands(raw_value: Any) -> Optional[int]:
    if raw_value is None or raw_value is False:
        return None

    if isinstance(raw_value, str):
        s = raw_value.strip().lower()
        if s in ("false", "none", "", "0", "all"):
            return None
        try:
            raw_value = int(s)
        except Exception as exc:
            raise ValueError("cv.inner_top_k_related_islands must be null or an integer >= 1.") from exc

    try:
        top_k = int(raw_value)
    except Exception as exc:
        raise ValueError("cv.inner_top_k_related_islands must be null or an integer >= 1.") from exc

    if top_k < 1:
        raise ValueError("cv.inner_top_k_related_islands must be null or an integer >= 1.")
    return top_k


def rank_inner_validation_islands_by_avg_grm(
    grm_mat: np.ndarray,
    locality: np.ndarray,
    idx_outer_train: np.ndarray,
    idx_outer_test: np.ndarray,
    code_to_label: Optional[Dict[int, str]],
) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    inner_islands = np.unique(locality[idx_outer_train])

    for inner_isl in inner_islands:
        inner_idx = idx_outer_train[locality[idx_outer_train] == inner_isl]
        if inner_idx.size == 0:
            continue
        avg_grm = float(grm_mat[np.ix_(inner_idx, idx_outer_test)].mean()) if idx_outer_test.size else 0.0
        rankings.append(
            {
                "island": int(inner_isl),
                "island_name": island_label(int(inner_isl), code_to_label),
                "avg_grm_to_outer_test": avg_grm,
                "n_samples": int(inner_idx.size),
            }
        )

    rankings.sort(key=lambda item: (-float(item["avg_grm_to_outer_test"]), int(item["island"])))
    return rankings
