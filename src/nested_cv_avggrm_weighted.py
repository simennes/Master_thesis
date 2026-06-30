"""Shared nested-CV helpers used by the MLP runner.

Originally this module hosted an avgGRM-weighted nested-CV entry-point of its
own. That runner was superseded by ``src.nested_cv_avggrm_weighted_unified``
(for E6 ridge / BPCRR) and ``src.nested_cv_importance_weighted_mlp`` (for E8).
What remains here are three small helpers that the MLP runner still imports.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.utils import make_loss

logger = logging.getLogger(__name__)


def _train_epochs_weighted(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    opt: torch.optim.Optimizer,
    loss_name: str,
    sample_weight: Optional[np.ndarray] = None,
):
    """Train an MLP for `epochs` epochs with optional per-sample weights.

    Only the MSE and MAE losses get the explicit weighted reduction; other
    losses fall back to the unweighted ``make_loss`` implementation.
    """
    loss_name = (loss_name or "mse").lower()
    if loss_name not in {"mse", "mae"}:
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
        per_sample = torch.abs(preds - y) if loss_name == "mae" else (preds - y) ** 2
        if w_t is not None:
            loss = (per_sample * w_t).sum() / torch.clamp(w_t.sum(), min=1e-12)
        else:
            loss = per_sample.mean()
        loss.backward()
        opt.step()


def _parse_selected_splits(raw_selected: Any) -> Optional[set[int]]:
    """Parse the `selected_splits` config / CLI value into a set of 1-based ints."""
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
    """Restrict (X, y, y_eval, ids, locality, GRM) to the requested island set.

    Returns the original arrays unchanged when ``include_islands`` is empty / None.
    """
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
