"""
In-run first-order data valuation utilities.

This module implements a one-run, checkpoint-accumulated scorer that reuses
existing TracIn gradient primitives but adds:
- checkpoint windowing (all steps or early window up to best cal loss)
- per-checkpoint weighting based on calibration-loss improvement
- optional score sign flip so higher score means more beneficial sample

The implementation is intentionally lightweight and compatible with the current
MLP + SGD workflow.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .tracin import Checkpoint, compute_tracin_scores

logger = logging.getLogger(__name__)


def _checkpoint_cal_losses(
    checkpoints: List[Checkpoint],
    model: nn.Module,
    x_cal: torch.Tensor,
    y_cal: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Evaluate calibration loss at each checkpoint."""
    if len(checkpoints) == 0:
        return np.empty(0, dtype=np.float64)

    losses = np.zeros(len(checkpoints), dtype=np.float64)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, ckpt in enumerate(checkpoints):
            model.load_state_dict(ckpt.state_dict)
            pred = model(x_cal)
            losses[i] = float(loss_fn(pred, y_cal).item())

    return losses


def _select_checkpoint_indices(
    cal_losses: np.ndarray,
    window: str,
) -> np.ndarray:
    """
    Select checkpoint indices for score accumulation.

    Parameters
    ----------
    cal_losses : np.ndarray
        Calibration losses for each checkpoint in temporal order.
    window : str
        - "all": use all checkpoints
        - "best_cal": use checkpoints from start through best-cal-loss checkpoint
    """
    if cal_losses.size == 0:
        return np.empty(0, dtype=np.int64)

    window_norm = str(window).lower()
    if window_norm == "all":
        return np.arange(cal_losses.size, dtype=np.int64)

    if window_norm == "best_cal":
        best_idx = int(np.argmin(cal_losses))
        return np.arange(best_idx + 1, dtype=np.int64)

    raise ValueError(f"Unknown in-run window: {window}")


def _build_step_weights(
    cal_losses: np.ndarray,
    selected_idx: np.ndarray,
    weight_mode: str,
) -> np.ndarray:
    """
    Build per-checkpoint weights for accumulation.

    Modes
    -----
    - "uniform": equal weights
    - "improvement": weight by positive loss drop from previous selected checkpoint
    """
    if selected_idx.size == 0:
        return np.empty(0, dtype=np.float64)

    mode = str(weight_mode).lower()
    if mode == "uniform":
        w = np.ones(selected_idx.size, dtype=np.float64)
    elif mode == "improvement":
        losses = cal_losses[selected_idx]
        w = np.ones_like(losses, dtype=np.float64)
        if losses.size > 1:
            prev = losses[:-1]
            curr = losses[1:]
            deltas = np.maximum(prev - curr, 0.0)
            w[1:] = deltas
        # If all later deltas are zero (flat/noisy), fallback to uniform.
        if float(np.sum(w)) <= 1e-12:
            w = np.ones_like(w, dtype=np.float64)
    else:
        raise ValueError(f"Unknown in-run weight_mode: {weight_mode}")

    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return np.ones_like(w, dtype=np.float64)
    return w


def compute_inrun_firstorder_scores(
    checkpoints: List[Checkpoint],
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_cal: torch.Tensor,
    y_cal: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    mode: str = "lastlayer",
    window: str = "best_cal",
    weight_mode: str = "improvement",
    flip_sign: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute one-run first-order scores from a single training run.

        The scorer accumulates per-checkpoint first-order contributions computed via
        existing TracIn primitives, then applies:
      1) checkpoint selection window
      2) optional loss-improvement weighting
            3) optional sign flip to match a downstream ranking convention

    Returns
    -------
    scores : np.ndarray
        Per-training-sample scores (shape: n_train).
    meta : dict
        Metadata with checkpoint losses, selected checkpoints, and weights.
    """
    if len(checkpoints) == 0:
        raise ValueError("No checkpoints provided for in-run scoring")

    cal_losses = _checkpoint_cal_losses(
        checkpoints=checkpoints,
        model=model,
        x_cal=x_cal,
        y_cal=y_cal,
        loss_fn=loss_fn,
        device=device,
    )

    selected_idx = _select_checkpoint_indices(cal_losses=cal_losses, window=window)
    if selected_idx.size == 0:
        raise ValueError("No checkpoints selected for in-run scoring")

    step_weights = _build_step_weights(
        cal_losses=cal_losses,
        selected_idx=selected_idx,
        weight_mode=weight_mode,
    )

    n_train = int(x_train.shape[0])
    acc = np.zeros(n_train, dtype=np.float64)

    logger.info(
        "In-run scoring: mode=%s, window=%s, weight_mode=%s, selected=%d/%d checkpoints",
        mode,
        window,
        weight_mode,
        int(selected_idx.size),
        int(len(checkpoints)),
    )

    total_weight = float(np.sum(step_weights))
    for pos, ckpt_i in enumerate(selected_idx.tolist()):
        ckpt = checkpoints[int(ckpt_i)]
        one_ckpt_scores = compute_tracin_scores(
            checkpoints=[ckpt],
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_cal=x_cal,
            y_cal=y_cal,
            loss_fn=loss_fn,
            device=device,
            mode=mode,
        )
        acc += float(step_weights[pos]) * one_ckpt_scores

    scores = acc / max(total_weight, 1e-12)
    if flip_sign:
        logger.warning(
            "flip_sign=True: in-run scores are multiplied by -1. "
            "Ensure downstream ranking logic matches this polarity."
        )
        scores = -scores

    meta: Dict[str, Any] = {
        "method": "inrun_firstorder",
        "mode": str(mode),
        "window": str(window),
        "weight_mode": str(weight_mode),
        "flip_sign": bool(flip_sign),
        "checkpoint_epochs": [int(c.epoch) for c in checkpoints],
        "cal_losses": [float(v) for v in cal_losses.tolist()],
        "selected_checkpoint_indices": [int(v) for v in selected_idx.tolist()],
        "selected_checkpoint_epochs": [int(checkpoints[int(i)].epoch) for i in selected_idx.tolist()],
        "step_weights": [float(v) for v in step_weights.tolist()],
    }
    return scores, meta
