"""
TracIn influence scoring for training set optimization.

Implements TracInCP (checkpoint-based TracIn) as described in:
"Estimating Training Data Influence by Tracing Gradient Descent" (Pruthi et al., 2020)

Supports:
- Full gradient dot product computation
- Efficient last-layer-only gradient approximation
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Stores model state and metadata at a checkpoint."""
    epoch: int
    state_dict: Dict[str, Any]
    learning_rate: float
    batch_size: int = 64  # For minibatch SGD approximation: η/|B|


def compute_per_sample_gradients_full(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    """
    Compute per-sample gradients for all model parameters.
    
    Returns a tensor of shape (n_samples, total_params) where each row
    contains the flattened gradients for that sample.
    
    Note: This is memory-intensive for large models and many samples.
    """
    model.eval()
    n_samples = x.shape[0]
    
    # Get total number of parameters
    param_list = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in param_list)
    
    gradients = torch.zeros(n_samples, total_params, device=x.device, dtype=x.dtype)
    
    for i in range(n_samples):
        model.zero_grad()
        xi = x[i:i+1]
        yi = y[i:i+1]
        pred = model(xi)
        loss = loss_fn(pred, yi)
        loss.backward()
        
        # Flatten all gradients into a single vector
        grad_flat = torch.cat([p.grad.flatten() for p in param_list])
        gradients[i] = grad_flat
    
    return gradients


def compute_per_sample_gradients_lastlayer(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute last-layer gradient components for efficient TracIn.
    
    For a linear output layer: y_hat = W @ h + b
    The gradient w.r.t. W is: (y_hat - y) * h^T
    The gradient w.r.t. b is: (y_hat - y)
    
    For efficient aggregation, we return:
    - h: hidden representations (n_samples, hidden_dim)
    - delta: residuals (y_hat - y) for MSE, shape (n_samples,)
    
    The influence score can then be computed as:
    score(i) ∝ delta_i * (h_i · S + s_0)
    where S = sum over cal of (delta_v * h_v) and s_0 = sum over cal of delta_v
    """
    model.eval()
    
    # Get the hidden representations before the final layer
    # We need to modify the forward pass to capture the pre-output activations
    hidden_reps = []
    predictions = []
    
    def hook_fn(module, input, output):
        # input[0] is the hidden representation fed to out_lin
        hidden_reps.append(input[0].detach())
    
    # Register hook on the output linear layer
    hook = model.out_lin.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        pred = model(x)
        predictions = pred.detach()
    
    hook.remove()
    
    h = hidden_reps[0]  # (n_samples, hidden_dim)
    delta = (predictions - y)  # (n_samples,) - residuals for MSE loss
    
    return h, delta


def compute_tracin_scores_full(
    checkpoints: List[Checkpoint],
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_cal: torch.Tensor,
    y_cal: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """
    Compute TracIn influence scores using full gradient dot products.
    
    For each training sample i, compute:
    score(i) = sum over checkpoints k of:
               sum over calibration samples v of:
               eta_k * grad_loss(w_k; x_i, y_i) · grad_loss(w_k; x_v, y_v)
    
    Returns:
        scores: np.ndarray of shape (n_train,) with influence scores
    """
    n_train = x_train.shape[0]
    
    scores = np.zeros(n_train, dtype=np.float64)
    
    for ckpt in checkpoints:
        logger.info(f"Processing checkpoint at epoch {ckpt.epoch}")
        
        # Load checkpoint state
        model.load_state_dict(ckpt.state_dict)
        model.to(device)
        
        # Compute gradients for training samples
        train_grads = compute_per_sample_gradients_full(
            model, x_train, y_train, loss_fn
        )  # (n_train, n_params)
        
        # Compute gradients for calibration samples
        cal_grads = compute_per_sample_gradients_full(
            model, x_cal, y_cal, loss_fn
        )  # (n_cal, n_params)
        
        # Compute dot products: (n_train, n_params) @ (n_params, n_cal) -> (n_train, n_cal)
        dot_products = train_grads @ cal_grads.T
        
        # Sum over calibration samples and scale by learning rate / batch_size
        # Minibatch SGD approximation: score_contribution = (eta_k / |B|) * sum_v(dot_products[:, v])
        scale = ckpt.learning_rate / ckpt.batch_size
        ckpt_scores = scale * dot_products.sum(dim=1).cpu().numpy()
        scores += ckpt_scores
        
        # Free memory
        del train_grads, cal_grads, dot_products
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return scores


def compute_tracin_scores_lastlayer(
    checkpoints: List[Checkpoint],
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_cal: torch.Tensor,
    y_cal: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Compute TracIn influence scores using last-layer gradients only.
    
    This is much more efficient than full gradients. For each checkpoint:
    
    1. Compute hidden representations h and residuals delta for all samples
    2. Aggregate calibration: S = sum(delta_v * h_v), s_0 = sum(delta_v)
    3. For each training sample: score(i) += eta_k * delta_i * (h_i · S + s_0)
    
    Returns:
        scores: np.ndarray of shape (n_train,) with influence scores
    """
    n_train = x_train.shape[0]
    
    scores = np.zeros(n_train, dtype=np.float64)
    
    for ckpt in checkpoints:
        logger.info(f"Processing checkpoint at epoch {ckpt.epoch} (last-layer mode)")
        
        # Load checkpoint state
        model.load_state_dict(ckpt.state_dict)
        model.to(device)
        
        # Get hidden reps and deltas for calibration set
        h_cal, delta_cal = compute_per_sample_gradients_lastlayer(
            model, x_cal, y_cal
        )  # h_cal: (n_cal, hidden_dim), delta_cal: (n_cal,)
        
        # Aggregate over calibration set
        # S = sum_v (delta_v * h_v) -> shape (hidden_dim,)
        S = (delta_cal.unsqueeze(1) * h_cal).sum(dim=0)  # (hidden_dim,)
        # s_0 = sum_v (delta_v) -> scalar
        s_0 = delta_cal.sum()
        
        # Get hidden reps and deltas for training set
        h_train, delta_train = compute_per_sample_gradients_lastlayer(
            model, x_train, y_train
        )  # h_train: (n_train, hidden_dim), delta_train: (n_train,)
        
        # Compute influence: score(i) = (eta_k / |B|) * delta_i * (h_i · S + s_0)
        # Using minibatch SGD approximation from TracIn paper
        h_dot_S = (h_train * S.unsqueeze(0)).sum(dim=1)  # (n_train,)
        scale = ckpt.learning_rate / ckpt.batch_size
        ckpt_scores = scale * delta_train * (h_dot_S + s_0)
        
        scores += ckpt_scores.cpu().numpy()
        
        # Free memory
        del h_cal, delta_cal, h_train, delta_train, S, h_dot_S
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return scores


def compute_tracin_scores(
    checkpoints: List[Checkpoint],
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_cal: torch.Tensor,
    y_cal: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    mode: str = "lastlayer",
) -> np.ndarray:
    """
    Main interface for computing TracIn scores.
    
    Parameters
    ----------
    checkpoints : List[Checkpoint]
        List of model checkpoints saved during training
    model : nn.Module
        The model architecture (will be loaded with checkpoint weights)
    x_train, y_train : torch.Tensor
        Training data (source individuals)
    x_cal, y_cal : torch.Tensor
        Calibration data (target island calibration subset)
    loss_fn : nn.Module
        Loss function used for training
    device : torch.device
        Device to use for computation
    mode : str
        "full" for full gradient computation, "lastlayer" for efficient approximation
        
    Returns
    -------
    scores : np.ndarray
        Influence scores for each training sample
    """
    if mode == "full":
        return compute_tracin_scores_full(
            checkpoints, model, x_train, y_train, x_cal, y_cal, loss_fn, device
        )
    elif mode == "lastlayer":
        return compute_tracin_scores_lastlayer(
            checkpoints, model, x_train, y_train, x_cal, y_cal, device
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'full' or 'lastlayer'.")


def rank_by_influence(
    scores: np.ndarray,
    ids: np.ndarray,
    localities: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rank training samples by influence score (ascending order).
    
    Lower scores = less beneficial / potentially harmful training examples.
    Higher scores = more beneficial training examples.
    
    Returns indices sorted by score (ascending).
    """
    return np.argsort(scores)
