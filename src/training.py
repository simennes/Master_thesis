"""
Training utilities with checkpoint saving for TracIn.

Provides training functions that save model checkpoints at specified epochs
for use in TracIn influence computation.
"""
from __future__ import annotations
import copy
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .tracin import Checkpoint

logger = logging.getLogger(__name__)


def get_checkpoint_epochs(total_epochs: int, n_checkpoints: int = 5) -> List[int]:
    """
    Compute epoch numbers at which to save checkpoints.
    
    By default, saves at [20%, 40%, 60%, 80%, 100%] of training.
    
    Parameters
    ----------
    total_epochs : int
        Total number of training epochs
    n_checkpoints : int
        Number of checkpoints to save
        
    Returns
    -------
    List[int]
        Epoch numbers (1-indexed) at which to save checkpoints
    """
    if n_checkpoints <= 0:
        return []
    if n_checkpoints == 1:
        return [total_epochs]
    
    # Evenly spaced checkpoints
    fractions = np.linspace(1.0 / n_checkpoints, 1.0, n_checkpoints)
    epochs = [max(1, int(f * total_epochs)) for f in fractions]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_epochs = []
    for e in epochs:
        if e not in seen:
            seen.add(e)
            unique_epochs.append(e)
    
    return unique_epochs


def train_with_checkpoints(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    n_checkpoints: int = 5,
    checkpoint_epochs: Optional[List[int]] = None,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[nn.Module, List[Checkpoint]]:
    """
    Train the model and save checkpoints at specified epochs.
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    x_train, y_train : torch.Tensor
        Training data
    epochs : int
        Number of training epochs
    optimizer : torch.optim.Optimizer
        Optimizer instance
    loss_fn : nn.Module
        Loss function
    device : torch.device
        Device for training
    n_checkpoints : int
        Number of checkpoints to save (ignored if checkpoint_epochs is provided)
    checkpoint_epochs : List[int], optional
        Specific epochs at which to save checkpoints
    batch_size : int
        Minibatch size for training
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    model : nn.Module
        Trained model
    checkpoints : List[Checkpoint]
        List of saved checkpoints
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Determine checkpoint epochs
    if checkpoint_epochs is None:
        checkpoint_epochs = get_checkpoint_epochs(epochs, n_checkpoints)
    checkpoint_epochs_set = set(checkpoint_epochs)
    
    logger.info(f"Training for {epochs} epochs, checkpoints at: {checkpoint_epochs}")
    
    # Create data loader
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    # Get learning rate from optimizer
    def get_lr():
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    checkpoints = []
    model.to(device)
    model.train()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        
        # Save checkpoint if needed
        if epoch in checkpoint_epochs_set:
            ckpt = Checkpoint(
                epoch=epoch,
                state_dict=copy.deepcopy(model.state_dict()),
                learning_rate=get_lr(),
                batch_size=batch_size,  # Store for minibatch SGD approximation
            )
            checkpoints.append(ckpt)
            logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f} [CHECKPOINT SAVED]")
        elif epoch % max(1, epochs // 10) == 0:
            logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
    
    return model, checkpoints


def train_simple(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    batch_size: int = 64,
    seed: int = 42,
    verbose: bool = False,
) -> nn.Module:
    """
    Train the model without checkpointing (for retraining during removal curve).
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    x_train, y_train : torch.Tensor
        Training data
    epochs : int
        Number of training epochs
    optimizer : torch.optim.Optimizer
        Optimizer instance
    loss_fn : nn.Module
        Loss function
    device : torch.device
        Device for training
    batch_size : int
        Minibatch size for training
    seed : int
        Random seed for reproducibility
    verbose : bool
        Whether to log training progress
        
    Returns
    -------
    model : nn.Module
        Trained model
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create data loader
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    model.to(device)
    model.train()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if verbose and epoch % max(1, epochs // 10) == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
    
    return model


def evaluate_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    y_eval: torch.Tensor = None,
) -> Tuple[float, float]:
    """
    Evaluate model on data.
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    x : torch.Tensor
        Input features
    y : torch.Tensor
        Adjusted phenotypes (for MSE computation)
    device : torch.device
        Device
    y_eval : torch.Tensor, optional
        Original phenotypes (for correlation). If None, uses y.
    
    Returns
    -------
    corr_eval : float
        Pearson correlation between predictions and y_eval (original phenotypes)
    mse_adj : float
        Mean squared error between predictions and y (adjusted phenotypes)
    """
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        x = x.to(device)
        preds = model(x).cpu().numpy()
    
    y_adj_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    y_eval_np = y_eval.cpu().numpy() if y_eval is not None else y_adj_np
    if isinstance(y_eval_np, torch.Tensor):
        y_eval_np = y_eval_np.numpy()
    
    # Pearson correlation with ORIGINAL phenotypes (y_eval)
    if len(preds) < 2:
        corr_eval = 0.0
    else:
        preds_centered = preds - preds.mean()
        y_centered = y_eval_np - y_eval_np.mean()
        denom = np.sqrt((preds_centered ** 2).sum()) * np.sqrt((y_centered ** 2).sum())
        if denom < 1e-12:
            corr_eval = 0.0
        else:
            corr_eval = float((preds_centered * y_centered).sum() / denom)
    
    # MSE with ADJUSTED phenotypes (y)
    mse_adj = float(((preds - y_adj_np) ** 2).mean())
    
    return corr_eval, mse_adj
