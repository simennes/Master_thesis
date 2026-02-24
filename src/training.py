"""
Training utilities with checkpoint saving for TracIn.

Provides training functions that save model checkpoints at specified epochs
for use in TracIn influence computation.
"""
from __future__ import annotations
import copy
import logging
import os
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
    x_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    x_test: Optional[torch.Tensor] = None,
    y_test: Optional[torch.Tensor] = None,
    loss_plot_path: Optional[str] = None,
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
    x_val, y_val : torch.Tensor, optional
        Validation data for tracking validation loss over epochs.
    x_test, y_test : torch.Tensor, optional
        Test data for tracking test loss over epochs.
    loss_plot_path : str, optional
        If provided, saves a loss curve plot (train/val) to this path.
        
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
    train_losses: List[float] = []
    val_losses: List[float] = []
    test_losses: List[float] = []
    has_val = x_val is not None and y_val is not None
    has_test = x_test is not None and y_test is not None
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
        train_losses.append(avg_loss)

        if has_val or has_test:
            model.eval()
            with torch.no_grad():
                if has_val:
                    val_preds = model(x_val.to(device))
                    val_loss = loss_fn(val_preds, y_val.to(device)).item()
                    val_losses.append(val_loss)
                if has_test:
                    test_preds = model(x_test.to(device))
                    test_loss = loss_fn(test_preds, y_test.to(device)).item()
                    test_losses.append(test_loss)
            model.train()
        
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
    
    if loss_plot_path:
        _plot_loss_curves(train_losses, val_losses, test_losses, loss_plot_path)

    return model, checkpoints


def _plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    test_losses: List[float],
    loss_plot_path: str,
) -> None:
    """Save training/validation/test loss curves to a PNG file."""
    if not train_losses:
        return

    os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses, label="train")
    if val_losses:
        plt.plot(epochs, val_losses, label="val")
    if test_losses:
        plt.plot(epochs, test_losses, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close()


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
    x_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    x_test: Optional[torch.Tensor] = None,
    y_test: Optional[torch.Tensor] = None,
    patience: int = 0,
    loss_plot_path: Optional[str] = None,
    sample_weights: Optional[np.ndarray] = None,
    restore_best: bool = False,
) -> Tuple[nn.Module, int]:
    """
    Train the model without checkpointing (for retraining during removal curve).
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    x_train, y_train : torch.Tensor
        Training data
    epochs : int
        Number of training epochs (max epochs if early stopping is used)
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
    x_val, y_val : torch.Tensor, optional
        Validation data for early stopping. If provided with patience > 0,
        training stops when validation loss hasn't improved for `patience` epochs.
    patience : int
        Number of epochs without validation improvement before stopping.
        0 means no early stopping (train for all epochs).
    loss_plot_path : str, optional
        If provided, saves a loss curve plot (train/val/test) to this path.
    sample_weights : np.ndarray, optional
        Per-sample weights for weighted sampling (shape: n_samples).
        If provided, uses WeightedRandomSampler instead of uniform shuffling.
    restore_best : bool
        If True and validation data is provided, always track the epoch with
        the lowest validation loss and restore that model at the end, even
        when early stopping is disabled (patience=0). This trains for all
        epochs but returns the best-validation-loss snapshot.
        
    Returns
    -------
    model : nn.Module
        Trained model (best validation state if early stopping or restore_best)
    epochs_trained : int
        Number of epochs actually completed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    has_val = x_val is not None and y_val is not None
    has_test = x_test is not None and y_test is not None
    use_early_stopping = patience > 0 and has_val
    track_best = (use_early_stopping or restore_best) and has_val
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    train_losses: List[float] = []
    val_losses: List[float] = []
    test_losses: List[float] = []
    
    # Create data loader with optional weighted sampling
    dataset = TensorDataset(x_train, y_train)
    if sample_weights is not None:
        from torch.utils.data import WeightedRandomSampler
        sample_weights_t = torch.from_numpy(sample_weights).float()
        sampler = WeightedRandomSampler(
            weights=sample_weights_t,
            num_samples=len(x_train),
            replacement=True,
        )
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    model.to(device)
    
    epochs_trained = 0
    for epoch in range(1, epochs + 1):
        model.train()
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
        train_losses.append(avg_loss)

        if verbose and epoch % max(1, epochs // 10) == 0:
            logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")

        val_loss = None
        if has_val or has_test:
            model.eval()
            with torch.no_grad():
                if has_val:
                    val_preds = model(x_val.to(device))
                    val_loss = loss_fn(val_preds, y_val.to(device)).item()
                    val_losses.append(val_loss)
                if has_test:
                    test_preds = model(x_test.to(device))
                    test_loss = loss_fn(test_preds, y_test.to(device)).item()
                    test_losses.append(test_loss)
            model.train()
        
        # Track best validation model
        if track_best and val_loss is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            # Early stopping check (only if patience > 0)
            if use_early_stopping and epochs_no_improve >= patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                epochs_trained = epoch
                break
        
        epochs_trained = epoch
    
    # Restore best model if early stopping or restore_best was used
    if track_best and best_state is not None:
        model.load_state_dict(best_state)
        if restore_best and not use_early_stopping:
            logger.info(f"Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.6f})")
    
    if loss_plot_path:
        _plot_loss_curves(train_losses, val_losses, test_losses, loss_plot_path)

    return model, epochs_trained


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
