from dataclasses import dataclass
from typing import List
import torch.nn as nn

from .mlp import FeedForwardMLP


@dataclass
class TrainParams:
    lr: float
    weight_decay: float
    epochs: int
    loss_name: str
    optimizer: str
    hidden_dims: List[int]
    dropout: float
    batch_norm: bool


def make_model(in_dim: int, tp: TrainParams) -> nn.Module:
    return FeedForwardMLP(
        in_dim=in_dim,
        hidden_dims=tp.hidden_dims,
        dropout=tp.dropout,
        batch_norm=tp.batch_norm,
    )
