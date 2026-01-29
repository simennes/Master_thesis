"""Hyperparameter suggestion utilities for Optuna-based tuning."""
from __future__ import annotations
from typing import Any, Dict
import optuna

from src.models import TrainParams
from src.utils import encode_choices_for_optuna, decode_choice


def suggest_params(trial: optuna.Trial, space: Dict[str, Any]) -> TrainParams:
    """Suggest hyperparameters for a trial based on search space configuration.
    
    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial
    space : dict
        Search space configuration with keys: model, training, feature_selection
        
    Returns
    -------
    TrainParams
        Dataclass containing suggested hyperparameters
    """
    m = space.get("model", {})
    t = space.get("training", {})
    fsel = space.get("feature_selection", {})

    hidden = m.get("hidden_dims_choices", [])
    hidden = encode_choices_for_optuna(hidden)
    hidden = trial.suggest_categorical("hidden_dims", hidden)

    dropout = trial.suggest_float("dropout", *m.get("dropout_range", (0.0, 0.5)))
    batch_norm = trial.suggest_categorical("batch_norm", m.get("batch_norm_choices", [True, False]))

    lr = trial.suggest_float("lr", *t.get("lr_loguniform", (1e-4, 5e-3)), log=True)
    wd = trial.suggest_float("weight_decay", *t.get("wd_loguniform", (1e-7, 1e-3)), log=True)
    epochs = trial.suggest_int("epochs", *t.get("epochs_range", (50, 300)))
    loss = trial.suggest_categorical("loss", t.get("loss_choices", ["mse", "mae"]))
    opt = trial.suggest_categorical("optimizer", t.get("optimizer_choices", ["adam", "sgd", "adamw"]))

    # Feature selection knobs (logged in trial; used outside TrainParams)
    use_fs = trial.suggest_categorical(
        "use_snp_selection", fsel.get("use_snp_selection_choices", [False, True])
    )
    if use_fs:
        # number of SNPs to select if feature selection is on
        _ = trial.suggest_int("num_snps", *fsel.get("num_snps_range", (2000, 60000)))

    return TrainParams(lr=lr, weight_decay=wd, epochs=epochs, loss_name=loss, optimizer=opt,
                       hidden_dims=decode_choice(hidden), dropout=dropout, batch_norm=bool(batch_norm))
