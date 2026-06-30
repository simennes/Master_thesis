"""Target-island split helper used by the E5 Shapley driver.

This module previously hosted the full TracIn removal-curve experiment; that
experiment is no longer part of the thesis and was removed during the
end-of-project cleanup. Only :func:`split_target_island` survives because
``src.tmc_shapley_islands`` imports it.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def split_target_island(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    locality: np.ndarray,
    target_island_code: int,
    y_eval: Optional[np.ndarray] = None,
    cal_fraction: float = 0.2,
    seed: int = 42,
    max_cal_fraction: Optional[float] = None,
    cal_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
    included_island_codes: Optional[List[int]] = None,
    n_cal_fixed: Optional[int] = None,
):
    """Split data into source (non-target islands) and target (cal + test).

    Parameters
    ----------
    X, y, ids, locality : np.ndarray
        Full dataset (y is the adjusted phenotype used for training).
    target_island_code : int
        Code of the target island to hold out.
    y_eval : np.ndarray, optional
        Original phenotypes used for evaluation (e.g. y_mean).
    cal_fraction : float
        Fraction of target-island individuals to put into the calibration set.
        Ignored when ``n_cal_fixed`` is given or when ``cal_idx``/``test_idx``
        are supplied.
    n_cal_fixed : int, optional
        Use a fixed calibration-set size (E5 uses this).
    cal_idx, test_idx : np.ndarray, optional
        Pre-computed permutations into the target island array. When provided
        they override the random split.
    included_island_codes : list[int], optional
        Restrict the source pool to a subset of islands.

    Returns
    -------
    dict
        Keys: X_source/y_source/ids_source/locality_source/y_eval_source,
        X_cal/y_cal/ids_cal/locality_cal/y_eval_cal,
        X_test/y_test/ids_test/locality_test/y_eval_test.
    """
    np.random.seed(seed)

    if y_eval is None:
        y_eval = y.copy()

    target_mask = (locality == target_island_code)
    source_mask = ~target_mask

    if included_island_codes is not None:
        included_set = set(int(v) for v in included_island_codes)
        source_mask = source_mask & np.isin(locality, list(included_set))

    X_source = X[source_mask]
    y_source = y[source_mask]
    ids_source = ids[source_mask]
    locality_source = locality[source_mask]
    y_eval_source = y_eval[source_mask]

    X_target = X[target_mask]
    y_target = y[target_mask]
    ids_target = ids[target_mask]
    locality_target = locality[target_mask]
    y_eval_target = y_eval[target_mask]

    n_target = len(X_target)
    if n_cal_fixed is not None:
        n_cal = max(1, min(int(n_cal_fixed), n_target - 1))
    else:
        n_cal = max(1, int(cal_fraction * n_target))

    if cal_idx is None or test_idx is None:
        perm = np.random.permutation(n_target)
        if max_cal_fraction is not None:
            n_cal_max = max(1, int(max_cal_fraction * n_target))
            if n_cal > n_cal_max:
                raise ValueError(
                    f"cal_fraction={cal_fraction} exceeds max_cal_fraction={max_cal_fraction}"
                )
            cal_idx = perm[:n_cal]
            test_idx = perm[n_cal_max:]
        else:
            cal_idx = perm[:n_cal]
            test_idx = perm[n_cal:]
    else:
        cal_idx = np.asarray(cal_idx)
        test_idx = np.asarray(test_idx)

    return {
        "X_source": X_source, "y_source": y_source, "ids_source": ids_source,
        "locality_source": locality_source, "y_eval_source": y_eval_source,
        "X_cal": X_target[cal_idx], "y_cal": y_target[cal_idx],
        "ids_cal": ids_target[cal_idx], "locality_cal": locality_target[cal_idx],
        "y_eval_cal": y_eval_target[cal_idx],
        "X_test": X_target[test_idx], "y_test": y_target[test_idx],
        "ids_test": ids_target[test_idx], "locality_test": locality_target[test_idx],
        "y_eval_test": y_eval_target[test_idx],
    }
