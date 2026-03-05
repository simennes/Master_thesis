"""
Genetic Algorithm for fixed-size subset selection.

Designed for training-set optimization where we want to select ``n_train``
individuals from a candidate pool to minimise an objective (e.g. PEVmean).

Implementation follows the approach described in:
  Akdemir, Sánchez & Jannink 2015 – "Optimization of genomic selection
  training populations with a genetic algorithm"

Key design choices
------------------
* Representation: each chromosome is a *sorted* 1-D int array of indices
  into the candidate pool.
* Crossover: *set-based* – the child is a random sample of size ``n_train``
  drawn from the union of two parents (preserves subset size exactly).
* Mutation: *swap* – remove ``n_swaps`` elements and replace with the same
  number drawn uniformly from the complement of the current subset.
* Elitism: the top ``n_elite`` solutions survive unchanged into the next
  generation.
* Caching: a dict mapping ``tuple(sorted_subset) -> fitness`` avoids
  redundant objective evaluations.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GAConfig:
    """Configuration for the genetic algorithm."""

    pop_size: int = 50
    """Number of chromosomes in the population."""

    n_generations: int = 100
    """Number of GA generations to run."""

    n_elite: int = 2
    """Number of top individuals carried over unchanged (elitism)."""

    tournament_k: int = 3
    """Tournament size for parent selection."""

    crossover_prob: float = 0.9
    """Probability of performing crossover (else clone parent)."""

    mutation_prob: float = 0.3
    """Probability of mutating an offspring."""

    n_swaps_per_mut: int = 2
    """Number of swaps per mutation event."""

    seed: int = 42
    """Master random seed."""

    verbose: bool = True
    """Log progress every generation."""

    stagnation_limit: int = 0
    """Stop early after this many generations with no improvement (0 = off)."""


# ------------------------------------------------------------------
# Core GA
# ------------------------------------------------------------------

def run_ga(
    n_candidates: int,
    n_train: int,
    fitness_fn: Callable[[np.ndarray], float],
    cfg: GAConfig,
    candidate_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Run a genetic algorithm to find a subset of size ``n_train`` that
    *minimises* ``fitness_fn(subset)``.

    Parameters
    ----------
    n_candidates : int
        Total size of the candidate pool.
    n_train : int
        Desired training-set size (fixed across all chromosomes).
    fitness_fn : callable
        ``fitness_fn(indices_1d) -> float`` – the objective to *minimise*.
        ``indices_1d`` is a sorted 1-D int array of length ``n_train``.
    cfg : GAConfig
        Algorithm hyper-parameters.
    candidate_indices : optional 1-D int array
        If given, the actual candidate indices (e.g. global row-IDs).
        Chromosomes will contain values from this array rather than 0..n-1.

    Returns
    -------
    best_subset : 1-D int array of length ``n_train``.
    best_fitness : float (lowest fitness found).
    stats : dict with convergence info.
    """
    rng = np.random.default_rng(cfg.seed)
    t0 = time.perf_counter()

    if candidate_indices is not None:
        pool = np.sort(np.asarray(candidate_indices, dtype=np.int64))
        n_candidates = len(pool)
    else:
        pool = np.arange(n_candidates, dtype=np.int64)

    if n_train >= n_candidates:
        logger.warning(
            "n_train (%d) >= n_candidates (%d); returning full pool",
            n_train,
            n_candidates,
        )
        score = fitness_fn(pool[:n_train])
        return pool[:n_train].copy(), score, {"generations_run": 0}

    # ---- Fitness cache -------------------------------------------------------
    cache: Dict[tuple, float] = {}

    def _evaluate(chrom: np.ndarray) -> float:
        key = tuple(chrom)
        if key not in cache:
            cache[key] = fitness_fn(chrom)
        return cache[key]

    # ---- Initialise population -----------------------------------------------
    pop: List[np.ndarray] = []
    for _ in range(cfg.pop_size):
        chrom = np.sort(rng.choice(pool, size=n_train, replace=False))
        pop.append(chrom)

    fitness = np.array([_evaluate(c) for c in pop])

    best_idx = int(np.argmin(fitness))
    best_fitness = float(fitness[best_idx])
    best_chrom = pop[best_idx].copy()

    history: List[Dict] = []
    stagnation_counter = 0

    for gen in range(cfg.n_generations):
        # ---- Selection + reproduction ----------------------------------------
        new_pop: List[np.ndarray] = []

        # Elitism
        elite_order = np.argsort(fitness)
        for ei in range(min(cfg.n_elite, cfg.pop_size)):
            new_pop.append(pop[elite_order[ei]].copy())

        # Fill rest via tournament selection + crossover + mutation
        while len(new_pop) < cfg.pop_size:
            p1 = _tournament_select(pop, fitness, cfg.tournament_k, rng)
            p2 = _tournament_select(pop, fitness, cfg.tournament_k, rng)

            if rng.random() < cfg.crossover_prob:
                child = _set_crossover(p1, p2, n_train, rng)
            else:
                child = p1.copy()

            if rng.random() < cfg.mutation_prob:
                child = _swap_mutation(child, pool, cfg.n_swaps_per_mut, rng)

            new_pop.append(child)

        pop = new_pop[: cfg.pop_size]
        fitness = np.array([_evaluate(c) for c in pop])

        gen_best_idx = int(np.argmin(fitness))
        gen_best = float(fitness[gen_best_idx])

        improved = gen_best < best_fitness
        if improved:
            best_fitness = gen_best
            best_chrom = pop[gen_best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        gen_info = {
            "generation": gen,
            "best_fitness": best_fitness,
            "gen_best": gen_best,
            "gen_mean": float(np.mean(fitness)),
            "gen_std": float(np.std(fitness)),
            "cache_size": len(cache),
            "improved": improved,
        }
        history.append(gen_info)

        if cfg.verbose and (gen % max(1, cfg.n_generations // 20) == 0 or gen == cfg.n_generations - 1):
            logger.info(
                "GA gen %3d/%d | best=%.6f  gen_best=%.6f  mean=%.6f  cache=%d",
                gen,
                cfg.n_generations,
                best_fitness,
                gen_best,
                gen_info["gen_mean"],
                len(cache),
            )

        if cfg.stagnation_limit > 0 and stagnation_counter >= cfg.stagnation_limit:
            logger.info(
                "GA early stop: no improvement for %d generations (gen %d)",
                cfg.stagnation_limit,
                gen,
            )
            break

    elapsed = time.perf_counter() - t0
    stats = {
        "generations_run": len(history),
        "best_fitness": best_fitness,
        "cache_size": len(cache),
        "elapsed_sec": elapsed,
        "history": history,
    }
    logger.info(
        "GA finished: best PEVmean=%.6f  evals=%d  time=%.1fs",
        best_fitness,
        len(cache),
        elapsed,
    )
    return best_chrom, best_fitness, stats


# ------------------------------------------------------------------
# GA operators
# ------------------------------------------------------------------

def _tournament_select(
    pop: List[np.ndarray],
    fitness: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select the best individual from a random tournament of size k."""
    idx = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
    winner = idx[int(np.argmin(fitness[idx]))]
    return pop[winner]


def _set_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    n_train: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Set-based crossover: draw ``n_train`` elements from union(parent1, parent2).

    Elements in the intersection are kept with higher probability
    (they appear in both parents, so they bias the sample naturally
    since they appear twice in the concatenated array – slight exploitation bias).
    """
    union = np.union1d(parent1, parent2)
    if len(union) <= n_train:
        return np.sort(union[:n_train])

    # Bias toward intersection: duplicate intersection members in the draw pool
    intersection = np.intersect1d(parent1, parent2)
    pool = np.concatenate([union, intersection])
    # Unique draw via shuffled pick (avoids duplicates)
    chosen_set = set()
    order = rng.permutation(len(pool))
    for i in order:
        chosen_set.add(int(pool[i]))
        if len(chosen_set) == n_train:
            break
    # If still not enough (shouldn't happen), fill randomly from union
    if len(chosen_set) < n_train:
        remaining = set(union.tolist()) - chosen_set
        extra = rng.choice(list(remaining), size=n_train - len(chosen_set), replace=False)
        chosen_set.update(extra.tolist())

    return np.sort(np.array(list(chosen_set), dtype=np.int64))


def _swap_mutation(
    chrom: np.ndarray,
    pool: np.ndarray,
    n_swaps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Remove ``n_swaps`` elements and add the same number from the complement."""
    chrom_set = set(chrom.tolist())
    complement = np.array([x for x in pool if x not in chrom_set], dtype=np.int64)
    if len(complement) == 0:
        return chrom.copy()

    actual_swaps = min(n_swaps, len(chrom), len(complement))
    remove_pos = rng.choice(len(chrom), size=actual_swaps, replace=False)
    add_vals = rng.choice(complement, size=actual_swaps, replace=False)

    new_chrom = chrom.copy()
    new_chrom[remove_pos] = add_vals
    return np.sort(new_chrom)
