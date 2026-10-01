"""
woa.py -- the Whale Optimization Algorithm.

WOA is inspired by how humpback whales hunt: they either circle in
tight on prey they can see, spiral inward while blowing a "bubble net"
around it, or -- if prey isn't nearby -- swim off toward some other
point of interest to search from there instead. Every whale in our
population picks one of these three moves, every iteration, based on
a single shrinking parameter that gradually shifts the whole
population from "explore broadly" (early iterations) to "home in on
the best area found so far" (later iterations).

WOA never looks at a job, a machine, or a schedule. It only ever moves
points around in a plain numeric space -- here, a chromosome as
defined in encoding.py, a flat vector of 2N numbers in [0, 1]. To
score how "good" a whale's position is, we decode it into a real
schedule (via encoding.decode_chromosome) and read its makespan.
That's the ONLY place this file touches the FJSP problem at all --
everything else below is generic numeric search.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Optional, Tuple


import numpy as np


from .encoding import get_operation_list, decode_chromosome

#The fitness function is defined as the makespan of the decoded chromosome. The lower the makespan, the better the fitness.
def _fitness(chromosome: np.ndarray, instance: Dict[str, Any]) -> int:
    return decode_chromosome(chromosome, instance)["makespan"]


def _clip_to_unit_interval(position: np.ndarray) -> np.ndarray:
    """
    Clip back into [0, 1] after every move, since WOA's update
    equations can overshoot.
    """
    return np.clip(position, 0.0, 1.0)

# Adaptive inertia weight functions

def _cosine_decay_weight(iteration: int, max_iterations: int, w_min: float, w_max: float) -> float:
    """
    w(t): cosine decay from w_max down to w_min, used to scale
    EXPLOITATION moves (encircle prey, spiral bubble-net attack).
    """
    t_ratio = iteration / max_iterations
    return w_min + (w_max - w_min) * (1 + math.cos(math.pi * t_ratio)) / 2.0


def _exponential_decay_weight(
    iteration: int, max_iterations: int, w_min: float, w_max: float, decay_rate: float = 2.0
) -> float:
    """
    v(t): exponential decay from w_max down to w_min, used to scale
    EXPLORATION moves (searching toward a random other whale). Decays
    faster than w(t) by design.
    """
    t_ratio = iteration / max_iterations
    safe_w_min = max(w_min, 1e-6)
    return w_max * (safe_w_min / w_max) ** (t_ratio ** decay_rate)

#for every operation picks fastest eligible machine.
def _fastest_machine_chromosome(
    instance: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a seed using the fastest eligible machine per operation."""
    return _machine_assignment_chromosome(instance, rng, balanced=False)

#Tries to balance load accross machine ie pick sthe one with which has the LEAST total load. 
def _balanced_machine_chromosome(
    instance: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a seed that balances estimated load across eligible machines."""
    return _machine_assignment_chromosome(instance, rng, balanced=True)

#Builds as array of float values in [0, 1] representing a machine assignment for each operation. If balanced is True, it tries to balance load across machines; if False, it picks the fastest eligible machine for each operation.
def _machine_assignment_chromosome(
    instance: Dict[str, Any],
    rng: np.random.Generator,
    balanced: bool,
) -> np.ndarray:
    operation_list = get_operation_list(instance)
    machine_load = {machine: 0 for machine in instance["machines"]}
    machine_genes = []
    for operation in operation_list:
        #machine load=>how much work is already piled on this machine
        #find min using two conditions, first total load. If it comes same then use processing time for that operation
        eligible = instance["E"][operation]
        if balanced:
            selected_machine = min(
                eligible,
                key=lambda machine: (
                    machine_load[machine]
                    + instance["P"][(operation[0], operation[1], machine)],
                    instance["P"][(operation[0], operation[1], machine)],
                ),
            )
        else:
            #pick fastest machine for this operation
            selected_machine = min(
                eligible,
                key=lambda machine: instance["P"][(operation[0], operation[1], machine)],
            )
        machine_load[selected_machine] += instance["P"][
            (operation[0], operation[1], selected_machine)
        ]
        machine_index = eligible.index(selected_machine)
        machine_genes.append((machine_index + 0.5) / len(eligible))

    return np.asarray(machine_genes + list(rng.random(len(operation_list))))
#So this functions builds a chromosome array with half values with machine assignments and half values (selected at random) which are used for sequencing machine operations.



def run_woa(
    instance: Dict[str, Any],
    population_size: int = 30,
    max_iterations: int = 100,
    seed: int = 0,
    spiral_shape_constant: float = 1.0,
    local_search_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], Tuple[np.ndarray, int]]] = None,
    local_search_frequency: Optional[int] = None,
    max_time_seconds: Optional[float] = None,
    use_adaptive_weight: bool = True,
    w_min: float = 0.3,
    w_max: float = 0.9,
    # optional external seed (e.g. CP-SAT's feasible schedule encoded as a chromosome,
    # If given, replaces ONE population
    # slot (not the whole population) -- this gives the seed a chance
    # to anchor the search without destroying population diversity, so
    # a bad or misleading seed can still be escaped by other whales.
    seed_chromosome: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, list]:
    """
    Runs WOA on the given FJSP instance and returns the best chromosome
    found, its makespan, and the history of the best makespan seen
    after every iteration.

    Parameters
    ----------
    instance :
        An FJSP instance dict, as produced by generator.generate_instance().
    population_size :
        How many whales search in parallel.
    max_iterations :
        How many rounds of movement the population goes through.
    seed :
        RNG seed for reproducibility.
    spiral_shape_constant :
        Controls how tightly the spiral (bubble-net) move curves in
        toward the best whale.
    local_search_fn :
        Optional hook for hybridizing with a local search step. In
        this architecture, this is LNS (see solve.py).
        Takes (chromosome, instance), returns an improved
        (chromosome, makespan) pair.
    local_search_frequency :
        If local_search_fn is given, how often (in iterations) to call
        it on the current best whale.
    max_time_seconds :
        Optional total wall-clock limit for WOA and its local-search hook.
    use_adaptive_weight, w_min, w_max :
        Adaptive inertia weight parameters.
    seed_chromosome :
        Optional externally-provided chromosome (e.g. encoded from a
        CP-SAT feasible schedule) to seed the population with.
        If None, population init is unchanged
        (falls back to the existing heuristic + random seeding).

    Returns
    -------
    (best_chromosome, best_makespan, history)
    """
    deadline = None if max_time_seconds is None else time.perf_counter() + max_time_seconds

    operation_list = get_operation_list(instance)
    n_operations = len(operation_list)
    dimension = 2 * n_operations

    rng = np.random.default_rng(seed)

    #seed the population
    if seed_chromosome is not None:
        # Use the external seed (e.g. CP-SAT's incumbent) in place of
        # the "fastest machine" heuristic whale for one slot only.
        first_whale = np.asarray(seed_chromosome, dtype=float)
    else:
        first_whale = _fastest_machine_chromosome(instance, rng)

    population = [first_whale]
    fitness_values = [_fitness(first_whale, instance)]
    if population_size > 1 and (deadline is None or time.perf_counter() < deadline):
        balanced_whale = _balanced_machine_chromosome(instance, rng)
        population.append(balanced_whale)
        fitness_values.append(_fitness(balanced_whale, instance))
    for _ in range(population_size - len(population)):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        whale = rng.random(dimension)
        population.append(whale)
        fitness_values.append(_fitness(whale, instance))

    best_index = int(np.argmin(fitness_values))
    best_position = population[best_index].copy()
    best_fitness = fitness_values[best_index]

    history = [best_fitness]

    for iteration in range(1, max_iterations + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            break

        current_population_size = len(population)
        a = 2.0 - iteration * (2.0 / max_iterations)

        if use_adaptive_weight:
            w_t = _cosine_decay_weight(iteration, max_iterations, w_min, w_max)
            v_t = _exponential_decay_weight(iteration, max_iterations, w_min, w_max)
        else:
            w_t = 1.0
            v_t = 1.0

        for i in range(current_population_size):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            r1 = rng.random()
            r2 = rng.random()

            A = 2.0 * a * r1 - a
            C = 2.0 * r2

            move_choice = rng.random()

            if move_choice < 0.5:
                if abs(A) < 1.0:
                    # Encircle prey (exploitation)scaled by w(t)
                    distance_to_best = np.abs(C * best_position - population[i])
                    new_position = w_t * best_position - A * distance_to_best
                else:
                    # --- Search for prey (exploration) -- scaled by v(t) ---
                    random_index = rng.integers(0, current_population_size)
                    random_whale = population[random_index]
                    distance_to_random = np.abs(C * random_whale - population[i])
                    new_position = v_t * random_whale - A * distance_to_random
            else:
                # Spiral bubble-net attack (exploitation) [scaled by w(t)]
                spiral_parameter = rng.uniform(-1.0, 1.0)
                distance_to_best = np.abs(best_position - population[i])
                new_position = (
                    distance_to_best
                    * np.exp(spiral_shape_constant * spiral_parameter)
                    * np.cos(2.0 * np.pi * spiral_parameter)
                    + w_t * best_position
                )

            population[i] = _clip_to_unit_interval(new_position)
            fitness_values[i] = _fitness(population[i], instance)

        worst_index = int(np.argmax(fitness_values))
        if best_fitness < fitness_values[worst_index]:
            population[worst_index] = best_position.copy()
            fitness_values[worst_index] = best_fitness

        current_best_index = int(np.argmin(fitness_values))
        if fitness_values[current_best_index] < best_fitness:
            best_fitness = fitness_values[current_best_index]
            best_position = population[current_best_index].copy()

        #  Local search polish step (LNS attaches here)
        if local_search_fn is not None and local_search_frequency:
            if iteration % local_search_frequency == 0:
                if deadline is None:
                    polished_position, polished_fitness = local_search_fn(
                        best_position, instance
                    )
                else:
                    polished_position, polished_fitness = local_search_fn(
                        best_position, instance, deadline=deadline
                    )
                if polished_fitness < best_fitness:
                    best_fitness = polished_fitness
                    best_position = np.asarray(polished_position)
                    worst_index = int(np.argmax(fitness_values))
                    population[worst_index] = best_position.copy()
                    fitness_values[worst_index] = best_fitness

        history.append(best_fitness)

    return best_position, best_fitness, history