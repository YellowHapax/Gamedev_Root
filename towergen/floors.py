"""Discretize the continuous embedding into floors, then anneal on a discrete Hamiltonian."""

from __future__ import annotations

import math
import random

from .graph import ProgramGraph
from .hamiltonian import HamiltonianWeights

OVERFLOW_PENALTY = 50.0
EMPTY_FLOOR_PENALTY = 25.0


def size_tower(graph: ProgramGraph) -> int:
    """Smallest floor count whose tapered capacity holds the program (plus slack)."""
    spec = graph.tower
    need = sum(r.area for r in graph.rooms) * (1.0 + spec.slack)
    largest = max(r.area for r in graph.rooms)
    for f in range(max(1, spec.min_floors), spec.max_floors + 1):
        caps = [spec.capacity(k, f) for k in range(f)]
        if caps[-1] <= 0:
            break
        if sum(caps) >= need and f >= _anchor_floors_needed(graph):
            return f
    if spec.capacity(0, 1) < largest:
        raise ValueError(f"largest room ({largest} m²) does not fit on the ground floor")
    raise ValueError("program does not fit within max_floors; widen the tower or reduce taper")


def _anchor_floors_needed(graph: ProgramGraph) -> int:
    kinds = {r.anchor for r in graph.rooms}
    return 2 if {"ground", "top"} <= kinds else 1


def discrete_energy(
    graph: ProgramGraph, floors: list[int], num_floors: int, w: HamiltonianWeights
) -> float:
    spec = graph.tower
    top = max(num_floors - 1, 1)
    e = 0.0
    for edge in graph.edges:
        fa, fb = floors[graph.index[edge.a]], floors[graph.index[edge.b]]
        gap = abs(fa - fb)
        if edge.kind == "adjacent":
            e += w.adjacency * edge.weight * gap
        else:
            e += w.separation * edge.weight * max(0, 2 - gap)
    load = [0.0] * num_floors
    for i, room in enumerate(graph.rooms):
        f = floors[i]
        load[f] += room.area
        e += w.gravity * room.weight * f / top
        if room.height_pref is not None:
            e += w.preference * (f / top - room.height_pref) ** 2
    for f in range(num_floors):
        cap = spec.capacity(f, num_floors)
        if load[f] > cap:
            e += OVERFLOW_PENALTY * (load[f] - cap) / max(cap, 1.0)
        if load[f] == 0:
            e += EMPTY_FLOOR_PENALTY
    return e


def _fixed_floor(anchor: str | None, num_floors: int) -> int | None:
    return {"ground": 0, "top": num_floors - 1}.get(anchor)


def initial_floors(graph: ProgramGraph, embedding: list[list[float]], num_floors: int) -> list[int]:
    """Greedy fill in order of embedded height, honouring anchors and capacity."""
    spec = graph.tower
    remaining = [spec.capacity(f, num_floors) for f in range(num_floors)]
    floors = [0] * graph.n
    free = []
    for i, room in enumerate(graph.rooms):
        fixed = _fixed_floor(room.anchor, num_floors)
        if fixed is None:
            free.append(i)
        else:
            floors[i] = fixed
            remaining[fixed] -= room.area
    free.sort(key=lambda i: embedding[i][0])
    f = 0
    for i in free:
        area = graph.rooms[i].area
        # Target the floor implied by the embedded height, but never go back down
        # past floors already filled in this sweep.
        target = max(f, min(num_floors - 1, int(embedding[i][0] * num_floors)))
        placed = next((k for k in range(target, num_floors) if remaining[k] >= area), None)
        if placed is None:
            placed = max(range(num_floors), key=lambda k: remaining[k])
        floors[i] = placed
        remaining[placed] -= area
        f = placed if remaining[placed] > 0 else min(placed + 1, num_floors - 1)
    return floors


def anneal(
    graph: ProgramGraph,
    floors: list[int],
    num_floors: int,
    w: HamiltonianWeights,
    iterations: int | None = None,
    t_start: float = 2.0,
    t_end: float = 0.01,
    seed: int = 0,
) -> list[int]:
    """Metropolis annealing with relocate / swap moves. Anchored rooms never move."""
    rng = random.Random(seed)
    movable = [i for i, r in enumerate(graph.rooms) if r.anchor is None]
    if not movable or num_floors == 1:
        return list(floors)
    iterations = iterations or 400 * graph.n
    cur = list(floors)
    e_cur = discrete_energy(graph, cur, num_floors, w)
    best, e_best = list(cur), e_cur
    for step in range(iterations):
        t = t_start * (t_end / t_start) ** (step / max(iterations - 1, 1))
        cand = list(cur)
        i = rng.choice(movable)
        if rng.random() < 0.5 and len(movable) > 1:
            j = rng.choice(movable)
            cand[i], cand[j] = cand[j], cand[i]
        else:
            cand[i] = rng.randrange(num_floors)
        e_cand = discrete_energy(graph, cand, num_floors, w)
        if e_cand <= e_cur or rng.random() < math.exp((e_cur - e_cand) / t):
            cur, e_cur = cand, e_cand
            if e_cur < e_best:
                best, e_best = list(cur), e_cur
    return best
