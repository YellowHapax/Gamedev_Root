"""Circulation as a Hamiltonian path: one walk that visits every room exactly once.

Climbing is expensive and descending is worse, so an optimal route is monotone
in floor. The global path therefore decomposes into one Hamiltonian path per
floor, chained by stair transitions: each floor is solved with its entry costs
conditioned on where the previous floor's route ended.

Per-floor paths are solved exactly with Held–Karp (O(n²·2ⁿ)) up to
EXACT_LIMIT rooms, and with nearest-neighbour + 2-opt beyond that.
"""

from __future__ import annotations

import math

from .graph import ProgramGraph
from .hamiltonian import bearing

EXACT_LIMIT = 12
ADJACENCY_BONUS = 1.0


def _angle_gap(a: float, b: float) -> float:
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d) / math.pi  # normalized to [0, 1]


def held_karp(cost: list[list[float]], entry: list[float]) -> tuple[float, list[int]]:
    """Min-cost open Hamiltonian path. entry[j] is the cost of starting at j."""
    n = len(entry)
    if n == 0:
        return 0.0, []
    full = (1 << n) - 1
    inf = math.inf
    dp = [[inf] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    for j in range(n):
        dp[1 << j][j] = entry[j]
    for mask in range(1, 1 << n):
        row = dp[mask]
        for j in range(n):
            if row[j] == inf:
                continue
            for k in range(n):
                if mask & (1 << k):
                    continue
                nm = mask | (1 << k)
                v = row[j] + cost[j][k]
                if v < dp[nm][k]:
                    dp[nm][k] = v
                    parent[nm][k] = j
    end = min(range(n), key=lambda j: dp[full][j])
    best = dp[full][end]
    path, mask, j = [], full, end
    while j != -1:
        path.append(j)
        pj = parent[mask][j]
        mask ^= 1 << j
        j = pj
    return best, path[::-1]


def path_cost(cost: list[list[float]], entry: list[float], path: list[int]) -> float:
    if not path:
        return 0.0
    return entry[path[0]] + sum(cost[a][b] for a, b in zip(path, path[1:]))


def heuristic_path(cost: list[list[float]], entry: list[float]) -> tuple[float, list[int]]:
    """Nearest neighbour from every start, then 2-opt; keep the best."""
    n = len(entry)
    best_path, best_cost = None, math.inf
    for start in range(n):
        path, left = [start], set(range(n)) - {start}
        while left:
            nxt = min(left, key=lambda k: cost[path[-1]][k])
            path.append(nxt)
            left.remove(nxt)
        improved = True
        while improved:
            improved = False
            for a in range(n - 1):
                for b in range(a + 1, n):
                    cand = path[:a] + path[a : b + 1][::-1] + path[b + 1 :]
                    if path_cost(cost, entry, cand) < path_cost(cost, entry, path) - 1e-12:
                        path, improved = cand, True
        c = path_cost(cost, entry, path)
        if c < best_cost:
            best_path, best_cost = path, c
    return best_cost, best_path


def solve_path(cost: list[list[float]], entry: list[float]) -> tuple[float, list[int]]:
    if len(entry) <= EXACT_LIMIT:
        return held_karp(cost, entry)
    return heuristic_path(cost, entry)


def plan_circulation(
    graph: ProgramGraph, floors: list[int], embedding: list[list[float]], num_floors: int
) -> list[list[int]]:
    """Return per-floor room orderings; concatenated they form the Hamiltonian path."""
    adj = graph.matrix("adjacent")
    sep = graph.matrix("separate")
    theta = [bearing(x) for x in embedding]

    def link(i: int, j: int) -> float:
        return _angle_gap(theta[i], theta[j]) - ADJACENCY_BONUS * adj[i][j] + sep[i][j]

    routes: list[list[int]] = []
    prev_exit: int | None = None
    prev_floor_rooms: list[int] = []
    for f in range(num_floors):
        rooms = [i for i in range(graph.n) if floors[i] == f]
        if not rooms:
            routes.append([])
            continue
        cost = [[link(a, b) for b in rooms] for a in rooms]
        entry = []
        for r in rooms:
            if prev_exit is None:
                # Ground floor: start at the front door, else any ground-anchored room.
                room = graph.rooms[r]
                entry.append(0.0 if room.kind == "entrance" else 1.0 if room.anchor == "ground" else 2.0)
            else:
                # Stair landing: be near the previous exit, and stack over
                # rooms on the floor below that this room is adjacent to.
                e = _angle_gap(theta[prev_exit], theta[r]) - ADJACENCY_BONUS * adj[prev_exit][r]
                e -= 0.25 * sum(adj[r][q] for q in prev_floor_rooms)
                entry.append(e)
        _, order = solve_path(cost, entry)
        route = [rooms[k] for k in order]
        routes.append(route)
        prev_exit = route[-1]
        prev_floor_rooms = rooms
    return routes
