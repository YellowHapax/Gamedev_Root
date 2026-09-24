"""Hamiltonian graph convolution: relax a room embedding under an energy operator.

Each room i carries a state x_i = (z, u, v): a normalized height z in [0, 1]
and a unit planar direction (u, v) giving its bearing around the tower.

The potential V(x) is

    V = ½ Σ_ij W_ij |x_i - x_j|²                 adjacency springs  (= xᵀ L x)
      + Σ_ij S_ij  exp(-|x_i - x_j|² / σ²)        "separate" repulsion
      + γ Σ_i<j    exp(-|x_i - x_j|² / σ²)        crowding (spreads the program)
      + α Σ_i weight_i · z_i                      gravity (heavy rooms sink)
      + β Σ_i (z_i - pref_i)²                     height preference
      + κ Σ_anchored (z_i - target_i)²            ground / top anchors

and the Hamiltonian is H(x, p) = ½|p|² + V(x).

One *convolution layer* is one damped symplectic-Euler step of Hamilton's
equations:

    p ← μ·p − η·∇V(x)
    x ← x + η·p

∇V is computed by message passing: node i aggregates W_ij(x_i − x_j) from its
graph neighbours plus the repulsive kernels. For the spring term alone the
update is x ← x − η²·L x + ..., i.e. the graph convolution x ← (I − η²L) x,
a first-order expansion of the propagator exp(−τL). Stacking layers diffuses
information across the program graph while momentum lets the embedding roll
out of shallow local minima. Friction μ < 1 makes it converge.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .graph import ProgramGraph


@dataclass
class HamiltonianWeights:
    adjacency: float = 1.0
    separation: float = 2.0
    crowding: float = 0.3
    gravity: float = 0.5
    preference: float = 3.0
    anchor: float = 25.0
    sigma: float = 0.45


def _anchor_target(anchor: str | None) -> float | None:
    return {"ground": 0.0, "top": 1.0}.get(anchor)


def potential(graph: ProgramGraph, x: list[list[float]], w: HamiltonianWeights) -> float:
    adj = graph.matrix("adjacent")
    sep = graph.matrix("separate")
    s2 = w.sigma ** 2
    e = 0.0
    n = graph.n
    for i in range(n):
        for j in range(i + 1, n):
            d2 = sum((a - b) ** 2 for a, b in zip(x[i], x[j]))
            e += w.adjacency * adj[i][j] * d2
            k = math.exp(-d2 / s2)
            e += (w.separation * sep[i][j] + w.crowding) * k
    for i, room in enumerate(graph.rooms):
        z = x[i][0]
        e += w.gravity * room.weight * z
        if room.height_pref is not None:
            e += w.preference * (z - room.height_pref) ** 2
        t = _anchor_target(room.anchor)
        if t is not None:
            e += w.anchor * (z - t) ** 2
    return e


def gradient(graph: ProgramGraph, x: list[list[float]], w: HamiltonianWeights) -> list[list[float]]:
    """∇V via message passing (pairwise messages + unary node fields)."""
    adj = graph.matrix("adjacent")
    sep = graph.matrix("separate")
    s2 = w.sigma ** 2
    n = graph.n
    g = [[0.0, 0.0, 0.0] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            diff = [a - b for a, b in zip(x[i], x[j])]
            d2 = sum(c * c for c in diff)
            # Spring: d/dx_i of W|x_i-x_j|² = 2W(x_i-x_j)
            coef = 2.0 * w.adjacency * adj[i][j]
            # Gaussian repulsion: d/dx_i of A·exp(-d²/σ²) = -2A/σ² · k · (x_i-x_j)
            coef -= 2.0 * (w.separation * sep[i][j] + w.crowding) * math.exp(-d2 / s2) / s2
            for c in range(3):
                g[i][c] += coef * diff[c]
                g[j][c] -= coef * diff[c]
    for i, room in enumerate(graph.rooms):
        z = x[i][0]
        g[i][0] += w.gravity * room.weight
        if room.height_pref is not None:
            g[i][0] += 2.0 * w.preference * (z - room.height_pref)
        t = _anchor_target(room.anchor)
        if t is not None:
            g[i][0] += 2.0 * w.anchor * (z - t)
    return g


def _project(xi: list[float], prev: list[float]) -> list[float]:
    """Constrain to the tower manifold: z ∈ [0,1], (u,v) on the unit circle."""
    z = min(1.0, max(0.0, xi[0]))
    norm = math.hypot(xi[1], xi[2])
    if norm < 1e-9:
        return [z, prev[1], prev[2]]
    return [z, xi[1] / norm, xi[2] / norm]


def initial_state(graph: ProgramGraph, rng: random.Random) -> list[list[float]]:
    x = []
    for i, room in enumerate(graph.rooms):
        z = _anchor_target(room.anchor)
        if z is None:
            z = room.height_pref if room.height_pref is not None else 0.5
        z = min(1.0, max(0.0, z + rng.uniform(-0.05, 0.05)))
        theta = 2.0 * math.pi * i / graph.n + rng.uniform(-0.2, 0.2)
        x.append([z, math.cos(theta), math.sin(theta)])
    return x


def convolve(
    graph: ProgramGraph,
    weights: HamiltonianWeights | None = None,
    layers: int = 300,
    step: float = 0.08,
    friction: float = 0.85,
    seed: int = 0,
) -> tuple[list[list[float]], list[float]]:
    """Run `layers` Hamiltonian convolution layers. Returns (embedding, energy trace)."""
    w = weights or HamiltonianWeights()
    rng = random.Random(seed)
    x = initial_state(graph, rng)
    p = [[0.0, 0.0, 0.0] for _ in range(graph.n)]
    trace = [potential(graph, x, w)]
    for _ in range(layers):
        g = gradient(graph, x, w)
        for i in range(graph.n):
            for c in range(3):
                p[i][c] = friction * p[i][c] - step * g[i][c]
            moved = [x[i][c] + step * p[i][c] for c in range(3)]
            x[i] = _project(moved, x[i])
        trace.append(potential(graph, x, w))
    return x, trace


def bearing(xi: list[float]) -> float:
    """Planar bearing of an embedded room, in radians [0, 2π)."""
    return math.atan2(xi[2], xi[1]) % (2.0 * math.pi)
