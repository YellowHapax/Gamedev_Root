"""Grid floor plans: place rooms, spiral stairs and corridors on every floor.

Each floor is planned on a square grid (TowerSpec.cell_size metres per cell):

1. Planar Hamiltonian convolution. Rooms and stairs are point masses with
   rectangular footprints. Springs follow the floor's Hamiltonian route (so
   consecutive rooms end up near each other), same-floor adjacency edges, and
   anchors to related rooms on the floor below. Box-overlap repulsion and a
   pull toward the tower axis keep the footprint compact. The same damped
   symplectic dynamics as `hamiltonian.convolve` relaxes the system.
2. Legalization. Footprints are snapped to the grid, largest first, at the
   nearest free position that leaves at least one clear cell between rooms,
   which is where corridors run.
3. Corridors. A* with a turn penalty and a discount for reusing existing
   corridor cells links stair-down -> route rooms in order -> stair-up. The
   first and last cells of each path become doors.

The stair-up of floor f becomes the (fixed) stair-down of floor f+1, so the
spiral stairs line up vertically.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from dataclasses import dataclass, field

from .graph import ProgramGraph
from .hamiltonian import bearing

GAP = 1  # clear cells kept between footprints
STAIR_CELLS = 2  # spiral stair footprint (2x2 cells = 3 m at 1.5 m cells)

Cell = tuple[int, int]


@dataclass
class Rect:
    key: str
    x: int
    y: int
    w: int
    h: int
    shape: str = "rect"

    def cells(self) -> list[Cell]:
        return [(i, j) for i in range(self.x, self.x + self.w) for j in range(self.y, self.y + self.h)]

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def intersects(self, o: "Rect", margin: int = 0) -> bool:
        return not (
            self.x + self.w + margin <= o.x
            or o.x + o.w + margin <= self.x
            or self.y + self.h + margin <= o.y
            or o.y + o.h + margin <= self.y
        )

    def reach(self) -> float:
        """Distance from the tower axis to the farthest corner."""
        return max(math.hypot(cx, cy) for cx in (self.x, self.x + self.w) for cy in (self.y, self.y + self.h))

    def door_fronts(self) -> list[tuple[Cell, Cell]]:
        """(outside cell, inside cell) pairs where a door may go."""

        def span(n: int) -> list[int]:
            if self.shape == "round":  # the circle only touches the box mid-side
                return [n // 2] if n % 2 else [n // 2 - 1, n // 2]
            return list(range(1, n - 1)) if n >= 3 else list(range(n))

        out = []
        for k in span(self.w):
            i = self.x + k
            out.append(((i, self.y - 1), (i, self.y)))
            out.append(((i, self.y + self.h), (i, self.y + self.h - 1)))
        for k in span(self.h):
            j = self.y + k
            out.append(((self.x - 1, j), (self.x, j)))
            out.append(((self.x + self.w, j), (self.x + self.w - 1, j)))
        return out

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]


@dataclass
class FloorPlan:
    index: int
    radius: float  # in cells
    rooms: dict[str, Rect] = field(default_factory=dict)
    stair_down: Rect | None = None
    stair_up: Rect | None = None
    corridors: set[Cell] = field(default_factory=set)
    # (outside cell, inside cell, owner key)
    doors: list[tuple[Cell, Cell, str]] = field(default_factory=list)
    # Exterior doors: (outside cell, inside cell, owner key)
    exits: list[tuple[Cell, Cell, str]] = field(default_factory=list)

    def footprints(self) -> list[Rect]:
        return list(self.rooms.values()) + [s for s in (self.stair_down, self.stair_up) if s]


def footprint(room, cell_size: float) -> tuple[int, int]:
    cells = max(4, round(room.area / cell_size ** 2))
    if room.shape == "round":
        n = max(3, round(math.sqrt(cells * 4 / math.pi)))
        return n, n
    # Deterministic per-room aspect ratio in [1, 1.5).
    aspect = 1.0 + int(hashlib.md5(room.id.encode()).hexdigest()[:4], 16) % 50 / 100
    w = max(2, round(math.sqrt(cells * aspect)))
    return w, max(2, round(cells / w))


# --- 1. planar Hamiltonian convolution ------------------------------------


def _relax(nodes, springs, anchors, radius, steps=400, step=0.25, friction=0.8):
    """nodes: [x, y, w, h, fixed]; springs: (i, j, k); anchors: (i, (x, y), k)."""
    n = len(nodes)
    p = [[0.0, 0.0] for _ in range(n)]
    for _ in range(steps):
        g = [[0.0, 0.0] for _ in range(n)]
        for i, j, k in springs:
            for c in range(2):
                d = nodes[i][c] - nodes[j][c]
                g[i][c] += k * d
                g[j][c] -= k * d
        for i, (ax, ay), k in anchors:
            g[i][0] += k * (nodes[i][0] - ax)
            g[i][1] += k * (nodes[i][1] - ay)
        for i in range(n):
            x, y, w, h, _ = nodes[i]
            g[i][0] += 0.03 * x
            g[i][1] += 0.03 * y
            r = math.hypot(x, y)
            excess = r + 0.5 * math.hypot(w, h) - radius
            if excess > 0 and r > 1e-9:
                g[i][0] += excess * x / r
                g[i][1] += excess * y / r
            for j in range(i + 1, n):
                dx, dy = x - nodes[j][0], y - nodes[j][1]
                ox = (w + nodes[j][2]) / 2 + GAP - abs(dx)
                oy = (h + nodes[j][3]) / 2 + GAP - abs(dy)
                if ox > 0 and oy > 0:
                    # Penalty ½·min(ox, oy)² pushes along the shallower axis.
                    if ox < oy:
                        s = 1.0 if dx > 0 or (dx == 0 and i < j) else -1.0
                        g[i][0] -= ox * s
                        g[j][0] += ox * s
                    else:
                        s = 1.0 if dy > 0 or (dy == 0 and i < j) else -1.0
                        g[i][1] -= oy * s
                        g[j][1] += oy * s
        for i in range(n):
            if nodes[i][4]:
                continue
            for c in range(2):
                p[i][c] = friction * p[i][c] - step * g[i][c]
                nodes[i][c] += step * p[i][c]
    return nodes


# --- 2. legalization --------------------------------------------------------


def _legalize(key, want, size, shape, placed, limit) -> Rect:
    cx0, cy0 = want
    best, best_cost = None, math.inf
    span = int(limit) + max(size) + 6
    for w, h in {size, size[::-1]}:
        x_base, y_base = round(cx0 - w / 2), round(cy0 - h / 2)
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                r = Rect(key, x_base + dx, y_base + dy, w, h, shape)
                c = r.center
                cost = math.hypot(c[0] - cx0, c[1] - cy0) + 4.0 * max(0.0, r.reach() - limit)
                if cost >= best_cost:
                    continue
                if any(r.intersects(o, GAP) for o in placed):
                    continue
                best, best_cost = r, cost
    return best


# --- 3. corridors -----------------------------------------------------------

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _route(a: Rect, b: Rect, blocked, corridors, bounds, radius):
    starts = {o: i for o, i in a.door_fronts() if o not in blocked}
    goals = {o: i for o, i in b.door_fronts() if o not in blocked}
    x0, y0, x1, y1 = bounds

    def cost(c: Cell) -> float:
        base = 0.35 if c in corridors else 1.0
        if math.hypot(c[0] + 0.5, c[1] + 0.5) > radius:
            base += 0.8
        return base

    def h(c: Cell) -> float:
        return 0.35 * min(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goals)

    heap, seen, parent = [], {}, {}
    for s in starts:
        g0 = cost(s)
        heapq.heappush(heap, (g0 + h(s), g0, s, None))
        seen[(s, None)] = g0
        parent[(s, None)] = None
    while heap:
        _, gc, c, d = heapq.heappop(heap)
        if gc > seen.get((c, d), math.inf):
            continue
        if c in goals:
            path, state = [], (c, d)
            while state is not None:
                path.append(state[0])
                state = parent[state]
            path.reverse()
            return path, starts[path[0]], goals[c]
        for nd in _DIRS:
            nc = (c[0] + nd[0], c[1] + nd[1])
            if nc in blocked or not (x0 <= nc[0] <= x1 and y0 <= nc[1] <= y1):
                continue
            ng = gc + cost(nc) + (0.6 if d is not None and nd != d else 0.0)
            if ng < seen.get((nc, nd), math.inf):
                seen[(nc, nd)] = ng
                parent[(nc, nd)] = (c, d)
                heapq.heappush(heap, (ng + h(nc), ng, nc, nd))
    raise RuntimeError(f"no corridor from {a.key} to {b.key}")


# --- driver ------------------------------------------------------------------


def plan_floors(
    graph: ProgramGraph,
    floors: list[int],
    routes: list[list[int]],
    embedding: list[list[float]],
    num_floors: int,
) -> list[FloorPlan]:
    spec = graph.tower
    adj = graph.matrix("adjacent")
    plans: list[FloorPlan] = []
    below: FloorPlan | None = None
    for f in range(num_floors):
        radius = spec.radius(f, num_floors) / spec.cell_size
        next_radius = spec.radius(f + 1, num_floors) / spec.cell_size if f + 1 < num_floors else radius
        plan = FloorPlan(index=f, radius=radius)
        route = routes[f]

        # Nodes: [stair_down] + route rooms + [stair_up]
        keys, nodes, shapes = [], [], []
        if below is not None:
            s = below.stair_up
            plan.stair_down = Rect(f"stair_{f - 1}_{f}", s.x, s.y, s.w, s.h, "round")
            keys.append(plan.stair_down.key)
            nodes.append([*plan.stair_down.center, s.w, s.h, True])
            shapes.append("round")
        for i in route:
            room = graph.rooms[i]
            w, h = footprint(room, spec.cell_size)
            t = bearing(embedding[i])
            keys.append(room.id)
            nodes.append([0.5 * radius * math.cos(t), 0.5 * radius * math.sin(t), w, h, False])
            shapes.append(room.shape)
        if f + 1 < num_floors:
            keys.append(f"stair_{f}_{f + 1}")
            last = nodes[-1] if nodes else [0.0, 0.0]
            nodes.append([last[0] * 0.5, last[1] * 0.5, STAIR_CELLS, STAIR_CELLS, False])
            shapes.append("round")

        springs = [(k, k + 1, 1.0) for k in range(len(nodes) - 1)]
        pos = {key: k for k, key in enumerate(keys)}
        for a, i in enumerate(route):
            for b in route[a + 1 :]:
                if adj[i][b] > 0:
                    springs.append((pos[graph.rooms[i].id], pos[graph.rooms[b].id], 0.3 * adj[i][b]))
        anchors = []
        if below is not None:
            for i in route:
                for rid, rect in below.rooms.items():
                    w = adj[i][graph.index[rid]]
                    if w > 0:
                        anchors.append((pos[graph.rooms[i].id], rect.center, 0.2 * w))
        # Front doors want the tower's edge (south by convention).
        for i in route:
            if graph.rooms[i].kind == "entrance" and f == 0:
                anchors.append((pos[graph.rooms[i].id], (0.0, -radius), 1.0))
        _relax(nodes, springs, anchors, radius)

        # Legalize: fixed stair first, then stair-up (must fit the floor above), then rooms by size.
        placed: list[Rect] = []
        order = sorted(range(len(nodes)), key=lambda k: (not nodes[k][4], not keys[k].startswith("stair"), -nodes[k][2] * nodes[k][3]))
        for k in order:
            if nodes[k][4]:
                placed.append(plan.stair_down)
                continue
            limit = next_radius - 0.5 if keys[k].startswith("stair") else radius
            r = _legalize(keys[k], (nodes[k][0], nodes[k][1]), (int(nodes[k][2]), int(nodes[k][3])), shapes[k], placed, limit)
            placed.append(r)
            if keys[k].startswith("stair"):
                plan.stair_up = r
            else:
                plan.rooms[keys[k]] = r

        # Corridors along the route.
        blocked = {c for r in placed for c in r.cells()}
        xs = [c for r in placed for c in (r.x, r.x + r.w)]
        ys = [c for r in placed for c in (r.y, r.y + r.h)]
        bounds = (min(xs) - 3, min(ys) - 3, max(xs) + 3, max(ys) + 3)
        by_key = {r.key: r for r in placed}
        for a, b in zip(keys, keys[1:]):
            path, door_a, door_b = _route(by_key[a], by_key[b], blocked, plan.corridors, bounds, radius)
            plan.corridors.update(path)
            plan.doors.append((path[0], door_a, a))
            plan.doors.append((path[-1], door_b, b))
        plan.doors = list(dict.fromkeys(plan.doors))
        if f == 0:
            used = blocked | plan.corridors
            for i in route:
                room = graph.rooms[i]
                if room.kind != "entrance":
                    continue
                fronts = [(o, c) for o, c in plan.rooms[room.id].door_fronts() if o not in used]
                if fronts:
                    o, c = max(fronts, key=lambda oc: math.hypot(oc[0][0] + 0.5, oc[0][1] + 0.5))
                    plan.exits.append((o, c, room.id))
        plans.append(plan)
        below = plan
    return plans
