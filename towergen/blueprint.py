"""End-to-end pipeline: program graph -> tower blueprint.

All plan coordinates are grid cells (Blueprint.cell_size metres each), with the
tower axis at (0, 0), x east and y north. A rect is [x, y, w, h] covering cells
x..x+w-1, y..y+h-1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .circulation import plan_circulation
from .floors import anneal, discrete_energy, initial_floors, size_tower
from .graph import ProgramGraph
from .hamiltonian import HamiltonianWeights, bearing, convolve
from .plan import FloorPlan, Rect, plan_floors


@dataclass
class PlacedRoom:
    id: str
    kind: str
    area: float
    shape: str
    floor: int
    route_index: int  # position along the global Hamiltonian path
    rect: list[int]  # [x, y, w, h] in cells
    # Each door is [outside_x, outside_y, inside_x, inside_y]: it sits on the
    # wall edge between those two cells.
    doors: list[list[int]] = field(default_factory=list)


@dataclass
class Floor:
    index: int
    elevation: float
    radius: float  # metres
    capacity: float
    load: float
    rooms: list[str]
    stair_down: list[int] | None  # [x, y, w, h] of the spiral stair from below
    stair_up: list[int] | None
    corridors: list[list[int]]  # corridor cells [x, y]
    doors: list[list[int]]  # stair doors, same format as PlacedRoom.doors
    exits: list[list[int]]  # exterior doors [outside_x, outside_y, inside_x, inside_y]


@dataclass
class Blueprint:
    name: str
    num_floors: int
    height: float
    cell_size: float
    floors: list[Floor]
    rooms: list[PlacedRoom]
    route: list[str]
    metrics: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def room(self, room_id: str) -> PlacedRoom:
        return next(r for r in self.rooms if r.id == room_id)


def generate(
    graph: ProgramGraph,
    weights: HamiltonianWeights | None = None,
    layers: int = 300,
    seed: int = 0,
) -> Blueprint:
    w = weights or HamiltonianWeights()
    spec = graph.tower

    # 1. Hamiltonian convolution: continuous (height, bearing) embedding.
    embedding, trace = convolve(graph, w, layers=layers, seed=seed)

    # 2. Discretize to floors and refine on the discrete Hamiltonian.
    num_floors = size_tower(graph)
    floors = initial_floors(graph, embedding, num_floors)
    e_initial = discrete_energy(graph, floors, num_floors, w)
    floors = anneal(graph, floors, num_floors, w, seed=seed)
    e_final = discrete_energy(graph, floors, num_floors, w)

    # 3. Circulation: Hamiltonian path through every room.
    routes = plan_circulation(graph, floors, embedding, num_floors)

    # 4. Grid floor plans: rooms, aligned spiral stairs, corridors, doors.
    plans = plan_floors(graph, floors, routes, embedding, num_floors)

    route_ids = [graph.rooms[i].id for route in routes for i in route]
    placed: dict[str, PlacedRoom] = {}
    out_floors: list[Floor] = []
    for plan, route in zip(plans, routes):
        doors: dict[str, list[list[int]]] = {}
        for (ox, oy), (ix, iy), key in plan.doors:
            doors.setdefault(key, []).append([ox, oy, ix, iy])
        for i in route:
            room = graph.rooms[i]
            placed[room.id] = PlacedRoom(
                id=room.id,
                kind=room.kind,
                area=room.area,
                shape=room.shape,
                floor=plan.index,
                route_index=route_ids.index(room.id),
                rect=plan.rooms[room.id].to_list(),
                doors=doors.get(room.id, []),
            )
        stair_doors = [d for k, ds in doors.items() if k.startswith("stair") for d in ds]
        out_floors.append(
            Floor(
                index=plan.index,
                elevation=plan.index * spec.floor_height,
                radius=round(spec.radius(plan.index, num_floors), 3),
                capacity=round(spec.capacity(plan.index, num_floors), 2),
                load=round(sum(graph.rooms[i].area for i in route), 2),
                rooms=[graph.rooms[i].id for i in route],
                stair_down=plan.stair_down.to_list() if plan.stair_down else None,
                stair_up=plan.stair_up.to_list() if plan.stair_up else None,
                corridors=sorted([list(c) for c in plan.corridors]),
                doors=stair_doors,
                exits=[[o[0], o[1], i[0], i[1]] for o, i, _ in plan.exits],
            )
        )

    rooms = [placed[r.id] for r in graph.rooms]
    metrics = _metrics(graph, placed, trace, e_initial, e_final, embedding)
    return Blueprint(
        name=graph.name,
        num_floors=num_floors,
        height=num_floors * spec.floor_height,
        cell_size=spec.cell_size,
        floors=out_floors,
        rooms=rooms,
        route=route_ids,
        metrics=metrics,
    )


def _rect(r: PlacedRoom) -> Rect:
    return Rect(r.id, *r.rect)


def _gap(a: Rect, b: Rect) -> int:
    """Clear cells between two footprints (Chebyshev)."""
    gx = max(a.x - (b.x + b.w), b.x - (a.x + a.w), 0)
    gy = max(a.y - (b.y + b.h), b.y - (a.y + a.h), 0)
    return max(gx, gy)


def _metrics(graph, placed, trace, e_initial, e_final, embedding) -> dict:
    adjacent_ok = separate_ok = n_adj = n_sep = 0
    detail = []
    for e in graph.edges:
        a, b = placed[e.a], placed[e.b]
        ra, rb = _rect(a), _rect(b)
        gap = abs(a.floor - b.floor)
        stacked = gap == 1 and ra.intersects(rb)
        near = gap == 0 and _gap(ra, rb) <= 2
        if e.kind == "adjacent":
            n_adj += 1
            ok = near or stacked
            adjacent_ok += ok
            relation = "same floor, near" if near else "stacked" if stacked else (
                f"same floor, {_gap(ra, rb)} cells apart" if gap == 0 else f"{gap} floor(s) apart"
            )
        else:
            n_sep += 1
            ok = gap > 0 and not stacked
            separate_ok += ok
            relation = f"{gap} floor(s) apart" + (" but stacked" if stacked else "")
        detail.append({"a": e.a, "b": e.b, "kind": e.kind, "satisfied": bool(ok), "relation": relation})
    return {
        "continuous_energy": {"initial": round(trace[0], 4), "final": round(trace[-1], 4)},
        "discrete_energy": {"initial": round(e_initial, 4), "annealed": round(e_final, 4)},
        "adjacency_satisfied": f"{adjacent_ok}/{n_adj}",
        "separation_satisfied": f"{separate_ok}/{n_sep}",
        "edges": detail,
        "embedding": {
            r.id: {"z": round(embedding[i][0], 4), "bearing_deg": round(bearing(embedding[i]) * 57.29578, 2)}
            for i, r in enumerate(graph.rooms)
        },
    }
