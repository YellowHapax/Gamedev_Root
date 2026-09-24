"""End-to-end pipeline: program graph -> tower blueprint."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .circulation import plan_circulation
from .floors import anneal, discrete_energy, initial_floors, size_tower
from .graph import ProgramGraph
from .hamiltonian import HamiltonianWeights, bearing, convolve


@dataclass
class PlacedRoom:
    id: str
    kind: str
    area: float
    floor: int
    route_index: int  # position along the global Hamiltonian path
    angle_start: float  # degrees, counter-clockwise from +x
    angle_end: float
    inner_radius: float
    outer_radius: float


@dataclass
class Floor:
    index: int
    elevation: float
    radius: float
    capacity: float
    load: float
    landing_angle: float  # where the stair from below arrives
    exit_angle: float  # where the stair to the floor above departs
    rooms: list[str] = field(default_factory=list)


@dataclass
class Blueprint:
    name: str
    num_floors: int
    height: float
    core_radius: float
    floors: list[Floor]
    rooms: list[PlacedRoom]
    route: list[str]
    metrics: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def room(self, room_id: str) -> PlacedRoom:
        return next(r for r in self.rooms if r.id == room_id)


def _overlap(a: PlacedRoom, b: PlacedRoom) -> bool:
    """Do two wedges share any bearing (i.e. stack vertically)?"""

    def arcs(r: PlacedRoom) -> list[tuple[float, float]]:
        s, e = r.angle_start % 360.0, r.angle_start % 360.0 + (r.angle_end - r.angle_start)
        return [(s, min(e, 360.0)), (0.0, e - 360.0)] if e > 360.0 else [(s, e)]

    return any(s1 < e2 and s2 < e1 for s1, e1 in arcs(a) for s2, e2 in arcs(b))


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

    # 4. Lay rooms out as wedges around the core, in route order; the spiral
    #    stair climbs from each floor's exit to the next floor's landing.
    placed: dict[str, PlacedRoom] = {}
    out_floors: list[Floor] = []
    route_ids: list[str] = []
    angle = 0.0
    for f, route in enumerate(routes):
        cap = spec.capacity(f, num_floors)
        load = sum(graph.rooms[i].area for i in route)
        radius = spec.radius(f, num_floors)
        landing = angle
        for i in route:
            room = graph.rooms[i]
            span = 360.0 * room.area / max(cap, load)
            placed[room.id] = PlacedRoom(
                id=room.id,
                kind=room.kind,
                area=room.area,
                floor=f,
                route_index=len(route_ids),
                angle_start=round(angle, 3),
                angle_end=round(angle + span, 3),
                inner_radius=spec.core_radius,
                outer_radius=round(radius, 3),
            )
            route_ids.append(room.id)
            angle += span
        out_floors.append(
            Floor(
                index=f,
                elevation=f * spec.floor_height,
                radius=round(radius, 3),
                capacity=round(cap, 2),
                load=round(load, 2),
                landing_angle=round(landing % 360.0, 3),
                exit_angle=round(angle % 360.0, 3),
                rooms=[graph.rooms[i].id for i in route],
            )
        )
        angle += spec.stair_sweep

    rooms = [placed[r.id] for r in graph.rooms]
    metrics = _metrics(graph, placed, trace, e_initial, e_final, embedding)
    return Blueprint(
        name=graph.name,
        num_floors=num_floors,
        height=num_floors * spec.floor_height,
        core_radius=spec.core_radius,
        floors=out_floors,
        rooms=rooms,
        route=route_ids,
        metrics=metrics,
    )


def _metrics(graph, placed, trace, e_initial, e_final, embedding) -> dict:
    adjacent_ok, separate_ok, detail = 0, 0, []
    n_adj = n_sep = 0
    for e in graph.edges:
        a, b = placed[e.a], placed[e.b]
        gap = abs(a.floor - b.floor)
        stacked = gap == 1 and _overlap(a, b)
        neighbours = gap == 0 and abs(a.route_index - b.route_index) == 1
        if e.kind == "adjacent":
            n_adj += 1
            ok = neighbours or stacked
            adjacent_ok += ok
            relation = "neighbours" if neighbours else "stacked" if stacked else f"route distance {abs(a.route_index - b.route_index)}"
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
