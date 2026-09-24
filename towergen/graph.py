"""Program graph: rooms are nodes, spatial relationships are weighted edges."""

from __future__ import annotations

from dataclasses import dataclass, field

ANCHORS = (None, "ground", "top")
SHAPES = ("rect", "round")
EDGE_KINDS = ("adjacent", "separate")


@dataclass
class Room:
    id: str
    kind: str
    area: float
    # Structural load factor. Heavy rooms (forges, cisterns, vaults) sink.
    weight: float = 1.0
    # Preferred normalized height in [0, 1], or None for no preference.
    height_pref: float | None = None
    # Hard constraint: pin to the ground floor or the top floor.
    anchor: str | None = None
    # Footprint: "rect" rooms or "round" turret rooms.
    shape: str = "rect"

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"room {self.id!r}: shape must be one of {SHAPES}")
        if self.area <= 0:
            raise ValueError(f"room {self.id!r}: area must be positive")
        if self.anchor not in ANCHORS:
            raise ValueError(f"room {self.id!r}: anchor must be one of {ANCHORS}")
        if self.height_pref is not None and not 0.0 <= self.height_pref <= 1.0:
            raise ValueError(f"room {self.id!r}: height_pref must be in [0, 1]")


@dataclass
class Edge:
    a: str
    b: str
    weight: float = 1.0
    # "adjacent": rooms want to be close; "separate": rooms want distance.
    kind: str = "adjacent"

    def __post_init__(self) -> None:
        if self.kind not in EDGE_KINDS:
            raise ValueError(f"edge {self.a}-{self.b}: kind must be one of {EDGE_KINDS}")
        if self.weight <= 0:
            raise ValueError(f"edge {self.a}-{self.b}: weight must be positive")


@dataclass
class TowerSpec:
    base_radius: float = 9.0
    # Fractional radius reduction from the ground floor to the top floor.
    taper: float = 0.3
    floor_height: float = 4.0
    # Plan grid resolution in metres (1.5 m = the classic 5 ft square).
    cell_size: float = 1.5
    # Fraction of the floor disc usable as room area (the rest is walls,
    # corridors, stairs and rock).
    efficiency: float = 0.6
    # Extra capacity demanded over the total room area when sizing the tower.
    slack: float = 0.1
    min_floors: int = 1
    max_floors: int = 64

    def radius(self, floor: int, num_floors: int) -> float:
        t = floor / (num_floors - 1) if num_floors > 1 else 0.0
        return self.base_radius * (1.0 - self.taper * t)

    def capacity(self, floor: int, num_floors: int) -> float:
        import math

        r = self.radius(floor, num_floors)
        return self.efficiency * math.pi * r * r


@dataclass
class ProgramGraph:
    rooms: list[Room]
    edges: list[Edge] = field(default_factory=list)
    name: str = "Tower"
    tower: TowerSpec = field(default_factory=TowerSpec)

    def __post_init__(self) -> None:
        self.index = {r.id: i for i, r in enumerate(self.rooms)}
        if len(self.index) != len(self.rooms):
            raise ValueError("room ids must be unique")
        for e in self.edges:
            for end in (e.a, e.b):
                if end not in self.index:
                    raise ValueError(f"edge references unknown room {end!r}")
            if e.a == e.b:
                raise ValueError(f"self-loop on room {e.a!r}")

    @property
    def n(self) -> int:
        return len(self.rooms)

    def matrix(self, kind: str) -> list[list[float]]:
        """Symmetric weight matrix for one edge kind (duplicates add up)."""
        m = [[0.0] * self.n for _ in range(self.n)]
        for e in self.edges:
            if e.kind != kind:
                continue
            i, j = self.index[e.a], self.index[e.b]
            m[i][j] += e.weight
            m[j][i] += e.weight
        return m

    def laplacian(self) -> list[list[float]]:
        """Graph Laplacian L = D - W of the attractive (adjacent) edges."""
        w = self.matrix("adjacent")
        return [
            [(sum(w[i]) if i == j else -w[i][j]) for j in range(self.n)]
            for i in range(self.n)
        ]

    @classmethod
    def from_dict(cls, data: dict) -> "ProgramGraph":
        return cls(
            rooms=[Room(**r) for r in data["rooms"]],
            edges=[Edge(**e) for e in data.get("edges", [])],
            name=data.get("name", "Tower"),
            tower=TowerSpec(**data.get("tower", {})),
        )
