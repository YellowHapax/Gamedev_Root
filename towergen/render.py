"""SVG floor plans and a plain-text summary of a Blueprint."""

from __future__ import annotations

import math
import random
from xml.sax.saxutils import escape

from .blueprint import Blueprint, Floor

PX = 24  # pixels per grid cell
COLS = 3
PAD = 4  # cells of sky around each floor (room for landing stages)

SKY = "#cfe3ee"
ROCK = "#a98664"
ROCK_EDGE = "#4a3826"
FLOOR = "#e4e0d7"
GRID = "#cbc4b6"
WALL_DARK = "#3d2f22"
WALL = "#c9a67a"
WOOD = "#8a5a33"
INK = "#2b2118"

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

DEFS = f"""<defs>
<pattern id="tiles" width="{PX}" height="{PX}" patternUnits="userSpaceOnUse">
  <rect width="{PX}" height="{PX}" fill="{FLOOR}"/>
  <path d="M{PX} 0V{PX}H0" fill="none" stroke="{GRID}" stroke-width="0.7"/>
</pattern>
<filter id="rock" x="-15%" y="-15%" width="130%" height="130%">
  <feTurbulence type="fractalNoise" baseFrequency="0.03" numOctaves="2" seed="7" result="n"/>
  <feDisplacementMap in="SourceGraphic" in2="n" scale="16" xChannelSelector="R" yChannelSelector="G" result="d"/>
  <feMorphology in="d" operator="dilate" radius="2" result="thick"/>
  <feFlood flood-color="{ROCK_EDGE}"/>
  <feComposite in2="thick" operator="in" result="outline"/>
  <feMerge><feMergeNode in="outline"/><feMergeNode in="d"/></feMerge>
</filter>
<filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
  <feDropShadow dx="3" dy="5" stdDeviation="4" flood-color="#1d3440" flood-opacity="0.35"/>
</filter>
</defs>"""


# --- per-floor geometry -----------------------------------------------------


class _FloorGeom:
    """Cell ownership for one floor: which room/stair/corridor owns each cell."""

    def __init__(self, bp: Blueprint, floor: Floor):
        self.floor = floor
        self.rooms = [bp.room(r) for r in floor.rooms]
        self.owner: dict[tuple[int, int], str] = {}
        self.round: set[str] = set()
        self.boxes: dict[str, list[int]] = {}
        for r in self.rooms:
            self._own(r.id, r.rect, r.shape == "round")
        self.stairs = []
        if floor.stair_down:
            self.stairs.append(("down", floor.stair_down))
            self._own("stair_down", floor.stair_down, True)
        if floor.stair_up:
            self.stairs.append(("up", floor.stair_up))
            self._own("stair_up", floor.stair_up, True)
        for x, y in floor.corridors:
            self.owner[(x, y)] = "corridor"
        self.exits = [tuple(d) for d in floor.exits]
        self.doors = [tuple(d) for r in self.rooms for d in r.doors] + [tuple(d) for d in floor.doors] + self.exits
        self.door_edges = {frozenset({(d[0], d[1]), (d[2], d[3])}) for d in self.doors}
        self.mass = self._mass()

    def _own(self, key, rect, is_round):
        x, y, w, h = rect
        self.boxes[key] = rect
        if is_round:
            self.round.add(key)
        for i in range(x, x + w):
            for j in range(y, y + h):
                self.owner[(i, j)] = key

    def _mass(self) -> set[tuple[int, int]]:
        """Interior dilated by one cell, with enclosed pockets filled."""
        mass = {(i + dx, j + dy) for i, j in self.owner for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        xs, ys = [c[0] for c in mass], [c[1] for c in mass]
        x0, x1, y0, y1 = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
        outside, stack = set(), [(x0, y0)]
        while stack:
            c = stack.pop()
            if c in outside or c in mass or not (x0 <= c[0] <= x1 and y0 <= c[1] <= y1):
                continue
            outside.add(c)
            stack.extend((c[0] + dx, c[1] + dy) for dx, dy in _DIRS)
        return {(i, j) for i in range(x0, x1 + 1) for j in range(y0, y1 + 1) if (i, j) not in outside}


# --- drawing helpers ----------------------------------------------------------


class _Pen:
    """Maps grid coordinates (y up) into a panel's pixel space (y down)."""

    def __init__(self, x0: int, y1: int):
        self.x0, self.y1 = x0, y1

    def p(self, X: float, Y: float) -> tuple[float, float]:
        return (X - self.x0) * PX, (self.y1 - Y) * PX

    def box(self, rect) -> tuple[float, float, float, float]:
        x, y, w, h = rect
        sx, sy = self.p(x, y + h)
        return sx, sy, w * PX, h * PX


def _wall_path(segments) -> str:
    return " ".join(f"M{a:.1f} {b:.1f}L{c:.1f} {d:.1f}" for a, b, c, d in segments)


def _furnish(kind: str, x: float, y: float, w: float, h: float, rng: random.Random) -> list[str]:
    """Simple top-down furniture glyphs inside a room box (pixel space)."""
    s = PX
    out = []
    cx, cy = x + w / 2, y + h / 2

    def rect(rx, ry, rw, rh, fill, stroke=WALL_DARK, sw=0.8):
        out.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" rx="1.5" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def bed(bx, by):
        rect(bx, by, 0.9 * s, 1.5 * s, "#e9e2cf")
        rect(bx + 0.1 * s, by + 0.1 * s, 0.7 * s, 0.35 * s, "#fbf8f0")
        rect(bx, by + 0.55 * s, 0.9 * s, 0.95 * s, "#6f9bb8")

    def table(tw, th, chairs=True):
        rect(cx - tw / 2, cy - th / 2, tw, th, "#b9875a")
        if chairs:
            n = max(1, int(tw / (0.8 * s)))
            for k in range(n):
                px = cx - tw / 2 + (k + 0.5) * tw / n - 0.17 * s
                rect(px, cy - th / 2 - 0.42 * s, 0.34 * s, 0.34 * s, "#8a5a33")
                rect(px, cy + th / 2 + 0.08 * s, 0.34 * s, 0.34 * s, "#8a5a33")

    def crate(bx, by):
        rect(bx, by, 0.7 * s, 0.7 * s, "#b58b5a")
        out.append(f'<path d="M{bx:.1f} {by:.1f}l{0.7 * s:.1f} {0.7 * s:.1f}M{bx + 0.7 * s:.1f} {by:.1f}l{-0.7 * s:.1f} {0.7 * s:.1f}" stroke="{WALL_DARK}" stroke-width="0.7"/>')

    if kind == "private":
        bed(x + 0.25 * s, y + 0.25 * s)
        out.append(f'<ellipse cx="{cx + 0.2 * s:.1f}" cy="{cy + 0.2 * s:.1f}" rx="{w * 0.22:.1f}" ry="{h * 0.18:.1f}" fill="#b5584a" opacity="0.55"/>')
        rect(x + w - 1.0 * s, y + 0.25 * s, 0.7 * s, 0.45 * s, "#7b4f2c")
    elif kind == "military":
        n = max(1, int((w - 0.4 * s) / (1.1 * s)))
        for k in range(n):
            bed(x + 0.25 * s + k * 1.1 * s, y + 0.2 * s)
        if h > 2.4 * s:
            for k in range(int((w - 0.4 * s) / (0.5 * s))):
                rect(x + 0.25 * s + k * 0.5 * s, y + h - 0.45 * s, 0.35 * s, 0.25 * s, "#8c9aa6")
    elif kind == "social":
        table(min(w - 1.4 * s, 4.5 * s), 0.8 * s)
    elif kind == "service":
        rect(x + 0.2 * s, y + 0.2 * s, w - 0.4 * s, 0.6 * s, "#a88a64")
        for k in range(min(3, int(w / (0.9 * s)))):
            crate(x + 0.25 * s + k * 0.85 * s, y + h - 0.95 * s)
        out.append(f'<circle cx="{cx:.1f}" cy="{cy + 0.1 * s:.1f}" r="{0.45 * s:.1f}" fill="#b9875a" stroke="{WALL_DARK}" stroke-width="0.8"/>')
    elif kind == "workshop":
        rect(x + 0.2 * s, y + 0.2 * s, w - 0.4 * s, 0.6 * s, "#9a7652")
        out.append(f'<circle cx="{cx:.1f}" cy="{cy + 0.3 * s:.1f}" r="{0.55 * s:.1f}" fill="#d9772f" stroke="{WALL_DARK}" stroke-width="1"/>')
        out.append(f'<circle cx="{cx:.1f}" cy="{cy + 0.3 * s:.1f}" r="{0.25 * s:.1f}" fill="#f2c14e"/>')
        for k in range(3):
            out.append(f'<circle cx="{x + w - 0.5 * s:.1f}" cy="{y + h - 0.5 * s - k * 0.45 * s:.1f}" r="{0.17 * s:.1f}" fill="#6fae9a" stroke="{WALL_DARK}" stroke-width="0.6"/>')
    elif kind == "study":
        rect(x + 0.15 * s, y + 0.15 * s, w - 0.3 * s, 0.3 * s, "#6e4a2b")
        rect(x + 0.15 * s, y + 0.5 * s, 0.3 * s, h - 0.65 * s, "#6e4a2b")
        for k in range(int((w - 0.3 * s) / 4)):
            color = rng.choice(["#b04a3a", "#3f6f8f", "#d3a441", "#4f7d4a"])
            out.append(f'<rect x="{x + 0.2 * s + k * 4:.1f}" y="{y + 0.18 * s:.1f}" width="2.5" height="{0.24 * s:.1f}" fill="{color}"/>')
        rect(cx - 0.6 * s, cy - 0.3 * s, 1.2 * s, 0.6 * s, "#b9875a")
        out.append(f'<circle cx="{cx:.1f}" cy="{cy + 0.6 * s:.1f}" r="{0.2 * s:.1f}" fill="#8a5a33"/>')
    elif kind == "vault":
        for k in range(min(3, int((w - 0.3 * s) / (0.85 * s)))):
            bx = x + 0.25 * s + k * 0.85 * s
            rect(bx, y + 0.25 * s, 0.7 * s, 0.5 * s, "#8a5a33")
            out.append(f'<rect x="{bx:.1f}" y="{y + 0.44 * s:.1f}" width="{0.7 * s:.1f}" height="2" fill="#e0b64a"/>')
        for _ in range(5):
            out.append(f'<circle cx="{cx + rng.uniform(-0.5, 0.5) * s:.1f}" cy="{cy + rng.uniform(0, 0.6) * s:.1f}" r="2.2" fill="#e0b64a" stroke="#9c7a22" stroke-width="0.5"/>')
    elif kind == "arcane":
        r = 0.36 * min(w, h)
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="#5a7fb0" stroke-width="1.6"/>')
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r * 0.78:.1f}" fill="none" stroke="#5a7fb0" stroke-width="0.8"/>')
        pts = " ".join(f"{cx + r * 0.78 * math.cos(math.pi / 2 + k * 4 * math.pi / 5):.1f},{cy - r * 0.78 * math.sin(math.pi / 2 + k * 4 * math.pi / 5):.1f}" for k in range(5))
        out.append(f'<polygon points="{pts}" fill="none" stroke="#5a7fb0" stroke-width="0.8"/>')
    elif kind == "utility":
        rect(cx - 0.35 * w, cy - 0.3 * h, 0.7 * w, 0.6 * h, "#7fb3d1", sw=1.5)
        rect(cx - 0.25 * w, cy - 0.2 * h, 0.5 * w, 0.4 * h, "#a9d2e6", stroke="none")
    elif kind == "entrance":
        m = 0.3 * min(w, h)
        rect(cx - m, cy - m, 2 * m, 2 * m, "#a9bfd3")
        rect(cx - m * 0.55, cy - m * 0.55, m * 1.1, m * 1.1, "#d9c28a")
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{m * 0.3:.1f}" fill="#3d4f66"/>')
    else:
        table(min(w - 1.4 * s, 2.0 * s), 0.8 * s)
    return out


def _stair(cx: float, cy: float, r: float, direction: str) -> list[str]:
    out = [f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{FLOOR}"/>']
    steps = "".join(
        f"M{cx + 0.2 * r * math.cos(a):.1f} {cy + 0.2 * r * math.sin(a):.1f}L{cx + r * math.cos(a):.1f} {cy + r * math.sin(a):.1f}"
        for a in (k * 2 * math.pi / 12 for k in range(12))
    )
    out.append(f'<path d="{steps}" stroke="#8d7b64" stroke-width="1"/>')
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{0.2 * r:.1f}" fill="#8d7b64"/>')
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="{WALL_DARK}" stroke-width="7"/>')
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="{WALL}" stroke-width="3.5"/>')
    arrow = "▲ up" if direction == "up" else "▼ down"
    out.append(f'<text x="{cx:.1f}" y="{cy + r + 12:.1f}" font-size="9" text-anchor="middle" fill="{INK}" '
               f'stroke="{FLOOR}" stroke-width="3" paint-order="stroke">{arrow}</text>')
    return out


def _draw_floor(bp: Blueprint, g: _FloorGeom, pen: _Pen, rng: random.Random) -> list[str]:
    out = []
    # Rock mass.
    out.append('<g filter="url(#rock)">')
    for i, j in sorted(g.mass):
        x, y = pen.p(i + 0.5, j + 0.5)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{0.78 * PX:.1f}" fill="{ROCK}"/>')
    out.append("</g>")

    # Wooden landing stage outside each exterior door.
    for ox, oy, ix, iy in g.exits:
        dx, dy = ox - ix, oy - iy
        length = 3.5
        ax, ay = pen.p(ox + 0.5 - 0.6 * abs(dy) + 0.0 * dx, oy + 0.5 - 0.6 * abs(dx))
        bx, by = pen.p(ox + 0.5 + 0.6 * abs(dy) + dx * length, oy + 0.5 + 0.6 * abs(dx) + dy * length)
        x0, y0, w, h = min(ax, bx), min(ay, by), abs(bx - ax), abs(by - ay)
        out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#c9a06a" stroke="{WALL_DARK}" stroke-width="1.2"/>')
        planks = int((w if dy == 0 else h) / 5)
        for k in range(1, planks):
            if dy == 0:
                out.append(f'<path d="M{x0 + k * 5:.1f} {y0:.1f}v{h:.1f}" stroke="#8a6a44" stroke-width="0.7"/>')
            else:
                out.append(f'<path d="M{x0:.1f} {y0 + k * 5:.1f}h{w:.1f}" stroke="#8a6a44" stroke-width="0.7"/>')

    # Floors: tiled cells for rect rooms and corridors, discs for round rooms.
    tiles = [c for c, o in g.owner.items() if o not in g.round]
    for i, j in tiles:
        x, y = pen.p(i, j + 1)
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{PX + 0.5}" height="{PX + 0.5}" fill="url(#tiles)"/>')
    for r in g.rooms:
        if r.shape == "round":
            bx, by, bw, bh = pen.box(r.rect)
            out.append(f'<circle cx="{bx + bw / 2:.1f}" cy="{by + bh / 2:.1f}" r="{bw / 2 - 1:.1f}" fill="url(#tiles)"/>')

    # Straight walls on every cell edge between different owners (except doors).
    segs = []
    for (i, j), o in g.owner.items():
        if o in g.round:
            continue
        for dx, dy in _DIRS:
            n = (i + dx, j + dy)
            o2 = g.owner.get(n)
            if o2 == o or frozenset({(i, j), n}) in g.door_edges:
                continue
            if dx == 1:
                segs.append((*pen.p(i + 1, j), *pen.p(i + 1, j + 1)))
            elif dx == -1:
                segs.append((*pen.p(i, j), *pen.p(i, j + 1)))
            elif dy == 1:
                segs.append((*pen.p(i, j + 1), *pen.p(i + 1, j + 1)))
            else:
                segs.append((*pen.p(i, j), *pen.p(i + 1, j)))
    path = _wall_path(segs)
    out.append(f'<path d="{path}" stroke="{WALL_DARK}" stroke-width="7" stroke-linecap="square"/>')
    out.append(f'<path d="{path}" stroke="{WALL}" stroke-width="3.5" stroke-linecap="square"/>')
    for r in g.rooms:
        if r.shape == "round":
            bx, by, bw, bh = pen.box(r.rect)
            for width, color in ((7, WALL_DARK), (3.5, WALL)):
                out.append(f'<circle cx="{bx + bw / 2:.1f}" cy="{by + bh / 2:.1f}" r="{bw / 2 - 1:.1f}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    # Furniture.
    for r in g.rooms:
        bx, by, bw, bh = pen.box(r.rect)
        if r.shape == "round":  # furnish the square inscribed in the circle
            k = bw * (1 - 1 / math.sqrt(2)) / 2
            bx, by, bw, bh = bx + k, by + k, bw - 2 * k, bh - 2 * k
        out.extend(_furnish(r.kind, bx, by, bw, bh, rng))

    # Spiral stairs.
    for direction, rect in g.stairs:
        bx, by, bw, bh = pen.box(rect)
        out.extend(_stair(bx + bw / 2, by + bh / 2, bw / 2 - 2, direction))

    # Doors: wooden leaves across the wall edge between the two cells.
    for ox, oy, ix, iy in g.doors:
        mx, my = pen.p((ox + ix) / 2 + 0.5, (oy + iy) / 2 + 0.5)
        horizontal = oy != iy  # the edge runs along x
        w, h = (0.7 * PX, 6) if horizontal else (6, 0.7 * PX)
        out.append(f'<rect x="{mx - w / 2:.1f}" y="{my - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{WOOD}" stroke="{WALL_DARK}" stroke-width="1"/>')

    # Labels.
    for r in g.rooms:
        bx, by, bw, bh = pen.box(r.rect)
        out.append(
            f'<text x="{bx + bw / 2:.1f}" y="{by + bh - 5:.1f}" font-size="9.5" font-style="italic" text-anchor="middle" '
            f'fill="{INK}" stroke="{FLOOR}" stroke-width="3" paint-order="stroke">{escape(r.id.replace("_", " "))}</text>'
        )
    return out


def _elevation(bp: Blueprint, geoms: list[_FloorGeom], x: float, y: float, w: float, h: float) -> list[str]:
    out = []
    widths = []
    for g in geoms:
        xs = [c[0] for c in g.mass]
        widths.append((max(xs) - min(xs) + 1))
    scale = (w - 20) / max(widths)
    slab = min(70.0, (h - 110) / bp.num_floors)
    base = y + h - 30
    cx = x + w / 2
    # Rock root under the ground floor.
    gw = widths[0] * scale
    out.append(f'<path d="M{cx - gw / 2:.1f} {base:.1f}Q{cx - gw * 0.2:.1f} {base + 25:.1f} {cx:.1f} {base + 28:.1f}Q{cx + gw * 0.25:.1f} {base + 22:.1f} {cx + gw / 2:.1f} {base:.1f}Z" fill="{ROCK}" stroke="{ROCK_EDGE}" stroke-width="2"/>')
    top_y = base
    for g, width in zip(geoms, widths):
        sw = width * scale
        sy = base - (g.floor.index + 1) * slab
        out.append(f'<rect x="{cx - sw / 2:.1f}" y="{sy:.1f}" width="{sw:.1f}" height="{slab + 1:.1f}" rx="6" fill="{ROCK}" stroke="{ROCK_EDGE}" stroke-width="2"/>')
        n = max(1, int(sw / 28))
        for k in range(n):
            wx = cx - sw / 2 + (k + 0.5) * sw / n
            out.append(f'<rect x="{wx - 3:.1f}" y="{sy + slab * 0.3:.1f}" width="6" height="{slab * 0.35:.1f}" rx="3" fill="#3b5670"/>')
        out.append(f'<text x="{cx - sw / 2 - 6:.1f}" y="{sy + slab / 2 + 4:.1f}" font-size="11" text-anchor="end" fill="{INK}">{g.floor.index}</text>')
        top_y = sy
    tw = widths[-1] * scale
    out.append(f'<path d="M{cx - tw / 2 - 6:.1f} {top_y + 2:.1f}L{cx:.1f} {top_y - 60:.1f}L{cx + tw / 2 + 6:.1f} {top_y + 2:.1f}Z" fill="#4f7fa3" stroke="#233a4d" stroke-width="2"/>')
    return out


def to_svg(bp: Blueprint, seed: int = 0) -> str:
    rng = random.Random(seed)
    geoms = [_FloorGeom(bp, f) for f in bp.floors]
    all_mass = [c for g in geoms for c in g.mass]
    x0 = min(c[0] for c in all_mass) - PAD
    x1 = max(c[0] for c in all_mass) + PAD + 1
    y0 = min(c[1] for c in all_mass) - PAD
    y1 = max(c[1] for c in all_mass) + PAD + 1
    cell_w, cell_h = (x1 - x0) * PX, (y1 - y0) * PX + 24
    cols = min(COLS, bp.num_floors)
    rows = math.ceil(bp.num_floors / cols)
    elev_w = 200
    width = elev_w + cols * cell_w + 20
    height = rows * cell_h + 60

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Georgia, serif">',
        DEFS,
        f'<rect width="100%" height="100%" fill="{SKY}"/>',
        f'<text x="20" y="36" font-size="24" font-weight="bold" fill="{INK}">{escape(bp.name)}</text>',
    ]
    out.extend(_elevation(bp, geoms, 10, 50, elev_w - 20, height - 60))
    for g in geoms:
        f = g.floor.index
        col, row = f % cols, rows - 1 - f // cols  # ground floor bottom-left
        ox, oy = elev_w + col * cell_w, 50 + row * cell_h
        out.append(f'<g transform="translate({ox},{oy + 24})" filter="url(#shadow)">')
        out.extend(_draw_floor(bp, g, _Pen(x0, y1), rng))
        out.append("</g>")
        out.append(
            f'<text x="{ox + cell_w / 2:.1f}" y="{oy + 16:.1f}" font-size="15" font-weight="bold" text-anchor="middle" fill="{INK}">'
            f"Floor {f} · +{g.floor.elevation:g} m</text>"
        )
    out.append("</svg>")
    return "\n".join(out)


def to_text(bp: Blueprint) -> str:
    lines = [f"{bp.name}: {bp.num_floors} floors, {bp.height:g} m tall"]
    for f in reversed(bp.floors):
        rooms = ", ".join(f.rooms) or "(empty)"
        lines.append(f"  F{f.index:<2} r={f.radius:5.2f}m {f.load:7.1f}/{f.capacity:7.1f} m²  {rooms}")
    m = bp.metrics
    lines.append(f"  route: {' -> '.join(bp.route)}")
    lines.append(
        f"  adjacency {m['adjacency_satisfied']}, separation {m['separation_satisfied']}, "
        f"energy {m['continuous_energy']['initial']:.2f} -> {m['continuous_energy']['final']:.2f}"
    )
    return "\n".join(lines)
