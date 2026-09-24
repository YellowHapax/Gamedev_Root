"""SVG and plain-text renderings of a Blueprint."""

from __future__ import annotations

import math
from xml.sax.saxutils import escape

from .blueprint import Blueprint

PLAN_SIZE = 260  # px per floor plan cell
MARGIN = 20


PALETTE = [
    "#e6b35c", "#8fb9e0", "#a8d08d", "#e89a8f", "#c3a6e0",
    "#7fcfc4", "#f0d78a", "#d4a5c0", "#b8c47a", "#9fb3c8",
]


def kind_colors(bp: Blueprint) -> dict[str, str]:
    kinds = sorted({r.kind for r in bp.rooms})
    return {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(kinds)}


def _polar(cx: float, cy: float, r: float, deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    return cx + r * math.cos(a), cy - r * math.sin(a)  # SVG y points down


def _wedge(cx, cy, r0, r1, a0, a1) -> str:
    if a1 - a0 >= 359.999:
        a1 = a0 + 359.999
    large = 1 if a1 - a0 > 180 else 0
    x0, y0 = _polar(cx, cy, r1, a0)
    x1, y1 = _polar(cx, cy, r1, a1)
    x2, y2 = _polar(cx, cy, r0, a1)
    x3, y3 = _polar(cx, cy, r0, a0)
    return (
        f"M{x0:.2f},{y0:.2f} A{r1:.2f},{r1:.2f} 0 {large} 0 {x1:.2f},{y1:.2f} "
        f"L{x2:.2f},{y2:.2f} A{r0:.2f},{r0:.2f} 0 {large} 1 {x3:.2f},{y3:.2f} Z"
    )


def to_svg(bp: Blueprint) -> str:
    cols = min(3, bp.num_floors)
    rows = math.ceil(bp.num_floors / cols)
    elev_w = 180
    width = elev_w + cols * PLAN_SIZE + 2 * MARGIN
    height = max(rows * PLAN_SIZE, 320) + 2 * MARGIN + 30
    max_r = max(f.radius for f in bp.floors)
    scale = (PLAN_SIZE / 2 - 22) / max_r
    by_id = {r.id: r for r in bp.rooms}
    colors = kind_colors(bp)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">',
        f'<rect width="100%" height="100%" fill="#fbfaf6"/>',
        f'<text x="{MARGIN}" y="{MARGIN + 6}" font-size="16" font-weight="bold">{escape(bp.name)}</text>',
    ]

    # Legend.
    lx = MARGIN + 180
    for kind, color in colors.items():
        out.append(f'<rect x="{lx}" y="{MARGIN - 4}" width="10" height="10" fill="{color}" stroke="#4a4030"/>')
        out.append(f'<text x="{lx + 14}" y="{MARGIN + 5}" font-size="10">{escape(kind)}</text>')
        lx += 20 + 6 * len(kind)

    # Elevation: tapered floor slabs, ground at the bottom.
    top_y = MARGIN + 40
    avail_h = height - top_y - MARGIN
    slab_h = avail_h / bp.num_floors
    ex = MARGIN + elev_w / 2 - 10
    escale = (elev_w / 2 - 20) / max_r
    for f in bp.floors:
        y = top_y + (bp.num_floors - 1 - f.index) * slab_h
        hw = f.radius * escale
        out.append(
            f'<rect x="{ex - hw:.1f}" y="{y:.1f}" width="{2 * hw:.1f}" height="{slab_h - 3:.1f}" '
            f'fill="#e8e2d4" stroke="#6b5d45"/>'
        )
        out.append(
            f'<text x="{ex:.1f}" y="{y + slab_h / 2 + 4:.1f}" font-size="11" text-anchor="middle">'
            f"F{f.index} · {len(f.rooms)} rooms</text>"
        )

    # Floor plans.
    for f in bp.floors:
        col, row = f.index % cols, f.index // cols
        ox = MARGIN + elev_w + col * PLAN_SIZE
        oy = MARGIN + 30 + row * PLAN_SIZE
        cx, cy = ox + PLAN_SIZE / 2, oy + PLAN_SIZE / 2
        R, r0 = f.radius * scale, bp.core_radius * scale
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{R:.1f}" fill="#f1ede3" stroke="#6b5d45" stroke-width="2"/>')
        for rid in f.rooms:
            room = by_id[rid]
            path = _wedge(cx, cy, r0, R, room.angle_start, room.angle_end)
            out.append(f'<path d="{path}" fill="{colors[room.kind]}" stroke="#4a4030"><title>{escape(rid)} ({room.area} m²)</title></path>')
            mid = (room.angle_start + room.angle_end) / 2
            lx, ly = _polar(cx, cy, (r0 + R) / 2, mid)
            out.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="9" text-anchor="middle" dominant-baseline="middle">'
                f"{room.route_index}. {escape(rid)}</text>"
            )
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r0:.1f}" fill="#d6cdb8" stroke="#4a4030"/>')
        # Stair: landing (green) and exit (red) on the core.
        for ang, color in ((f.landing_angle, "#2e8b57"), (f.exit_angle, "#c0392b")):
            px, py = _polar(cx, cy, r0, ang)
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>')
        out.append(
            f'<text x="{cx:.1f}" y="{oy + 12:.1f}" font-size="12" text-anchor="middle" font-weight="bold">'
            f"Floor {f.index} · +{f.elevation:g} m · {f.load:g}/{f.capacity:g} m²</text>"
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
