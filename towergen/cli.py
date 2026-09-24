"""Command line: python -m towergen PROGRAM.json [-o OUT_DIR] [--seed N]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .blueprint import generate
from .graph import ProgramGraph
from .render import to_svg, to_text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="towergen", description="Tower blueprint creator")
    ap.add_argument("program", type=Path, help="program graph JSON")
    ap.add_argument("-o", "--out", type=Path, default=Path("out"), help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=300, help="Hamiltonian convolution layers")
    args = ap.parse_args(argv)

    graph = ProgramGraph.from_dict(json.loads(args.program.read_text()))
    bp = generate(graph, layers=args.layers, seed=args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.program.stem
    (args.out / f"{stem}.blueprint.json").write_text(json.dumps(bp.to_dict(), indent=2))
    (args.out / f"{stem}.svg").write_text(to_svg(bp))
    print(to_text(bp))
    print(f"wrote {args.out / (stem + '.blueprint.json')} and {args.out / (stem + '.svg')}")
    return 0
