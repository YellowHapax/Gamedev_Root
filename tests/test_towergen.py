import contextlib
import io
import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path

from towergen import HamiltonianWeights, ProgramGraph, generate
from towergen.circulation import heuristic_path, held_karp, path_cost
from towergen.cli import main as cli_main
from towergen.hamiltonian import convolve, gradient, initial_state, potential
from towergen.plan import Rect

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "wizard_tower.json"


def load_example() -> ProgramGraph:
    return ProgramGraph.from_dict(json.loads(EXAMPLE.read_text(encoding="utf-8")))


class HamiltonianTests(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        g = load_example()
        w = HamiltonianWeights()
        x = initial_state(g, random.Random(3))
        grad = gradient(g, x, w)
        h = 1e-6
        for i in (0, 5, 13):
            for c in range(3):
                xp = [row[:] for row in x]
                xm = [row[:] for row in x]
                xp[i][c] += h
                xm[i][c] -= h
                fd = (potential(g, xp, w) - potential(g, xm, w)) / (2 * h)
                self.assertAlmostEqual(grad[i][c], fd, places=4)

    def test_convolution_lowers_energy_and_respects_anchors(self):
        g = load_example()
        x, trace = convolve(g, layers=200)
        self.assertLess(trace[-1], trace[0])
        z = {r.id: x[i][0] for i, r in enumerate(g.rooms)}
        self.assertLess(z["gate"], 0.1)
        self.assertGreater(z["observatory"], 0.9)


class PathTests(unittest.TestCase):
    def _instance(self, n, seed):
        rng = random.Random(seed)
        cost = [[rng.uniform(-1, 3) for _ in range(n)] for _ in range(n)]
        entry = [rng.uniform(0, 2) for _ in range(n)]
        return cost, entry

    def test_held_karp_is_optimal(self):
        for seed in range(5):
            cost, entry = self._instance(6, seed)
            best = min(path_cost(cost, entry, list(p)) for p in itertools.permutations(range(6)))
            c, path = held_karp(cost, entry)
            self.assertAlmostEqual(c, best)
            self.assertEqual(sorted(path), list(range(6)))
            self.assertAlmostEqual(path_cost(cost, entry, path), c)

    def test_heuristic_returns_hamiltonian_path(self):
        cost, entry = self._instance(9, 1)
        c, path = heuristic_path(cost, entry)
        self.assertEqual(sorted(path), list(range(9)))
        self.assertGreaterEqual(c + 1e-9, held_karp(cost, entry)[0])


class BlueprintTests(unittest.TestCase):
    def setUp(self):
        self.graph = load_example()
        self.bp = generate(self.graph, seed=0)

    def test_route_visits_every_room_once(self):
        self.assertEqual(sorted(self.bp.route), sorted(r.id for r in self.graph.rooms))
        self.assertEqual(self.bp.route[0], "gate")

    def test_anchors_and_capacity(self):
        self.assertEqual(self.bp.room("gate").floor, 0)
        self.assertEqual(self.bp.room("cistern").floor, 0)
        self.assertEqual(self.bp.room("observatory").floor, self.bp.num_floors - 1)
        for f in self.bp.floors:
            self.assertLessEqual(f.load, f.capacity)
            self.assertTrue(f.rooms, f"floor {f.index} is empty")

    def test_route_climbs_monotonically(self):
        floors = [self.bp.room(r).floor for r in self.bp.route]
        self.assertEqual(floors, sorted(floors))

    def _footprints(self, f):
        rects = [Rect(r, *self.bp.room(r).rect) for r in f.rooms]
        for key, st in (("down", f.stair_down), ("up", f.stair_up)):
            if st:
                rects.append(Rect(key, *st))
        return rects

    def test_footprints_keep_a_clear_cell_between_them(self):
        for f in self.bp.floors:
            rects = self._footprints(f)
            for a, b in itertools.combinations(rects, 2):
                self.assertFalse(a.intersects(b, margin=1), f"floor {f.index}: {a.key} touches {b.key}")
            owned = {c for r in rects for c in r.cells()}
            self.assertFalse(owned & {tuple(c) for c in f.corridors}, "corridor runs through a room")

    def test_every_room_is_reachable(self):
        """Flood through corridors and doors from the stair (or the front door)."""
        for f in self.bp.floors:
            rects = self._footprints(f)
            corridors = {tuple(c) for c in f.corridors}
            doors = [tuple(d) for r in f.rooms for d in self.bp.room(r).doors] + [tuple(d) for d in f.doors]
            owner = {c: r.key for r in rects for c in r.cells()}
            links, fronts = {}, {}
            for ox, oy, ix, iy in doors:
                self.assertIn((ox, oy), corridors)
                links.setdefault((ox, oy), set()).add(owner[(ix, iy)])
                fronts.setdefault(owner[(ix, iy)], set()).add((ox, oy))
            start = next(iter(corridors), None)
            if start is None:
                self.assertEqual(len(rects), 1)
                continue
            # Walk corridors, and through a room from any of its doors to the others.
            seen, stack, reached = set(), [start], set()
            while stack:
                c = stack.pop()
                if c in seen:
                    continue
                seen.add(c)
                for room in links.get(c, ()):
                    reached.add(room)
                    stack.extend(fronts[room])
                stack.extend(n for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1)) if n in corridors)
            self.assertEqual(reached, {r.key for r in rects}, f"floor {f.index}")

    def test_spiral_stairs_line_up(self):
        for lower, upper in zip(self.bp.floors, self.bp.floors[1:]):
            self.assertIsNotNone(lower.stair_up)
            self.assertEqual(lower.stair_up, upper.stair_down)
        self.assertIsNone(self.bp.floors[0].stair_down)
        self.assertIsNone(self.bp.floors[-1].stair_up)

    def test_front_door(self):
        exits = self.bp.floors[0].exits
        self.assertEqual(len(exits), 1)
        gate = Rect("gate", *self.bp.room("gate").rect)
        self.assertIn(tuple(exits[0][2:]), gate.cells())

    def test_svg_is_well_formed(self):
        from xml.dom.minidom import parseString

        from towergen.render import to_svg

        parseString(to_svg(self.bp))

    def test_cli_writes_utf8_files(self):
        # The SVG holds non-ASCII glyphs; Windows' default cp1252 cannot encode them.
        with tempfile.TemporaryDirectory() as out:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main([str(EXAMPLE), "-o", out]), 0)
            svg = (Path(out) / "wizard_tower.svg").read_text(encoding="utf-8")
            self.assertTrue(svg.rstrip().endswith("</svg>"))
            bp = json.loads((Path(out) / "wizard_tower.blueprint.json").read_text(encoding="utf-8"))
            self.assertEqual(bp["route"][0], "gate")

    def test_heavy_rooms_sink(self):
        self.assertLess(self.bp.room("forge").floor, self.bp.room("library").floor)

    def test_deterministic(self):
        self.assertEqual(generate(self.graph, seed=0).to_dict(), self.bp.to_dict())

    def test_rejects_bad_program(self):
        with self.assertRaises(ValueError):
            ProgramGraph.from_dict({"rooms": [{"id": "a", "kind": "x", "area": 1}],
                                    "edges": [{"a": "a", "b": "nope"}]})


if __name__ == "__main__":
    unittest.main()
