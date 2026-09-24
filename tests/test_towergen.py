import itertools
import json
import random
import unittest
from pathlib import Path

from towergen import HamiltonianWeights, ProgramGraph, generate
from towergen.circulation import heuristic_path, held_karp, path_cost
from towergen.hamiltonian import convolve, gradient, initial_state, potential

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "wizard_tower.json"


def load_example() -> ProgramGraph:
    return ProgramGraph.from_dict(json.loads(EXAMPLE.read_text()))


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

    def test_wedges_do_not_overlap_within_floor(self):
        for f in self.bp.floors:
            rooms = [self.bp.room(r) for r in f.rooms]
            total = sum(r.angle_end - r.angle_start for r in rooms)
            self.assertLessEqual(total, 360.0 + 1e-6)
            for a, b in zip(rooms, rooms[1:]):
                self.assertAlmostEqual(a.angle_end, b.angle_start, places=2)

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
