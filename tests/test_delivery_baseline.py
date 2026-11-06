import unittest
from pathlib import Path

import numpy as np

from delivery_baseline import load_locations, plan, route_length, two_opt


ROOT = Path(__file__).resolve().parents[1]


class DeliveryTests(unittest.TestCase):
    def test_existing_coordinate_files(self):
        for name in ("locations1.csv", "locations2.csv"):
            points = load_locations(ROOT / name)
            self.assertEqual(points.shape, (500, 2))
            result = plan(points, trucks=10)
            visited = [i for route in result["routes"] for i in route["locations"]]
            self.assertEqual(sorted(visited), list(range(500)))
            self.assertAlmostEqual(result["total_distance"], sum(r["distance"] for r in result["routes"]))

    def test_two_opt_never_worsens_crossing_route(self):
        points = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        route = [0, 1, 2, 3]
        self.assertLess(route_length(two_opt(route, distances), distances), route_length(route, distances))

    def test_invalid_coordinates_and_truck_count(self):
        with self.assertRaises(ValueError):
            plan(np.array([[0, 0]]), 2)
        with self.assertRaises(ValueError):
            load_locations(ROOT / "Project 2 Report.pdf")


if __name__ == "__main__":
    unittest.main()
