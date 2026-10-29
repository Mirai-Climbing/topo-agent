# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scoring hold finders (topo_agent/eval.py)."""
import json
import os
import tempfile
import unittest

from PIL import Image

from topo_agent import eval as hold_eval
from topo_agent.baseline import bench as hold_bench


def photo_pair(d, name="wall", tap=(0.5, 0.5), polys=(((0.1, 0.1), (0.2, 0.1), (0.15, 0.2)),)):
    img = Image.new("RGB", (200, 100), (10, 20, 30))
    img.save(os.path.join(d, name + ".jpg"), "JPEG")
    with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
        json.dump({"tap": {"x": tap[0], "y": tap[1]},
                   "truth": [{"points": [list(p) for p in poly]} for poly in polys]}, f)


class EvalTest(unittest.TestCase):
    def test_match_pairs_each_hold_once(self):
        truth = [{"x": 0.1, "y": 0.1, "r": 0.02}, {"x": 0.5, "y": 0.5, "r": 0.02}]
        found = [{"x": 0.105, "y": 0.1, "r": 0.02}, {"x": 0.1, "y": 0.105, "r": 0.02}, {"x": 0.9, "y": 0.9, "r": 0.02}]
        self.assertEqual(hold_eval.match(truth, found, 1000, 1000), (1, 3, 2))  # two finds on one hold count once
        self.assertEqual(hold_eval.match(truth, [], 1000, 1000), (0, 0, 2))

    def test_tap_is_a_plain_hold_and_stable(self):
        holds = [{"use": "start", "x": 0.1}, {"use": "any", "x": 0.2}, {"use": "any", "x": 0.3}]
        tap = hold_eval.pick_tap(holds, "p1")
        self.assertEqual(tap["use"], "any")
        self.assertEqual(tap, hold_eval.pick_tap(holds, "p1"))

    def test_color_finder_on_the_hard_walls(self):
        # Hard sample walls (uneven light, black holds, a same-color problem far away, small same-color specks),
        # so that a drop in score is noticed
        items = hold_bench.items(("light", "black", "twins", "busy"), seeds=(11,))
        self.assertEqual(len(items[0]["marks"]["marks"]), hold_bench.PER_PROBLEM)
        rows = hold_eval.evaluate(items, hold_eval.finders({}, ["color"]))
        s = hold_eval.summary(rows)["color"]
        self.assertGreaterEqual(s["recall"], 0.7)
        self.assertGreaterEqual(s["precision"], 0.9)
        self.assertEqual(set(hold_eval.by_scene(rows)), {"light", "black", "twins", "busy"})

    def test_photo_items_loads_pairs_and_scores(self):
        with tempfile.TemporaryDirectory() as d:
            photo_pair(d, polys=(((0.1, 0.1), (0.2, 0.1), (0.15, 0.2)),
                                        ((0.5, 0.5), (0.6, 0.5), (0.55, 0.6))))
            items = hold_eval.photo_items(d)
            self.assertEqual(len(items), 1)
            it = items[0]
            self.assertEqual(it["tap"], {"x": 0.5, "y": 0.5})
            self.assertEqual(it["marks"]["photo"], {"width": 200, "height": 100})
            self.assertEqual(len(it["marks"]["marks"]), 2)
            self.assertAlmostEqual(it["marks"]["marks"][0]["x"], 0.15)
            rows = hold_eval.evaluate(items, {"fake": lambda jpeg, point: [{"x": 0.15, "y": 0.133, "r": 0.01}]})
            self.assertEqual((rows[0]["hits"], rows[0]["found"], rows[0]["truth"]), (1, 1, 2))

    def test_photo_items_rejects_bad_cases(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                hold_eval.photo_items(d)  # no pairs
        with tempfile.TemporaryDirectory() as d:
            photo_pair(d)
            os.remove(os.path.join(d, "wall.jpg"))
            with self.assertRaises(SystemExit):
                hold_eval.photo_items(d)  # json without a photo
        with tempfile.TemporaryDirectory() as d:
            photo_pair(d, tap=(9, 0))
            with self.assertRaises(SystemExit):
                hold_eval.photo_items(d)  # tap outside 0-1
        with tempfile.TemporaryDirectory() as d:
            photo_pair(d, polys=())
            with self.assertRaises(SystemExit):
                hold_eval.photo_items(d)  # no truth holds

    def test_photo_items_loads_two_routes_from_one_image(self):
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "images"))
            img = Image.new("RGB", (200, 100), (10, 20, 30))
            img.save(os.path.join(d, "images", "wall.jpg"), "JPEG")
            route = {"tap": {"x": 0.15, "y": 0.15},
                     "truth": [{"points": [[0.1, 0.1], [0.2, 0.1], [0.15, 0.2]]}]}
            with open(os.path.join(d, "annotations.json"), "w", encoding="utf-8") as f:
                json.dump({"images": [{"id": "wall", "file": "images/wall.jpg", "routes": [
                    {"route_id": "1", **route}, {"route_id": "2", **route}]}]}, f)
            items = hold_eval.photo_items(d)
            self.assertEqual([item["id"] for item in items], ["wall_r1", "wall_r2"])
            self.assertEqual([len(item["marks"]["marks"]) for item in items], [1, 1])
            self.assertEqual(items[0]["jpeg"], items[1]["jpeg"])

    def test_photo_items_rejects_duplicate_manifest_route(self):
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "images"))
            Image.new("RGB", (200, 100)).save(os.path.join(d, "images", "wall.jpg"), "JPEG")
            route = {"route_id": "1", "tap": {"x": 0.15, "y": 0.15},
                     "truth": [{"points": [[0.1, 0.1], [0.2, 0.1], [0.15, 0.2]]}]}
            with open(os.path.join(d, "annotations.json"), "w", encoding="utf-8") as f:
                json.dump({"images": [{"id": "wall", "file": "images/wall.jpg", "routes": [route, route]}]}, f)
            with self.assertRaises(SystemExit):
                hold_eval.photo_items(d)


if __name__ == "__main__":
    unittest.main()
