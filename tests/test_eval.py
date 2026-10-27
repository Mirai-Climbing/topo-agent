# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scoring hold finders (topo_agent/eval.py)."""
import unittest

from topo_agent import eval as hold_eval
from topo_agent.baseline import bench as hold_bench


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


if __name__ == "__main__":
    unittest.main()
