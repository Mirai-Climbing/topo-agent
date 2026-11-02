"""Tests for the YOLO-seg candidate shim (no torch needed)."""
import threading
import unittest

from topo_agent.yolo import YoloSeg, parse_point_prompt, point_in_poly, poly_to_candidate


class PolyTest(unittest.TestCase):
    def test_candidate_fields(self):
        poly = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.5), (0.1, 0.5)]
        c = poly_to_candidate(poly, 2800, 1864, 7, 0.9)
        self.assertEqual(c["id"], 7)
        self.assertAlmostEqual(c["x"], 0.2)
        self.assertAlmostEqual(c["y"], 0.3)
        self.assertAlmostEqual(c["r"], 0.2)
        self.assertAlmostEqual(c["area"], 0.2 * 0.4, places=4)
        self.assertEqual(c["box"], [0.1, 0.1, 0.3, 0.5])
        self.assertEqual(c["conf"], 0.9)
        self.assertEqual(len(c["points"]), 4)

    def test_long_outline_subsampled(self):
        poly = [(i / 100, 0.5) for i in range(100)]
        c = poly_to_candidate(poly, 100, 100, 0, 0.5)
        self.assertLessEqual(len(c["points"]), 32)
        self.assertGreaterEqual(len(c["points"]), 3)


class PromptTest(unittest.TestCase):
    def test_concept_is_none(self):
        self.assertIsNone(parse_point_prompt("climbing hold"))

    def test_point_parsed(self):
        p = "<0f> x=320;y=160;w=640;h=480 tail"
        self.assertEqual(parse_point_prompt(p), (0.5, 160 / 480))

    def test_containment(self):
        square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.assertTrue(point_in_poly(0.5, 0.5, square))
        self.assertFalse(point_in_poly(1.5, 0.5, square))


class SegmentTest(unittest.TestCase):
    def test_prompt_filtering(self):
        seg = YoloSeg.__new__(YoloSeg)
        seg.lock = threading.Lock()
        seg.cache = {}
        inner = [([(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)], 0.9)]
        outer = [([(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)], 0.8)]
        seg._infer = lambda jpeg: (100, 100, inner + outer)
        jpeg = b"fake"
        all_cands = seg.segment(jpeg, "climbing hold")
        self.assertEqual(len(all_cands), 2)
        tap = seg.segment(jpeg, "p x=50;y=50;w=100;h=100")
        self.assertEqual([c["conf"] for c in tap], [0.9])
        # second photo clears the single-entry cache
        seg._infer = lambda jpeg: (100, 100, outer)
        self.assertEqual(len(seg.segment(b"other", "climbing hold")), 1)


if __name__ == "__main__":
    unittest.main()
