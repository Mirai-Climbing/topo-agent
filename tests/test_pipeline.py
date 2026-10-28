# SPDX-License-Identifier: AGPL-3.0-or-later
"""The three-stage pipeline (topo_agent/pipeline.py), with a fake SAM and fake Spark (no network)."""
import io
import json
import unittest
from unittest import mock

from PIL import Image, ImageDraw

from tests.test_sam import record
from topo_agent import agent, pipeline, sam

W, H = 800, 600
RED, BLUE, WALL = (200, 40, 40), (40, 70, 200), (214, 206, 192)
HOLDS = [((200, 200, 260, 250), RED), ((330, 260, 360, 285), RED), ((400, 200, 430, 225), BLUE)]


def wall(highlight=True):
    img = Image.new("RGB", (W, H), WALL)
    d = ImageDraw.Draw(img)
    for box, color in HOLDS:
        d.ellipse(box, fill=color)
    if highlight:  # a washed-out spot inside the first red hold
        d.ellipse((215, 210, 245, 235), fill=(245, 240, 235))
    return img


def jpeg_of(img):
    out = io.BytesIO()
    img.save(out, "JPEG", quality=95)
    return out.getvalue()


def candidates(img=None):
    objs = sam.parse("<0f>" + ",".join(record(i, box) for i, (box, _) in enumerate(HOLDS)))
    return pipeline.classify_colors(img or wall(), [sam.to_candidate(o) for o in objs])


def dilate(pts, factor=1.2):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return [[cx + (x - cx) * factor, cy + (y - cy) * factor] for x, y in pts]


class ColorTest(unittest.TestCase):
    def test_lab_basics(self):
        self.assertAlmostEqual(pipeline.rgb_to_lab((255, 255, 255))[0], 100, delta=0.5)
        self.assertAlmostEqual(pipeline.rgb_to_lab((0, 0, 0))[0], 0, delta=0.5)
        self.assertGreater(pipeline.rgb_to_lab((200, 40, 40))[1], 20)  # red is +a

    def test_classify_names(self):
        cands = candidates()
        self.assertEqual([c["color_name"] for c in cands], ["red", "red", "blue"])
        for c in cands:
            self.assertEqual(len(c["hsv"]), 3)
            self.assertEqual(len(c["lab"]), 3)

    def test_highlight_does_not_shift_hue(self):
        red = candidates()[0]
        plain = pipeline.classify_colors(wall(highlight=False), [dict(red, points=list(red["points"]))])[0]
        self.assertEqual(red["color_name"], "red")
        self.assertAlmostEqual(red["hsv"][0], plain["hsv"][0], delta=0.03)

    def test_boundary_fringe_is_excluded(self):
        tight = candidates()[1]
        wide = dict(tight, points=dilate(tight["points"]))  # loose outline leaking wall pixels in
        wide = pipeline.classify_colors(wall(), [wide])[0]
        self.assertEqual(wide["color_name"], "red")
        self.assertAlmostEqual(wide["hsv"][0], tight["hsv"][0], delta=0.03)

    def test_groups_and_neighbors(self):
        cands = pipeline.add_neighbors(pipeline.group_by_color(candidates()), W, H)
        self.assertEqual(cands[0]["group"], cands[1]["group"])  # the two reds
        self.assertNotEqual(cands[0]["group"], cands[2]["group"])  # blue is its own group
        self.assertIn(1, cands[0]["near"])  # same-group holds near each other
        self.assertNotIn(2, cands[0]["near"])  # other groups are never neighbors


class ParseTest(unittest.TestCase):
    def test_valid(self):
        out = pipeline.parse_answer(json.dumps({"hold_ids": [1, 4], "confidence": 0.9}), {1, 4, 9})
        self.assertEqual(out["error"], "")
        self.assertEqual(out["ids"], [1, 4])
        self.assertEqual(out["confidence"], 0.9)

    def test_fences_are_unwrapped(self):
        out = pipeline.parse_answer('```json\n{"hold_ids": [1], "confidence": 0.5}\n```', {1})
        self.assertEqual(out["error"], "")

    def test_rejects(self):
        bad = [("not json", "not valid JSON"),
               (json.dumps({"hold_ids": [], "confidence": 0.9}), "missing hold_ids"),
               (json.dumps({"hold_ids": [1]}), "missing hold_ids"),
               (json.dumps({"hold_ids": [2], "confidence": 0.9}), "unknown id"),
               (json.dumps({"hold_ids": [1, 1], "confidence": 0.9}), "twice"),
               (json.dumps({"hold_ids": [1], "confidence": 2}), "missing hold_ids"),
               (json.dumps({"routes": []}), "missing hold_ids")]
        for text, hint in bad:
            with self.subTest(text=text[:40]):
                self.assertIn(hint, pipeline.parse_answer(text, {1})["error"])


class RunOnceTest(unittest.TestCase):
    def route_reply(self, ids=(0, 1)):
        def spark(messages):
            spark.seen.append(messages)
            return {"text": json.dumps({"hold_ids": list(ids), "confidence": 0.9}),
                    "usage": {"prompt_tokens": 10}}
        spark.seen = []
        return spark

    def test_single_pass_returns_tapped_route(self):
        cands = pipeline.add_neighbors(pipeline.group_by_color(candidates()), W, H)
        spark = self.route_reply()
        ids, info = pipeline.run_once(spark, wall(), cands, 0)
        self.assertEqual(ids, [0, 1])
        self.assertEqual(info["fallback"], "")
        self.assertEqual(info["confidence"], 0.9)
        self.assertEqual(len(spark.seen), 1)  # exactly one model call
        user = spark.seen[0][1]["content"]
        self.assertEqual(len(user), 4)  # table text + photo + numbered candidates + tap close-up
        self.assertIn("0*", user[0]["text"])  # the tap is marked in the table

    def test_invalid_json_falls_back_without_retry(self):
        cands = pipeline.add_neighbors(pipeline.group_by_color(candidates()), W, H)
        calls = []

        def spark(messages):
            calls.append(messages)
            return {"text": "not json", "usage": {}}

        ids, info = pipeline.run_once(spark, wall(), cands, 0)
        self.assertEqual(len(calls), 1)  # no retry
        self.assertEqual(ids, [0, 1])  # the tapped hold's CV color group
        self.assertEqual(info["fallback"], "not valid JSON")

    def test_spark_failure_falls_back(self):
        cands = pipeline.add_neighbors(pipeline.group_by_color(candidates()), W, H)

        def spark(messages):
            raise RuntimeError("503")

        ids, info = pipeline.run_once(spark, wall(), cands, 0)
        self.assertEqual(ids, [0, 1])
        self.assertIn("503", info["fallback"])

    def test_tap_is_included_even_when_model_omits_it(self):
        cands = pipeline.add_neighbors(pipeline.group_by_color(candidates()), W, H)
        ids, info = pipeline.run_once(self.route_reply(ids=(1,)), wall(), cands, 0)
        self.assertEqual(ids, [0, 1])  # clean_ids puts the tapped hold first
        self.assertEqual(info["fallback"], "")


class FinderTest(unittest.TestCase):
    def test_find_with_reports_stages(self):
        jpeg = jpeg_of(wall())
        tap = {"x": 230 / W, "y": 225 / H}

        def post(body):
            return {"output_text": "<0f>" + ",".join(record(i, box) for i, (box, _) in enumerate(HOLDS))}

        spark = RunOnceTest().route_reply()
        traces = []
        holds = pipeline.find_with(sam.Sam(post), spark, jpeg, tap, on_trace=traces.append)
        self.assertEqual(len(holds), 2)
        self.assertEqual([s["tool"] for s in traces[0]], ["(sam)", "(color)", "(spark)", "(post)"])
        self.assertEqual(traces[0][0]["args"]["count"], 3)
        self.assertEqual(traces[0][2]["args"]["spark_calls"], 1)
        self.assertEqual(len(spark.seen), 1)

    def test_route_finder_needs_key_and_survives_sam_failure(self):
        self.assertIsNone(pipeline.route_finder({}))
        jpeg = jpeg_of(wall())
        tap = {"x": 230 / W, "y": 225 / H}

        def broken(body):
            raise RuntimeError("503")

        traces = []
        with mock.patch.object(sam, "http_post", lambda env: broken), \
                mock.patch.object(agent, "http_chat", lambda env: broken):
            holds = pipeline.route_finder({}, on_trace=traces.append)(jpeg, tap)
        self.assertTrue(holds)  # color only still finds the red problem
        self.assertEqual(traces[0][-1]["tool"], "(fallback: color)")

    def test_single_call_sends_no_tools(self):
        sent = []

        def post(body):
            sent.append(body)
            return {"choices": [{"message": {"content": "{}"}}], "usage": {"total_tokens": 5}}

        out = pipeline.single_call(post, effort="")([])  # empty effort: the API default
        self.assertEqual(out, {"text": "{}", "usage": {"total_tokens": 5}})
        self.assertEqual(sent[0]["temperature"], 0.2)  # same sampling as the ReAct agent
        self.assertIs(sent[0]["store"], False)
        self.assertNotIn("tools", sent[0])
        self.assertNotIn("reasoning_effort", sent[0])
