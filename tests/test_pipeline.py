# SPDX-License-Identifier: AGPL-3.0-or-later
"""The three-stage pipeline (topo_agent/pipeline.py), with a fake SAM and fake Spark (no network)."""
import io
import json
import os
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


class PromptVariantTest(unittest.TestCase):
    def test_resolver_defaults_to_v1(self):
        self.assertEqual(pipeline.system_prompt(""), ("v1", pipeline.SYSTEM3))
        self.assertEqual(pipeline.system_prompt(None), ("v1", pipeline.SYSTEM3))
        self.assertEqual(pipeline.system_prompt("v1"), ("v1", pipeline.SYSTEM3))
        self.assertEqual(pipeline.system_prompt("v2"), ("v2", pipeline.SYSTEM3_V2))
        self.assertEqual(pipeline.system_prompt("v3"), ("v3", pipeline.SYSTEM3_V3))

    def test_unknown_variant_falls_back_to_v1(self):
        with self.assertLogs(pipeline.log, "WARNING"):
            self.assertEqual(pipeline.system_prompt("v9"), ("v1", pipeline.SYSTEM3))

    def test_all_variants_keep_the_json_contract(self):
        for name in ("v1", "v2", "v3"):
            text = pipeline.system_prompt(name)[1]
            with self.subTest(variant=name):
                self.assertIn('{"hold_ids": [1, 4, 7], "confidence": 0.95}', text)
                self.assertIn("Always include the tapped hold", text)
                self.assertIn("single inference pass", text)
                self.assertIn("tape", text)

    def test_real_photo_rules(self):
        v2 = pipeline.system_prompt("v2")[1]
        self.assertIn("Trust your eyes over the table", v2)  # tap color override
        self.assertIn("Pale gray, white and beige", v2)  # pale-wall skepticism
        self.assertIn("ignore brightness", v2)  # shadowed holds still count
        v3 = pipeline.system_prompt("v3")[1]
        self.assertIn("Trust your eyes over the table", v3)
        self.assertIn("Pale gray, white and beige", v3)
        self.assertIn("OTHER groups", v3)  # routes split across groups are merged
        self.assertIn("through chains", v3)  # near is transitive, not just direct
        self.assertIn("4 to 15 holds", v3)  # tiny answers are re-checked
        self.assertNotIn("OTHER groups", pipeline.SYSTEM3)  # v1 is untouched

    def test_run_once_sends_selected_variant_in_one_call(self):
        cands = pipeline.add_neighbors(pipeline.group_by_color(candidates()), W, H)
        for prompt, name, expected in (("", "v1", pipeline.SYSTEM3),
                                       ("v2", "v2", pipeline.SYSTEM3_V2),
                                       ("v3", "v3", pipeline.SYSTEM3_V3)):
            with self.subTest(prompt=prompt or "default"):
                spark = RunOnceTest().route_reply()
                ids, info = pipeline.run_once(spark, wall(), cands, 0, prompt=prompt)
                self.assertEqual(ids, [0, 1])
                self.assertEqual(len(spark.seen), 1)  # still exactly one model call
                self.assertEqual(spark.seen[0][0]["content"], expected)
                self.assertEqual(info["prompt"], name)

    def test_fallback_reports_variant_without_retry(self):
        cands = pipeline.add_neighbors(pipeline.group_by_color(candidates()), W, H)
        calls = []

        def spark(messages):
            calls.append(messages)
            return {"text": "not json", "usage": {}}

        ids, info = pipeline.run_once(spark, wall(), cands, 0, prompt="v3")
        self.assertEqual(len(calls), 1)  # no retry
        self.assertEqual(ids, [0, 1])  # the tapped hold's CV color group
        self.assertEqual(info["prompt"], "v3")
        self.assertEqual(info["fallback"], "not valid JSON")

    def test_find_with_threads_prompt_and_traces_it(self):
        jpeg = jpeg_of(wall())
        tap = {"x": 230 / W, "y": 225 / H}

        def post(body):
            return {"output_text": "<0f>" + ",".join(record(i, box) for i, (box, _) in enumerate(HOLDS))}

        spark = RunOnceTest().route_reply()
        traces = []
        holds = pipeline.find_with(sam.Sam(post), spark, jpeg, tap,
                                   on_trace=traces.append, prompt="v2")
        self.assertEqual(len(holds), 2)
        self.assertEqual(spark.seen[0][0]["content"], pipeline.SYSTEM3_V2)
        self.assertEqual(traces[0][2]["args"]["prompt"], "v2")
        self.assertEqual(traces[0][2]["args"]["spark_calls"], 1)

    def test_route_finder_honors_meta3_prompt_env(self):
        jpeg = jpeg_of(wall())
        tap = {"x": 230 / W, "y": 225 / H}
        sent = []

        def sam_post(body):
            return {"output_text": "<0f>" + ",".join(record(i, box) for i, (box, _) in enumerate(HOLDS))}

        def chat(body):
            sent.append(body)
            return {"choices": [{"message": {"content": json.dumps({"hold_ids": [0, 1],
                                                                                  "confidence": 0.9})}}],
                    "usage": {}}

        with mock.patch.object(sam, "http_post", lambda env: sam_post), \
                mock.patch.object(agent, "http_chat", lambda env: chat):
            for env, expected in (({}, pipeline.SYSTEM3),  # default keeps v1
                                  ({"META3_PROMPT": "v2"}, pipeline.SYSTEM3_V2),
                                  ({"META3_PROMPT": "v3"}, pipeline.SYSTEM3_V3)):
                with self.subTest(env=env):
                    traces = []
                    holds = pipeline.route_finder(env, on_trace=traces.append)(jpeg, tap)
                    self.assertEqual(len(holds), 2)
                    body = sent[-1]
                    self.assertEqual(body["messages"][0]["content"], expected)
                    self.assertEqual(body["temperature"], 0.2)
                    self.assertIs(body["store"], False)
                    self.assertNotIn("tools", body)


def fake_embedder(crops):
    """Fixed vectors for the three test holds (no torch): the two reds match, blue differs."""
    assert len(crops) == 3
    return [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]


class EmbedTest(unittest.TestCase):
    def test_embed_top_from_fake_vectors(self):
        cands = pipeline.add_embed_similar(wall(), candidates(), embedder=fake_embedder, top_k=2)
        self.assertEqual([i for i, _ in cands[0]["embed_top"]], [1, 2])  # the other red first
        self.assertGreater(cands[0]["embed_top"][0][1], 0.99)
        self.assertEqual(cands[2]["embed_top"][0][0], 1)
        self.assertEqual(len(cands[1]["embed_top"]), 2)

    def test_prompt_table_shows_embed_only_when_present(self):
        cands = pipeline.add_neighbors(pipeline.group_by_color(candidates()), W, H)
        self.assertNotIn("embed", pipeline.prompt_table(cands, 0))  # default off: no column
        pipeline.add_embed_similar(wall(), cands, embedder=fake_embedder, top_k=1)
        table = pipeline.prompt_table(cands, 0)
        self.assertIn("| embed", table)
        self.assertIn("1:0.99", table)  # hold 0* lists hold 1 at cosine 0.99

    def test_hold_crop_masked_and_circle(self):
        img = wall()
        outlined = candidates()[0]
        masked = pipeline.hold_crop(img, outlined)
        plain = pipeline.hold_crop(img, outlined, masked=False)
        self.assertEqual(masked.size, plain.size)
        self.assertGreater(masked.width, 4)
        self.assertGreater(masked.height, 4)
        circle = pipeline.hold_crop(img, dict(outlined, points=[]))  # no outline: ellipse fallback
        self.assertGreater(circle.width, 4)
        self.assertGreater(circle.height, 4)

    def test_embed_enabled_defaults_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(pipeline.embed_enabled())
            self.assertFalse(pipeline.embed_enabled({}))
            self.assertTrue(pipeline.embed_enabled({"META3_EMBED": "1"}))
        with mock.patch.dict(os.environ, {"META3_EMBED": "1"}):
            self.assertTrue(pipeline.embed_enabled())
            self.assertFalse(pipeline.embed_enabled({"META3_EMBED": "0"}))  # explicit env wins

    def find(self, **kwargs):
        jpeg = jpeg_of(wall())
        tap = {"x": 230 / W, "y": 225 / H}

        def post(body):
            return {"output_text": "<0f>" + ",".join(record(i, box) for i, (box, _) in enumerate(HOLDS))}

        spark = RunOnceTest().route_reply()
        traces = []
        with mock.patch.dict(os.environ, {}, clear=True):
            holds = pipeline.find_with(sam.Sam(post), spark, jpeg, tap,
                                       on_trace=traces.append, **kwargs)
        return holds, traces[0], spark

    def test_find_with_embed_adds_trace_and_table_signal(self):
        holds, trace, spark = self.find(use_embed=True, embedder=fake_embedder)
        self.assertEqual(len(holds), 2)
        self.assertEqual([s["tool"] for s in trace], ["(sam)", "(color)", "(embed)", "(spark)", "(post)"])
        self.assertIn("| embed", spark.seen[0][1]["content"][0]["text"])

    def test_find_with_embed_off_by_default(self):
        holds, trace, spark = self.find()
        self.assertEqual(len(holds), 2)
        self.assertEqual([s["tool"] for s in trace], ["(sam)", "(color)", "(spark)", "(post)"])
        self.assertNotIn("embed", spark.seen[0][1]["content"][0]["text"])

    def test_find_with_embed_failure_is_silent(self):
        for boom in (ImportError("no torch"), RuntimeError("cuda busy")):
            def broken(crops, err=boom):
                raise err

            holds, trace, spark = self.find(use_embed=True, embedder=broken)
            self.assertEqual(len(holds), 2)  # the Spark answer is still used
            self.assertEqual([s["tool"] for s in trace], ["(sam)", "(color)", "(spark)", "(post)"])
            self.assertNotIn("embed", spark.seen[0][1]["content"][0]["text"])

    def test_add_embed_similar_reraises_import_error_of_explicit_embedder(self):
        def broken(crops):
            raise ImportError("no torch")

        with self.assertRaises(ImportError):
            pipeline.add_embed_similar(wall(), candidates(), embedder=broken)
