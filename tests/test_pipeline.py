# SPDX-License-Identifier: AGPL-3.0-or-later
"""The three-stage pipeline (topo_agent/pipeline.py), with a fake SAM and fake Spark (no network)."""
import base64
import io
import json
import os
import unittest
from unittest import mock

from PIL import Image, ImageDraw

from tests.test_sam import fake_api, record
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


FLAGS = ["TOPO_SKIP_COVERED_POINT", "TOPO_PRUNE_PROMPT", "TOPO_PROMPT_MAX",
         "TOPO_AI_SIDE", "TOPO_JPEG_QUALITY", "TOPO_SAM_SIDE", "TOPO_SAM_QUALITY"]


def clean_env(**over):
    """The speedup flags cleared (defaults), plus overrides. Tests never inherit the shell env."""
    env = {k: v for k, v in os.environ.items() if k not in FLAGS}
    env.update(over)
    return mock.patch.dict(os.environ, env, clear=True)


def many_cands(n=60, group_size=8):
    """Synthetic holds: the tap's red group first, then four blue groups spread over the wall."""
    cands = []
    for i in range(n):
        red = i < group_size
        cands.append({"id": i, "x": 0.05 + 0.9 * (i % 10) / 10, "y": 0.05 + 0.9 * (i // 10) / 6,
                      "r": 0.03, "points": [], "conf": 0.9, "area": 0.001,
                      "hsv": (0.0, 0.8, 0.6) if red else (0.6, 0.8, 0.6),
                      "lab": (50.0, 40.0, 10.0), "color_name": "red" if red else "blue",
                      "group": 0 if red else 1 + (i - group_size) % 4, "near": []})
    return pipeline.add_neighbors(cands, W, H)


def image_bytes(part):
    return base64.b64decode(part["image_url"]["url"].split(",", 1)[1])


class SpeedupTest(unittest.TestCase):
    def covered_post(self):
        tap = {"x": 230 / W, "y": 225 / H}  # inside the first hold: the concepts cover it
        text = "<0f>" + ",".join(record(i, box) for i, (box, _) in enumerate(HOLDS))
        return tap, fake_api({"climbing hold": text, "climbing volume": text})

    def test_default_prepare_sends_three_calls(self):
        tap, (post, seen) = self.covered_post()
        with clean_env():
            _, cands, tap_id = agent.prepare(sam.Sam(post), jpeg_of(wall()), tap)
        self.assertEqual(len(seen), 3)  # two concepts and one point, as before
        self.assertEqual(tap_id, 0)

    def test_skip_point_when_tap_is_covered(self):
        tap, (post, seen) = self.covered_post()
        with clean_env():
            _, plain_cands, plain_tap = agent.prepare(sam.Sam(post), jpeg_of(wall()), tap)
        tap, (post, seen) = self.covered_post()
        with clean_env(TOPO_SKIP_COVERED_POINT="1"):
            stats = {}
            _, cands, tap_id = agent.prepare(sam.Sam(post), jpeg_of(wall()), tap, stats)
        self.assertEqual(len(seen), 2)  # the point prompt is skipped
        self.assertEqual(stats, {"sam_calls": 2})
        self.assertEqual(tap_id, plain_tap)
        self.assertEqual([(c["x"], c["y"]) for c in cands],
                         [(c["x"], c["y"]) for c in plain_cands])  # same holds either way

    def test_point_is_still_sent_when_tap_is_missed(self):
        tap = {"x": 415 / W, "y": 212 / H}  # the blue hold, missed by the concept prompts
        text = "<0f>" + record(0, HOLDS[0][0]) + "," + record(1, HOLDS[1][0])
        blue = "<0f>" + record(1, HOLDS[2][0])
        post, seen = fake_api({"climbing hold": text, "climbing volume": text,
                               sam.point_prompt(tap["x"], tap["y"], W, H): blue})
        with clean_env(TOPO_SKIP_COVERED_POINT="1"):
            stats = {}
            _, cands, tap_id = agent.prepare(sam.Sam(post), jpeg_of(wall()), tap, stats)
        self.assertEqual(len(seen), 3)  # uncovered: the point prompt still goes out
        self.assertEqual(stats, {"sam_calls": 3})
        self.assertEqual(tap_id, 2)
        self.assertEqual(agent.color_name(cands[2]["hsv"]), "blue")

    def test_find_with_reports_actual_sam_calls(self):
        jpeg = jpeg_of(wall())
        tap = {"x": 230 / W, "y": 225 / H}

        def run(**env):
            text = "<0f>" + ",".join(record(i, box) for i, (box, _) in enumerate(HOLDS))
            post, _ = fake_api({"climbing hold": text, "climbing volume": text})
            spark = RunOnceTest().route_reply()
            traces = []
            with clean_env(**env):
                holds = pipeline.find_with(sam.Sam(post), spark, jpeg, tap, on_trace=traces.append)
            return holds, traces[0][0]["args"]["sam_calls"]

        holds, default_calls = run()
        self.assertEqual(default_calls, 3)
        self.assertEqual(len(holds), 2)
        holds, skipped_calls = run(TOPO_SKIP_COVERED_POINT="1")
        self.assertEqual(skipped_calls, 2)
        self.assertEqual(len(holds), 2)  # same answer with one fewer SAM call

    def test_prompt_is_full_by_default(self):
        cands = many_cands()
        with clean_env():
            self.assertIsNone(pipeline.prompt_cap())
            text = pipeline.prompt_table(cands, 0)
        self.assertNotIn("Showing", text)
        rows = [line for line in text.splitlines()[1:] if " | " in line]
        self.assertEqual(len(rows), len(cands))  # one full row per hold

    def test_pruned_prompt_keeps_tap_group_and_counts(self):
        cands = many_cands()
        with clean_env(TOPO_PRUNE_PROMPT="1", TOPO_PROMPT_MAX="12"):
            cap = pipeline.prompt_cap()
            text = pipeline.prompt_table(cands, 0, cap)
        self.assertEqual(cap, 12)
        rows = [line for line in text.splitlines()[1:] if " | " in line]
        self.assertLess(len(rows), len(cands))
        self.assertIn("0*", text)  # the tap keeps its mark
        for i in range(8):  # the tap's whole color group keeps full rows
            self.assertTrue(any(line.startswith(f"{i} ") or line.startswith(f"{i}*")
                                for line in rows), f"group row {i} is shown")
        self.assertIn(f"Showing {len(rows)} of {len(cands)} holds", text)
        for c in cands:  # every id is still named in the group listing, so all are answerable
            self.assertIn(str(c["id"]), text.split("Color groups:")[1])
        self.assertEqual(pipeline.parse_answer(
            json.dumps({"hold_ids": [0, 59], "confidence": 0.8}),
            {c["id"] for c in cands})["error"], "")  # a pruned id still parses

    def test_prune_fills_with_nearest_first(self):
        cands = many_cands()
        kept = pipeline.select_prompt_cands(cands, 0, 10)
        kept_ids = {c["id"] for c in kept}
        self.assertTrue({c["id"] for c in cands if c["group"] == 0} <= kept_ids)
        rest = [c for c in cands if c["id"] not in kept_ids]
        tap = cands[0]
        farthest_kept = max((c["x"] - tap["x"]) ** 2 + (c["y"] - tap["y"]) ** 2
                            for c in kept if c["group"] != 0)
        nearest_dropped = min((c["x"] - tap["x"]) ** 2 + (c["y"] - tap["y"]) ** 2 for c in rest)
        self.assertLessEqual(farthest_kept, nearest_dropped)
        self.assertEqual([c["id"] for c in kept], sorted(kept_ids))  # id order is kept

    def test_pruned_prompt_reaches_spark(self):
        cands = many_cands(n=60)
        spark = RunOnceTest().route_reply(ids=[0])
        with clean_env(TOPO_PRUNE_PROMPT="1", TOPO_PROMPT_MAX="12"):
            ids, info = pipeline.run_once(spark, wall(), cands, 0)
        self.assertEqual(ids, [0])
        self.assertEqual(info["fallback"], "")
        self.assertIn("Showing", spark.seen[0][1]["content"][0]["text"])

    def test_spark_payload_defaults_are_unchanged(self):
        cands = pipeline.add_neighbors(pipeline.group_by_color(candidates()), W, H)
        with clean_env():
            self.assertEqual(agent.ai_side(), agent.AI_SIDE)
            self.assertEqual(agent.jpeg_quality(), 85)
            spark = RunOnceTest().route_reply()
            pipeline.run_once(spark, wall(), cands, 0)
            default_parts = spark.seen[0][1]["content"][1:]
        with Image.open(io.BytesIO(image_bytes(default_parts[0]))) as img:
            self.assertEqual(img.size, (W, H))  # 800x600 wall is shown as is
        with clean_env(TOPO_AI_SIDE="640", TOPO_JPEG_QUALITY="60"):
            spark = RunOnceTest().route_reply()
            pipeline.run_once(spark, wall(), cands, 0)
            small_parts = spark.seen[0][1]["content"][1:]
        with Image.open(io.BytesIO(image_bytes(small_parts[0]))) as img:
            self.assertEqual(img.size, (640, 480))
        default_len = sum(len(image_bytes(p)) for p in default_parts)
        small_len = sum(len(image_bytes(p)) for p in small_parts)
        self.assertLess(small_len, default_len)
        self.assertLess(small_len, default_len * 0.7)  # a real payload cut, not a rounding error

    def test_sam_shrink_knobs(self):
        big = Image.new("RGB", (2000, 1500), (90, 200, 200))
        out = io.BytesIO()
        big.save(out, "JPEG", quality=95)
        jpeg = out.getvalue()
        with clean_env():
            small, w, h = sam.shrink(jpeg)
            self.assertEqual(max(w, h), sam.SEND_SIDE)  # 1600, as before
            explicit, _, _ = sam.shrink(jpeg, 800)
        with clean_env(TOPO_SAM_SIDE="800", TOPO_SAM_QUALITY="70"):
            tiny, w, h = sam.shrink(jpeg)
            self.assertEqual(max(w, h), 800)
        self.assertLess(len(tiny), len(small))
        with Image.open(io.BytesIO(explicit)) as img:
            self.assertEqual(max(img.size), 800)  # an explicit side still wins


def rank_post(body):
    return {"output_text": "<0f>" + ",".join(record(i, box) for i, (box, _) in enumerate(HOLDS))}


RANK_TAP = {"x": 230 / W, "y": 225 / H}  # inside the first red hold


def red_cand(i, x, y):
    return {"id": i, "x": x, "y": y, "r": 0.03, "points": [], "conf": 0.9, "area": 0.001,
            "hsv": (0.0, 0.8, 0.6), "lab": (50.0, 40.0, 10.0), "color_name": "red"}


class RankTest(unittest.TestCase):
    def test_color_sim_basics(self):
        red, blue = candidates()[0]["hsv"], candidates()[2]["hsv"]
        self.assertEqual(pipeline.color_sim(red, red), 1.0)
        self.assertGreaterEqual(pipeline.color_sim(red, blue), 0.0)
        self.assertLess(pipeline.color_sim(red, blue), 0.5)  # red and blue are far apart

    def test_scores_follow_the_formula(self):
        cands = candidates()
        cos = {0: 1.0, 1: 0.5, 2: 0.0}
        scores = pipeline.rank_scores(cands, 0, cos)
        self.assertEqual(scores[0], 1.0)  # the tap's self-score
        for i in (1, 2):
            expected = pipeline.color_sim(cands[i]["hsv"], cands[0]["hsv"]) * (0.5 + 0.5 * cos[i])
            self.assertAlmostEqual(scores[i], expected)
        self.assertGreater(scores[1], scores[2])  # same color and closer embedding wins

    def test_scores_without_cosines_are_color_only(self):
        cands = candidates()
        scores = pipeline.rank_scores(cands, 0)
        for c in cands:
            self.assertAlmostEqual(scores[c["id"]], pipeline.color_sim(c["hsv"], cands[0]["hsv"]))

    def test_rank_cosines_to_tap_from_fake_vectors(self):
        cos = pipeline.rank_cosines(wall(), candidates(), 0, embedder=fake_embedder)
        self.assertEqual(cos[0], 1.0)
        self.assertGreater(cos[1], 0.99)  # the other red
        self.assertAlmostEqual(cos[2], 0.0)  # blue is orthogonal

    def test_rank_cosines_rejects_a_wrong_count(self):
        with self.assertRaises(ValueError):
            pipeline.rank_cosines(wall(), candidates(), 0, embedder=lambda crops: [[1.0]])

    def test_linking_chains_but_does_not_jump(self):
        tap, mid, far, lone = red_cand(0, 0.1, 0.5), red_cand(1, 0.3, 0.5), \
            red_cand(2, 0.5, 0.5), red_cand(3, 0.95, 0.5)
        cands = [tap, mid, far, lone]  # tap-mid-far chain in LINK steps; lone is isolated
        keep = {0, 1, 2, 3}
        self.assertEqual(pipeline.rank_linked(cands, 0, keep, W, H), {0, 1, 2})
        self.assertEqual(pipeline.rank_linked([tap, lone], 0, {0, 3}, W, H), {0})
        self.assertEqual(pipeline.rank_linked(cands, 0, set(), W, H), {0})  # the tap always qualifies

    def test_select_orders_scores_and_caps_holds(self):
        cands = [red_cand(0, 0.1, 0.5), red_cand(5, 0.12, 0.5), red_cand(3, 0.14, 0.5)]
        ids = pipeline.rank_select(cands, 0, W, H, cos_to_tap={0: 1.0, 5: 0.5, 3: 1.0})
        self.assertEqual(ids, [0, 3, 5])  # tap first, then by score, ties by id
        many = [red_cand(i, 0.05 + 0.01 * i, 0.5) for i in range(70)]  # one LINK-chained line
        capped = pipeline.rank_select(many, 0, W, H)
        self.assertEqual(len(capped), agent.MAX_HOLDS)
        self.assertEqual(capped[0], 0)
        self.assertEqual(capped, pipeline.rank_select(many, 0, W, H))  # deterministic

    def test_select_always_includes_the_tap(self):
        cands = candidates()
        self.assertEqual(pipeline.rank_select(cands, 0, W, H, tap_frac=2.0), [0])

    def run_rank(self, **kwargs):
        traces = []
        holds = pipeline.find_rank_with(sam.Sam(rank_post), jpeg_of(wall()), RANK_TAP,
                                        on_trace=traces.append, **kwargs)
        return holds, traces[0]

    def test_find_ranks_the_tapped_route(self):
        holds, trace = self.run_rank(embedder=fake_embedder)
        self.assertEqual(len(holds), 2)  # the two reds, not the blue
        self.assertEqual([s["tool"] for s in trace], ["(sam)", "(color)", "(embed)", "(rank)"])
        self.assertEqual(trace[0]["args"]["count"], 3)
        self.assertEqual(trace[2]["args"]["count"], 3)
        self.assertEqual(trace[3]["args"]["count"], 2)
        self.assertEqual(trace[3]["args"]["spark_calls"], 0)  # no model calls at all

    def test_find_is_deterministic(self):
        first, _ = self.run_rank(embedder=fake_embedder)
        second, _ = self.run_rank(embedder=fake_embedder)
        self.assertEqual(first, second)

    def test_find_without_torch_scores_on_color_alone(self):
        def no_torch(crops):
            raise ImportError("no torch")

        holds, trace = self.run_rank(embedder=no_torch)
        self.assertEqual(len(holds), 2)  # the reds still win on color
        self.assertEqual([s["tool"] for s in trace], ["(sam)", "(color)", "(rank)"])

    def test_find_without_embedder_uses_color_alone(self):
        with mock.patch.object(pipeline, "dino_embed", side_effect=ImportError("no torch")):
            holds, trace = self.run_rank()  # the default path, offline: no torch, no network
        self.assertEqual(len(holds), 2)
        self.assertEqual([s["tool"] for s in trace], ["(sam)", "(color)", "(rank)"])

    def test_find_with_embed_off_leaves_the_embedder_untouched(self):
        def boom(crops):
            raise AssertionError("must not be called")

        holds, trace = self.run_rank(embedder=boom, use_embed=False)
        self.assertEqual(len(holds), 2)
        self.assertEqual([s["tool"] for s in trace], ["(sam)", "(color)", "(rank)"])

    def test_find_embed_failure_drops_the_step(self):
        def broken(crops):
            raise RuntimeError("cuda busy")

        holds, trace = self.run_rank(embedder=broken)
        self.assertEqual(len(holds), 2)
        self.assertEqual([s["tool"] for s in trace], ["(sam)", "(color)", "(rank)"])

    def test_find_strict_fraction_keeps_only_the_tap(self):
        holds, trace = self.run_rank(embedder=fake_embedder, tap_frac=1.0)
        self.assertEqual(len(holds), 1)
        self.assertAlmostEqual(holds[0]["x"], 230 / W, delta=0.05)

    def test_find_survives_sam_failure(self):
        def broken(body):
            raise RuntimeError("503")

        traces = []
        holds = pipeline.find_rank_with(sam.Sam(broken), jpeg_of(wall()), RANK_TAP,
                                        on_trace=traces.append)
        self.assertTrue(holds)  # color only still finds the red problem
        self.assertEqual(traces[0][-1]["tool"], "(fallback: color)")

    def test_rank_finder_needs_a_sam_key(self):
        self.assertIsNone(pipeline.rank_finder({}))

    def test_rank_finder_never_touches_the_chat_api(self):
        def boom(env):
            raise AssertionError("the ranker makes no Spark calls")

        with mock.patch.object(sam, "http_post", lambda env: rank_post), \
                mock.patch.object(agent, "http_chat", boom):
            find = pipeline.rank_finder({"MODEL_API_KEY": "x"}, embedder=fake_embedder)
            holds = find(jpeg_of(wall()), RANK_TAP)
        self.assertEqual(len(holds), 2)

    def test_rank_finder_honors_env_knobs(self):
        jpeg = jpeg_of(wall())
        with mock.patch.object(sam, "http_post", lambda env: rank_post):
            strict = pipeline.rank_finder({"MODEL_API_KEY": "x", "RANK_TAP_FRAC": "1.0"},
                                          embedder=fake_embedder)(jpeg, RANK_TAP)
            self.assertEqual(len(strict), 1)
            bogus = pipeline.rank_finder({"MODEL_API_KEY": "x", "RANK_TAP_FRAC": "bogus"},
                                         embedder=fake_embedder)(jpeg, RANK_TAP)
            self.assertEqual(len(bogus), 2)  # a bad fraction falls back to the default
            traces = []
            color_only = pipeline.rank_finder({"MODEL_API_KEY": "x", "RANK_EMBED": "0"},
                                              on_trace=traces.append,
                                              embedder=fake_embedder)(jpeg, RANK_TAP)
            self.assertEqual(len(color_only), 2)
            self.assertNotIn("(embed)", [s["tool"] for s in traces[0]])

    def test_rank_is_registered_in_eval(self):
        from topo_agent import eval as hold_eval
        with mock.patch.object(sam, "http_post", lambda env: rank_post):
            self.assertIn("rank", hold_eval.finders({"MODEL_API_KEY": "x"}, ["rank"]))
        self.assertEqual(hold_eval.finders({}, ["rank"]), {})  # skipped without a key
