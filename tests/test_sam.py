# SPDX-License-Identifier: AGPL-3.0-or-later
"""Calling SAM 3.1 and reading its output (topo_agent/sam.py), with a fake Meta Model API (no network)."""
import io
import runpy
import sys
import unittest
import warnings
from unittest import mock

from meta_sam_parser._mask_codec import _encode_raster
from PIL import Image, ImageDraw

from topo_agent import eval as hold_eval
from topo_agent import sam
from topo_agent.baseline import bench as hold_bench

W, H = 800, 600
HOLDS = [(200, 200, 260, 250), (330, 260, 360, 285), (400, 200, 430, 225)]  # ellipse boxes (pixels)


def wall_jpeg():
    img = Image.new("RGB", (W, H), (90, 200, 200))
    d = ImageDraw.Draw(img)
    for box in HOLDS:
        d.ellipse(box, fill=(150, 60, 170))
    out = io.BytesIO()
    img.save(out, "JPEG")
    return out.getvalue()


def record(i, box, w=W, h=H, conf=0.9):
    """One object in the real format (a box-local mask in Meta's encoding)."""
    x1, y1, x2, y2 = box
    mw, mh = x2 - x1 + 1, y2 - y1 + 1
    m = Image.new("1", (mw, mh), 0)
    ImageDraw.Draw(m).ellipse((0, 0, mw - 1, mh - 1), fill=1)
    bits = [1 if m.getpixel((x, y)) else 0 for y in range(mh) for x in range(mw)]
    payload = _encode_raster(bits, mw, mh)
    return (f"{i}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h};c={conf}|>"
            f"<|mask;x=0;y=0;c={conf};data={mh},{mw},{payload}|>")


def fake_api(lines_by_prompt):
    """A fake API answering per prompt. Keeps the request bodies."""
    seen = []

    def post(body):
        seen.append(body)
        prompt = body["input"][0]["content"][0]["text"]
        text = lines_by_prompt.get(prompt, "")
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
    return post, seen


class SamTest(unittest.TestCase):
    def test_request_body(self):
        body = sam.request_body(b"\xff\xd8jpeg", "climbing hold")
        self.assertEqual(body["model"], "sam-3.1")
        content = body["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "climbing hold"})
        self.assertTrue(content[1]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(body["metadata"]["include_confidence"], "true")  # a string, as Meta requires
        self.assertIs(body["store"], False)  # not kept at Meta
        self.assertEqual(sam.point_prompt(0.5, 0.25, 1600, 1200), "<0f>1<|point;x=800;y=300;w=1600;h=1200|>")

    def test_parse_and_candidates(self):
        text = "<0f>" + ",".join(record(i, b) for i, b in enumerate(HOLDS))
        objs = sam.parse(text)
        self.assertEqual([o["id"] for o in objs], [0, 1, 2])
        self.assertEqual(objs[1]["box"], (330, 260, 360, 285))
        self.assertAlmostEqual(objs[0]["conf"], 0.9)
        c = sam.to_candidate(objs[0])
        self.assertAlmostEqual(c["x"], 230 / W, delta=0.003)  # mask center as a fraction of the photo
        self.assertAlmostEqual(c["y"], 225 / H, delta=0.003)
        self.assertAlmostEqual(c["r"], 30 / W, delta=0.002)
        self.assertGreaterEqual(len(c["points"]), 6)
        self.assertLessEqual(len(c["points"]), sam.MAX_POINTS)
        self.assertTrue(all(200 / W - 0.01 <= x <= 260 / W + 0.01 for x, _ in c["points"]))  # outline stays in the box
        self.assertEqual(sam.parse(""), [])  # no match is a valid answer

    def test_candidates_merge_concepts(self):
        hold = "<0f>" + ",".join(record(i, b) for i, b in enumerate(HOLDS[:2]))
        volume = "<0f>" + record(5, HOLDS[1], conf=0.5) + "," + record(7, HOLDS[2])  # one duplicates a hold
        post, seen = fake_api({"climbing hold": hold, "climbing volume": volume})
        jpeg = wall_jpeg()
        found = sam.Sam(post).candidates(jpeg)
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(found), 3)
        self.assertEqual([c["id"] for c in found], [0, 1, 2])  # renumbered

    def test_point_prompt_returns_one_shape(self):
        post, seen = fake_api({sam.point_prompt(0.2875, 0.375, W, H): "<0f>" + record(0, HOLDS[0])})
        c = sam.Sam(post).at(wall_jpeg(), {"x": 0.2875, "y": 0.375})
        self.assertAlmostEqual(c["x"], 230 / W, delta=0.003)
        self.assertIsNone(sam.Sam(fake_api({})[0]).at(wall_jpeg(), {"x": 0.9, "y": 0.9}))

    def test_no_key_no_client(self):
        self.assertIsNone(sam.http_post({}))

    def test_shrink_limits_long_side(self):
        big = Image.new("RGB", (3000, 2000), (10, 20, 30))
        buf = io.BytesIO()
        big.save(buf, "JPEG")
        small, w, h = sam.shrink(buf.getvalue())
        self.assertEqual(max(w, h), sam.SEND_SIDE)
        self.assertAlmostEqual(w / h, 1.5, delta=0.01)  # aspect kept
        with Image.open(io.BytesIO(small)) as img:
            self.assertEqual(img.size, (w, h))  # valid JPEG of the reported size
        small, w, h = sam.shrink(wall_jpeg())  # small photos are left alone
        self.assertEqual((w, h), (W, H))

    def test_output_text_shapes(self):
        self.assertEqual(sam.output_text({"output_text": "<0f>"}), "<0f>")
        resp = {"output": [{"content": [{"type": "output_text", "text": "a"}, {"type": "other", "text": "x"}]},
                           {"content": [{"type": "output_text", "text": "b"}]}]}
        self.assertEqual(sam.output_text(resp), "ab")
        self.assertEqual(sam.output_text({}), "")

    def test_parse_skips_junk_and_reads_floats(self):
        no_conf = record(3, HOLDS[0]).replace(";c=0.9", "")
        floats = record(4, HOLDS[1]).replace("x1=330", "x1=330.0").replace("y2=285", "y2=285.0")
        objs = sam.parse("junk line\n<0f>" + no_conf + "\n<1f>" + floats + "\n")
        self.assertEqual([o["id"] for o in objs], [3, 4])
        self.assertEqual([o["frame"] for o in objs], [0, 1])
        self.assertIsNone(objs[0]["conf"])  # missing confidence stays None
        self.assertEqual(objs[1]["box"], (330, 260, 360, 285))  # floats become ints

    def test_empty_mask_is_no_candidate(self):
        x1, y1, x2, y2 = HOLDS[0]
        mw, mh = x2 - x1 + 1, y2 - y1 + 1
        payload = _encode_raster([0] * (mw * mh), mw, mh)
        text = (f"<0f>0<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={W};h={H};c=0.9|>"
                f"<|mask;x=0;y=0;c=0.9;data={mh},{mw},{payload}|>")
        self.assertIsNone(sam.to_candidate(sam.parse(text)[0]))
        self.assertEqual(sam.decode((mh, mw, payload))[0][0], 0)

    def test_overlap_and_merge(self):
        self.assertAlmostEqual(sam._overlap([0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2]), 1.0)
        self.assertAlmostEqual(sam._overlap([0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]), 0.0)
        a = {"id": 5, "box": [0.1, 0.1, 0.2, 0.2], "conf": 0.9}
        b = {"id": 6, "box": [0.11, 0.11, 0.19, 0.19], "conf": 0.5}  # inside a: a duplicate
        c = {"id": 7, "box": [0.5, 0.5, 0.6, 0.6], "conf": 0.8}
        out = sam.merge([[a], [b, c]])
        self.assertEqual([(x["conf"], x["id"]) for x in out], [(0.9, 0), (0.8, 1)])

    def test_segment_reads_output_text_shape(self):
        text = "<0f>" + ",".join(record(i, b) for i, b in enumerate(HOLDS))
        found = sam.Sam(lambda body: {"output_text": text}).segment(wall_jpeg(), "climbing hold")
        self.assertEqual(len(found), 3)

    def test_http_post_auth_and_error(self):
        post = sam.http_post({"MODEL_API_KEY": "k"})
        self.assertIsNotNone(post)
        with mock.patch("httpx.post") as m:
            m.return_value = mock.Mock(status_code=503, text="overloaded")
            with self.assertRaises(RuntimeError) as ctx:
                post({"model": "sam-3.1"})
            self.assertIn("SAM 503", str(ctx.exception))
        with mock.patch("httpx.post") as m:
            m.return_value = mock.Mock(status_code=200, text="{}")
            m.return_value.json.return_value = {"output_text": ""}
            self.assertEqual(post({"model": "x"}), {"output_text": ""})
            args, kw = m.call_args
            self.assertEqual(kw["headers"]["Authorization"], "Bearer k")
            self.assertTrue(args[0].endswith("/responses"))

    def test_cli_usage_errors(self):
        def run_cli(argv):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # runpy re-executes an imported module
                with mock.patch.object(sys, "argv", argv):
                    runpy.run_module("topo_agent.sam", run_name="__main__")

        with self.assertRaises(SystemExit) as ctx:
            run_cli(["sam"])
        self.assertIn("usage", str(ctx.exception))
        with mock.patch.dict("os.environ", clear=True):
            with self.assertRaises(SystemExit) as ctx:
                run_cli(["sam", "photo.jpg"])
        self.assertIn("MODEL_API_KEY", str(ctx.exception))

    def test_candidates_on_bench_scene(self):
        # A scripted SAM returning the marked holds: the whole path (shrink, parse, decode, merge)
        # must find every hold of the problem
        s = hold_bench.scene("similar", seed=11)
        with Image.open(io.BytesIO(s["jpeg"])) as img:
            Wb, Hb = img.size
        recs = []
        for i, t in enumerate(s["truth"]):
            xs = [p[0] * Wb for p in t["points"]]
            ys = [p[1] * Hb for p in t["points"]]
            recs.append(record(i, (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), w=Wb, h=Hb))
        text = "<0f>" + ",".join(recs)
        post, seen = fake_api({"climbing hold": text, "climbing volume": text})
        found = sam.Sam(post).candidates(s["jpeg"])
        self.assertEqual(len(seen), 2)  # one request per concept
        self.assertEqual(len(found), 7)  # duplicates merged
        self.assertEqual([c["id"] for c in found], list(range(7)))
        for c in found:
            self.assertTrue(0 <= c["x"] <= 1 and 0 <= c["y"] <= 1 and 0 < c["r"] <= 1)
        marks = [{"x": t["x"], "y": t["y"], "r": t["r"], "points": t["points"]} for t in s["truth"]]
        self.assertEqual(hold_eval.match(marks, found, Wb, Hb), (7, 7, 7))


if __name__ == "__main__":
    unittest.main()
