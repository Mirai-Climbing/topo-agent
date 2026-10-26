# SPDX-License-Identifier: AGPL-3.0-or-later
"""SAM 3.1 の呼び方と答えの読み方(topo_agent/sam.py)を、偽の Meta Model API で確かめる(本物にはつながない)。"""
import io
import unittest

from meta_sam_parser._mask_codec import _encode_raster
from PIL import Image, ImageDraw

from topo_agent import sam

W, H = 800, 600
HOLDS = [(200, 200, 260, 250), (330, 260, 360, 285), (400, 200, 430, 225)]  # 楕円の箱(画素)


def wall_jpeg():
    img = Image.new("RGB", (W, H), (90, 200, 200))
    d = ImageDraw.Draw(img)
    for box in HOLDS:
        d.ellipse(box, fill=(150, 60, 170))
    out = io.BytesIO()
    img.save(out, "JPEG")
    return out.getvalue()


def record(i, box, w=W, h=H, conf=0.9):
    """本物と同じ形の1つの物(箱の中だけの形を、Meta の符号で書く)。"""
    x1, y1, x2, y2 = box
    mw, mh = x2 - x1 + 1, y2 - y1 + 1
    m = Image.new("1", (mw, mh), 0)
    ImageDraw.Draw(m).ellipse((0, 0, mw - 1, mh - 1), fill=1)
    bits = [1 if m.getpixel((x, y)) else 0 for y in range(mh) for x in range(mw)]
    payload = _encode_raster(bits, mw, mh)
    return (f"{i}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h};c={conf}|>"
            f"<|mask;x=0;y=0;c={conf};data={mh},{mw},{payload}|>")


def fake_api(lines_by_prompt):
    """prompt(言葉)ごとに決まった答えを返す偽の API。受け取った中身も残す。"""
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
        self.assertEqual(body["metadata"]["include_confidence"], "true")  # 文字列で送る(Meta の決まり)
        self.assertIs(body["store"], False)  # Meta 側に残さない
        self.assertEqual(sam.point_prompt(0.5, 0.25, 1600, 1200), "<0f>1<|point;x=800;y=300;w=1600;h=1200|>")

    def test_parse_and_candidates(self):
        text = "<0f>" + ",".join(record(i, b) for i, b in enumerate(HOLDS))
        objs = sam.parse(text)
        self.assertEqual([o["id"] for o in objs], [0, 1, 2])
        self.assertEqual(objs[1]["box"], (330, 260, 360, 285))
        self.assertAlmostEqual(objs[0]["conf"], 0.9)
        c = sam.to_candidate(objs[0])
        self.assertAlmostEqual(c["x"], 230 / W, delta=0.003)  # 形の真ん中を、写真の割合で
        self.assertAlmostEqual(c["y"], 225 / H, delta=0.003)
        self.assertAlmostEqual(c["r"], 30 / W, delta=0.002)
        self.assertGreaterEqual(len(c["points"]), 6)
        self.assertLessEqual(len(c["points"]), sam.MAX_POINTS)
        self.assertTrue(all(200 / W - 0.01 <= x <= 260 / W + 0.01 for x, _ in c["points"]))  # 輪郭は箱の中
        self.assertEqual(sam.parse(""), [])  # 1つもないのも正しい答え

    def test_candidates_merge_concepts(self):
        hold = "<0f>" + ",".join(record(i, b) for i, b in enumerate(HOLDS[:2]))
        volume = "<0f>" + record(5, HOLDS[1], conf=0.5) + "," + record(7, HOLDS[2])  # 1つは同じ物の重なり
        post, seen = fake_api({"climbing hold": hold, "climbing volume": volume})
        jpeg = wall_jpeg()
        found = sam.Sam(post).candidates(jpeg)
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(found), 3)
        self.assertEqual([c["id"] for c in found], [0, 1, 2])  # 番号は付け直す

    def test_point_prompt_returns_one_shape(self):
        post, seen = fake_api({sam.point_prompt(0.2875, 0.375, W, H): "<0f>" + record(0, HOLDS[0])})
        c = sam.Sam(post).at(wall_jpeg(), {"x": 0.2875, "y": 0.375})
        self.assertAlmostEqual(c["x"], 230 / W, delta=0.003)
        self.assertIsNone(sam.Sam(fake_api({})[0]).at(wall_jpeg(), {"x": 0.9, "y": 0.9}))

    def test_no_key_no_client(self):
        self.assertIsNone(sam.http_post({}))


if __name__ == "__main__":
    unittest.main()
