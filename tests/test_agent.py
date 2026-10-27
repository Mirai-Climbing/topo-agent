# SPDX-License-Identifier: AGPL-3.0-or-later
"""トポエージェント(topo_agent/agent.py)を、偽の SAM と偽の Muse Spark で確かめる(Meta にはつながない)。"""
import io
import unittest

from PIL import Image, ImageDraw

from tests.test_sam import fake_api, record
from topo_agent import agent, sam

W, H = 800, 600
PURPLE, BLUE = (150, 60, 170), (40, 70, 200)
HOLDS = [((200, 200, 260, 250), PURPLE), ((330, 260, 360, 285), PURPLE), ((400, 200, 430, 225), BLUE)]


def wall():
    img = Image.new("RGB", (W, H), (214, 206, 192))
    d = ImageDraw.Draw(img)
    for box, color in HOLDS:
        d.ellipse(box, fill=color)
    return img


def jpeg_of(img):
    out = io.BytesIO()
    img.save(out, "JPEG", quality=95)
    return out.getvalue()


def candidates():
    objs = sam.parse("<0f>" + ",".join(record(i, box) for i, (box, _) in enumerate(HOLDS)))
    return agent.measure(wall(), [sam.to_candidate(o) for o in objs])


def scripted(*replies):
    """決まった順に道具を呼ぶ偽のモデル。受け取ったメッセージも残す。"""
    seen = []

    def call(messages):
        seen.append(list(messages))
        calls = replies[min(len(seen) - 1, len(replies) - 1)]
        calls = [{"id": f"c{len(seen)}_{n}", **c} for n, c in enumerate(calls)]
        return {"calls": calls, "text": "", "raw": {"role": "assistant", "content": "", "tool_calls": calls}}
    return call, seen


class AgentTest(unittest.TestCase):
    def test_colors_and_sam_color(self):
        cands = candidates()
        self.assertEqual([agent.color_name(c["hsv"]) for c in cands], ["purple", "purple", "blue"])
        self.assertLess(agent.color_distance(cands[0]["hsv"], cands[1]["hsv"]), agent.SIMILAR)
        self.assertGreater(agent.color_distance(cands[0]["hsv"], cands[2]["hsv"]), agent.SIMILAR)
        self.assertEqual([c["id"] for c in agent.sam_color(cands, 0)], [0, 1])  # 色だけなら紫の2つ
        self.assertEqual(agent.tapped_candidate(cands, {"x": 230 / W, "y": 225 / H}, W, H)["id"], 0)
        self.assertIsNone(agent.tapped_candidate(cands, {"x": 0.9, "y": 0.9}, W, H))

    def test_agent_uses_tools_then_submits(self):
        model, seen = scripted(
            [{"name": "inspect_candidates", "args": {"ids": [0]}}, {"name": "find_similar", "args": {"id": 0}}],
            [{"name": "look_closer", "args": {"ids": [1, 2]}}],
            [{"name": "submit_holds", "args": {"ids": [1, 1, 99, "x"], "note": "紫の2つ"}}])
        ids, trace = agent.run_agent(model, wall(), candidates(), 0)
        self.assertEqual(ids, [0, 1])  # タップした候補はいつも入る。知らない番号・重なりは捨てる
        self.assertEqual([t["tool"] for t in trace if t["tool"] != "(model)"],
                         ["inspect_candidates", "find_similar", "look_closer", "submit_holds"])
        self.assertTrue(all("ms" in t for t in trace))
        first = seen[0][1]["content"]  # 写真と、番号つきの候補の絵の2枚
        self.assertEqual([p["type"] for p in first], ["text", "image_url", "image_url"])
        self.assertIn("candidate 0", first[0]["text"])
        tool_msgs = [m for m in seen[1] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["c1_0", "c1_1"])  # 道具の結果は、呼んだ番号に返す
        self.assertIn('"purple"', tool_msgs[0]["content"])
        self.assertIn('"id": 1', tool_msgs[1]["content"])  # 似た色の候補に、もう1つの紫
        self.assertEqual(seen[2][-1]["role"], "user")  # 拡大した絵は、道具の結果のあとに絵として渡す
        self.assertEqual(seen[2][-1]["content"][1]["type"], "image_url")

    def test_turn_limit_and_call_limit(self):
        many = [{"name": "inspect_candidates", "args": {"ids": [0]}}] * (agent.MAX_CALLS + 2)
        model, seen = scripted(many)
        ids, trace = agent.run_agent(model, wall(), candidates(), 0, turns=3)
        self.assertIsNone(ids)
        self.assertEqual(len(seen), 3)
        self.assertEqual(trace[-1]["tool"], "(上限で終了)")
        self.assertIn("at most", seen[1][-1]["content"])  # 1回に使える道具の数を超えた分は断る
        self.assertIn("submit_holds now", seen[2][-1]["content"])  # 最後の前に、答えるよう伝える

    def test_unknown_tool_and_no_answer(self):
        model, _ = scripted([{"name": "rm_rf", "args": {}}], [])
        ids, trace = agent.run_agent(model, wall(), candidates(), 0)
        self.assertIsNone(ids)
        self.assertIn("unknown tool", trace[1]["result"])

    def test_prepare_adds_the_tapped_hold_when_sam_missed_it(self):
        img = wall()
        jpeg = jpeg_of(img)
        tap = {"x": 0.52, "y": 0.355}  # 青のホールド(言葉の頼みでは見つからなかったことにする)
        post, seen = fake_api({"climbing hold": "<0f>" + record(0, HOLDS[0][0]) + "," + record(1, HOLDS[1][0]),
                               sam.point_prompt(tap["x"], tap["y"], W, H): "<0f>" + record(1, HOLDS[2][0])})
        _, cands, tap_id = agent.prepare(sam.Sam(post), jpeg, tap)
        self.assertEqual(len(seen), 3)  # 言葉2つと、点1つ
        self.assertEqual(tap_id, 2)
        self.assertEqual(agent.color_name(cands[2]["hsv"]), "blue")

    def test_chat_call_reads_tool_calls(self):
        reply = {"choices": [{"message": {"content": "look", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "find_similar", "arguments": '{"id": 3}'}},
            {"id": "call_2", "type": "function", "function": {"name": "look_closer", "arguments": "{broken"}}]}}]}
        sent = []
        call = agent.chat_call(lambda body: sent.append(body) or reply)
        out = call([{"role": "user", "content": "hi"}])
        self.assertEqual(out["calls"], [{"id": "call_1", "name": "find_similar", "args": {"id": 3}},
                                        {"id": "call_2", "name": "look_closer", "args": {}}])
        self.assertEqual(out["raw"]["tool_calls"][0]["id"], "call_1")
        self.assertIs(sent[0]["store"], False)
        self.assertEqual({t["function"]["name"] for t in sent[0]["tools"]},
                         {"inspect_candidates", "find_similar", "look_closer", "submit_holds"})

    def test_off_without_key(self):
        self.assertIsNone(agent.meta_hold_finder({}))
        self.assertIsNone(agent.sam_color_finder({}))


if __name__ == "__main__":
    unittest.main()
