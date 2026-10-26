# SPDX-License-Identifier: AGPL-3.0-or-later
"""ホールドを探すエージェント(topo_agent/baseline/gemini_agent.py)を、偽のモデルで確かめる(Gemini にはつながない)。"""
import unittest

from PIL import Image, ImageDraw

from topo_agent.baseline import gemini_agent as hold_agent


def wall():
    img = Image.new("RGB", (800, 600), (90, 200, 200))  # 青緑の壁
    d = ImageDraw.Draw(img)
    d.ellipse((200, 200, 260, 250), fill=(150, 60, 170))  # 紫(押した)
    d.ellipse((330, 260, 360, 285), fill=(150, 60, 170))  # 紫(同じ課題)
    d.ellipse((400, 200, 430, 225), fill=(40, 70, 200))  # 青(ほかの課題)
    return img


def scripted(*replies):
    """決まった順に答える偽のモデル。受け取った履歴も残す。"""
    seen = []

    def call(history):
        seen.append(list(history))
        calls = replies[min(len(seen) - 1, len(replies) - 1)]
        return {"calls": calls, "text": "", "raw": {"calls": calls}}
    return call, seen


class HoldAgentTest(unittest.TestCase):
    def test_agent_uses_tools_then_submits(self):
        tap = {"x": 0.2875, "y": 0.375}
        model, seen = scripted(
            [{"name": "inspect_hold", "args": {"x": 287, "y": 375}}],
            [{"name": "find_same_color", "args": {"x": 287, "y": 375}}],
            [{"name": "submit_holds", "args": {"holds": [{"x": 287, "y": 375, "size": 40}, {"x": 431, "y": 455},
                                                         {"x": 2000, "y": 10}], "note": "紫の2つ"}}])
        holds, trace = hold_agent.run_agent(model, wall(), tap)
        self.assertEqual([t["tool"] for t in trace], ["inspect_hold", "find_same_color", "submit_holds"])
        self.assertEqual(len(holds), 2)  # 写真の外の座標は捨てる
        self.assertAlmostEqual(holds[1]["x"], 0.431)
        # 道具の結果は、次のモデルの呼び出しに渡る
        inspected = seen[1][-1]["tool_results"][0]["response"]
        self.assertEqual(inspected["color"], "紫")
        self.assertTrue(inspected["outline_found"])
        found = seen[2][-1]["tool_results"][0]["response"]["candidates"]
        self.assertEqual(len(found), 1)  # 同じ色の紫だけ(青は入らない)
        self.assertAlmostEqual(found[0]["x"], 431, delta=15)
        self.assertTrue(all("ms" in t for t in trace))  # かかった時間を記録する

    def test_step_limit_stops_the_agent(self):
        # 道具を使い続けるモデルは、上限で止め、それまでに分かった物で答える
        model, seen = scripted([{"name": "find_same_color", "args": {"x": 287, "y": 375}}])
        holds, trace = hold_agent.run_agent(model, wall(), {"x": 0.2875, "y": 0.375}, steps=3)
        ran = [t for t in trace if t["tool"] == "find_same_color" and "上限" not in t["result"]]
        self.assertEqual(len(ran), 3)  # 上限を超えた分は、道具を動かさずに断る
        self.assertEqual(trace[-1]["tool"], "(上限で終了)")
        self.assertEqual(len(holds), 2)  # 押した所 + 色で見つけた1つ
        self.assertLessEqual(len(seen), 5)

    def test_unknown_tool_and_no_answer(self):
        model, _ = scripted([{"name": "delete_everything", "args": {}}], [])
        holds, trace = hold_agent.run_agent(model, wall(), {"x": 0.2875, "y": 0.375})
        self.assertIn("知らない道具", trace[0]["result"])
        self.assertEqual(trace[1]["tool"], "(答えなし)")

    def test_clean_answer_limits(self):
        many = [{"x": 10, "y": 10}] * (hold_agent.MAX_HOLDS + 5) + [{"x": "a", "y": 1}, "x"]
        self.assertEqual(len(hold_agent.clean_answer(many)), hold_agent.MAX_HOLDS)
        self.assertEqual(hold_agent.clean_answer(None), [])

    def test_color_only_when_gemini_fails(self):
        # Gemini が答えないときの代わり:押したホールドと、同じ色のホールドだけ(青は入れない)
        holds = hold_agent.color_only(wall(), {"x": 230 / 800, "y": 225 / 600})
        self.assertEqual(len(holds), 2)
        self.assertTrue(all(h["x"] < 400 / 800 for h in holds))
        self.assertEqual(hold_agent.color_only(wall(), None), [])

    def test_off_without_settings(self):
        self.assertIsNone(hold_agent.gemini_hold_finder({}))
        self.assertIsNone(hold_agent.gemini_page_checker({}))

    def test_check_finds_a_model_that_works(self):
        class Models:
            def generate_content(self, model, contents):
                if model != "gemini-2.5-flash":
                    raise RuntimeError("404 model not found")

        fake = type("Client", (), {"models": Models()})()
        real = hold_agent.gemini_client
        hold_agent.gemini_client = lambda env: fake
        try:
            found, tried = hold_agent.check({})
            self.assertEqual(found, "gemini-2.5-flash")
            self.assertEqual(len(tried), 3)  # 使えなかったモデルも、理由つきで出す
            self.assertEqual(hold_agent.check({"MIRAI_GEMINI_MODEL": "gemini-x"})[0], None)  # 決めたモデルだけ試す
        finally:
            hold_agent.gemini_client = real
        self.assertEqual(hold_agent.check({})[0], None)  # 設定がなければ使えない

    def test_color_names(self):
        self.assertEqual(hold_agent.color_name((0.75, 0.6, 0.6)), "紫")
        self.assertEqual(hold_agent.color_name((0.0, 0.05, 0.95)), "白")
        self.assertEqual(hold_agent.color_name((0.0, 0.05, 0.1)), "黒")


if __name__ == "__main__":
    unittest.main()
