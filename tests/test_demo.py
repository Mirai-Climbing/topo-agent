# SPDX-License-Identifier: AGPL-3.0-or-later
"""The demo command (topo_agent/demo.py): recorded runs replay without a key or network."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from topo_agent import demo
from topo_agent.eval import match


def run(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        demo.main(list(args), env={})
    return out.getvalue()


class DemoTest(unittest.TestCase):
    def test_recorded_runs_replay_without_a_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            for scene in demo.SCENES:
                png = Path(tmp) / f"{scene}.png"
                text = run("--scene", scene, "--out", str(png))
                self.assertIn("RECORDED run", text)
                self.assertIn("What the AI did:", text)
                self.assertIn("submit_holds", text)
                self.assertIn("Against the answer:", text)
                with Image.open(png) as img:
                    self.assertEqual(img.size, (demo.bench.WIDTH, demo.bench.HEIGHT))

    def test_live_needs_a_key(self):
        with self.assertRaises(SystemExit) as stop:
            run("--live", "--scene", "tape", "--out", str(Path(tempfile.gettempdir()) / "x.png"))
        self.assertIn("MODEL_API_KEY", str(stop.exception))

    def test_photo_needs_live_and_tap(self):
        with self.assertRaises(SystemExit):
            run("--photo", "wall.jpg")

    def test_recordings_still_match_the_sample_walls(self):
        """If a drawn wall changes (bench.scene), its recorded run no longer fits: record it again with --live --record."""
        for scene in demo.SCENES:
            run_ = json.loads((demo.RUNS / f"{scene}.json").read_text(encoding="utf-8"))
            self.assertEqual((run_["scene"], run_["seed"]), (scene, demo.SEED))
            _, _, truth = demo.sample(scene)
            hits, found, total = match(truth, run_["holds"], demo.bench.WIDTH, demo.bench.HEIGHT)
            self.assertEqual((hits, found), (total, total), f"recorded run for {scene!r} no longer fits the wall")

    def test_bad_tap_or_photo_explains_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            wall = Path(tmp) / "wall.jpg"
            Image.new("RGB", (40, 30)).save(wall)
            for tap in ("garbage", "0.5", "1.5,0.2"):
                with self.assertRaises(SystemExit) as stop:
                    run("--live", "--photo", str(wall), "--tap", tap)
                self.assertIn("--tap", str(stop.exception))
            with self.assertRaises(SystemExit) as stop:
                run("--live", "--photo", str(Path(tmp) / "missing.jpg"), "--tap", "0.5,0.5")
            self.assertIn("Can't read the photo", str(stop.exception))

    def test_describe_steps(self):
        self.assertEqual(demo.describe({"tool": "(sam)", "args": {"count": 29}, "ms": 3400}),
                         "SAM 3.1 outlined 29 hold candidates (3.4 s)")
        self.assertIn("instead of a tool call", demo.describe({"tool": "(model)", "turn": 1, "text": "<x>"}))
        self.assertEqual(demo.describe({"tool": "look_closer", "args": {"ids": [1, 2]}, "ms": 0}),
                         "look_closer [1, 2] (0.0 s)")


if __name__ == "__main__":
    unittest.main()
