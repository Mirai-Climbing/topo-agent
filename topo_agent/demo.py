# SPDX-License-Identifier: AGPL-3.0-or-later
"""Try the topo agent on one wall: tap one hold, get the whole problem.

Without a key, it replays a recorded run on a drawn sample wall (clearly marked as a recording):

    python -m topo_agent.demo                    # tape problem (recorded)
    python -m topo_agent.demo --scene twins      # same color on the next wall (recorded)

With MODEL_API_KEY set, it calls SAM 3.1 and Muse Spark for real:

    python -m topo_agent.demo --live --scene tape
    python -m topo_agent.demo --live --photo wall.jpg --tap 0.42,0.63   # your own photo; tap as 0-1 fractions

Each run prints what the agent did and writes an image with the problem outlined (demo-result.png).
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from topo_agent.baseline import bench
from topo_agent.eval import match

RUNS = Path(__file__).resolve().parent / "demo_runs"  # recorded live runs on the sample walls
SCENES = ("tape", "twins", "light")
SEED = 11
OUTLINE = (255, 40, 190)


def sample(name):
    """A drawn sample wall: (JPEG, tap point, truth holds with centers)."""
    s = bench.scene(name, SEED)
    truth = []
    for t in s["truth"]:
        pts = t["points"]
        truth.append({"points": pts, "x": sum(p[0] for p in pts) / len(pts), "y": sum(p[1] for p in pts) / len(pts)})
    return s["jpeg"], s["tap"], truth


def describe(step):
    """One line of "What the AI did"."""
    tool, args = step.get("tool", ""), step.get("args") or {}
    ms = f" ({step['ms'] / 1000:.1f} s)" if step.get("ms") is not None else ""
    if tool == "(sam)":
        return f"SAM 3.1 outlined {args.get('count')} hold candidates{ms}"
    if tool == "(model)":
        text = (step.get("text") or "").strip().replace("\n", " ")
        if text.startswith("<"):  # tool call written as text: the agent reminds it to use the tool
            text = "(answered in text instead of a tool call; asked to use submit_holds)"
        return f"Muse Spark, turn {step.get('turn')}{ms}" + (f": {text[:120]}" if text else "")
    if tool == "submit_holds":
        return f"submit_holds: {args.get('count')} holds. Note: {args.get('note', '')}"
    if tool.startswith("("):
        return f"{tool} {json.dumps(args) if args else step.get('result', '')}".strip()
    picked = args.get("ids", args.get("id", ""))
    return f"{tool} {picked}{ms}"


def draw(jpeg, tap, holds, label):
    """The photo with the found holds outlined and the tap marked."""
    with Image.open(io.BytesIO(jpeg)) as raw:
        img = raw.convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    width = max(3, w // 250)
    for hold in holds:
        pts = [(x * w, y * h) for x, y in hold.get("points") or []]
        if len(pts) >= 3:
            d.line(pts + [pts[0]], fill=OUTLINE, width=width, joint="curve")
    cx, cy, r = tap["x"] * w, tap["y"] * h, max(14, w // 45)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=width)
    size = max(16, h // 30)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    d.rectangle([0, h - size * 2, w, h], fill=(0, 0, 0))
    d.text((size // 2, h - size * 1.6), label, fill=(255, 255, 255), font=font)
    return img


def live(env, jpeg, tap):
    """Run SAM 3.1 + Muse Spark: (holds, trace)."""
    from topo_agent.agent import meta_hold_finder
    find = meta_hold_finder(env)
    if find is None:
        sys.exit("MODEL_API_KEY is not set. Run without --live to replay a recorded run.")
    got = {}
    holds = find(jpeg, tap, on_trace=lambda trace: got.setdefault("trace", trace))
    return holds, got.get("trace", [])


def main(argv=None, env=None):
    env = dict(os.environ) if env is None else env
    ap = argparse.ArgumentParser(description="Tap one hold, get the whole problem")
    ap.add_argument("--scene", default="tape", choices=SCENES, help="drawn sample wall (default: tape)")
    ap.add_argument("--live", action="store_true", help="call Meta Model API (needs MODEL_API_KEY)")
    ap.add_argument("--photo", help="your own wall photo (with --live)")
    ap.add_argument("--tap", help="tapped point for --photo, as x,y fractions (0-1) from the top left")
    ap.add_argument("--out", default="demo-result.png", help="output image")
    ap.add_argument("--record", action="store_true", help=argparse.SUPPRESS)  # save a live sample run for replay
    a = ap.parse_args(argv)

    truth = None
    if a.photo:
        if not (a.live and a.tap):
            sys.exit("--photo needs --live and --tap x,y")
        jpeg = Path(a.photo).read_bytes()
        x, y = (float(v) for v in a.tap.split(","))
        tap, name = {"x": x, "y": y}, Path(a.photo).name
    else:
        jpeg, tap, truth = sample(a.scene)
        name = f"sample wall '{a.scene}'"

    if a.live:
        started = time.monotonic()
        holds, trace = live(env, jpeg, tap)
        took = time.monotonic() - started
        mode = "LIVE run (SAM 3.1 + Muse Spark)"
        if a.record and not a.photo:
            RUNS.mkdir(exist_ok=True)
            run = {"scene": a.scene, "seed": SEED, "recorded": time.strftime("%Y-%m-%d"), "seconds": round(took, 1),
                   "holds": [{k: h.get(k) for k in ("x", "y", "r", "points")} for h in holds], "trace": trace}
            (RUNS / f"{a.scene}.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    else:
        run = json.loads((RUNS / f"{a.scene}.json").read_text(encoding="utf-8"))
        holds, trace, took = run["holds"], run["trace"], run["seconds"]
        mode = f"RECORDED run from {run['recorded']} (no API call; use --live with MODEL_API_KEY to run it yourself)"

    print(f"{mode}\n{name}, tap at x={tap['x']:.2f}, y={tap['y']:.2f}\n\nWhat the AI did:")
    for step in trace:
        print("  - " + describe(step))
    print(f"\nFound {len(holds)} holds in {took:.1f} s.")
    if truth:
        w, h = bench.WIDTH, bench.HEIGHT
        hits, found, total = match(truth, holds, w, h)
        print(f"Against the answer: {hits} of {total} problem holds found, {found - hits} wrong.")
    label = ("LIVE" if a.live else "RECORDED") + f" - {len(holds)} holds"
    draw(jpeg, tap, holds, label).save(a.out)
    print(f"Image: {a.out}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # consoles that can't print every character (e.g. cp932 on Windows)
        sys.stdout.reconfigure(errors="replace")
    main()
