# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score hold finders (the database-free part of mirai-climbing-v2's server/hold_eval.py).

Pretend the user tapped one marked hold, then compare each method's answer with the marked holds:
- recall:    marked holds that the method outlined
- precision: outlined holds that are marked holds
- seconds per photo

Hard sample walls (--bench, topo_agent/baseline/bench.py) are drawn by code, so they may be sent to external AI APIs.

    python -m topo_agent.eval --bench --methods color,gemini,sam_color,meta

- color      color only (pre-hackathon, no AI)
- gemini     Gemini agent (pre-hackathon)
- sam_color  SAM 3.1 candidates filtered by color only (no AI judgement)
- meta       topo agent (SAM 3.1 + Muse Spark)

Real problems from the app database are scored by mirai-climbing-v2's server/hold_eval.py
(only photos from users who agreed are sent to external AI APIs).
"""
import argparse
import io
import os
import random
import sys
import time

from PIL import Image

from topo_agent.baseline.shape import _covers


def holds_of(marks):
    """Hold marks of a topo (start and finish included, lines excluded)."""
    return [m for m in (marks or {}).get("marks", []) if m.get("kind", "hold") == "hold"]


def pick_tap(holds, seed):
    """The hold we pretend was tapped: a plain hold ("any") if there is one. Fixed by seed so runs are repeatable."""
    plain = [h for h in holds if h.get("use") == "any"] or holds
    return random.Random(seed).choice(plain)


def match(truth, found, width, height):
    """Pair marked holds (truth) with found holds one to one. A pair is when one's center lies inside the other.
    Returns (pairs, found count, truth count)."""
    used, hits = set(), 0
    for t in truth:
        best = None
        for i, f in enumerate(found):
            if i in used:
                continue
            if _covers(t, f["x"], f["y"], width, height) or _covers(f, t["x"], t["y"], width, height):
                d = (t["x"] - f["x"]) ** 2 + (t["y"] - f["y"]) ** 2
                if best is None or d < best[0]:
                    best = (d, i)
        if best:
            used.add(best[1])
            hits += 1
    return hits, len(found), len(truth)


def finders(env, names):
    """Method name → function (JPEG, point → holds). Methods without settings (API keys) are skipped."""
    from topo_agent.agent import meta_hold_finder, sam_color_finder
    from topo_agent.baseline.gemini_agent import color_only, gemini_hold_finder

    def color(jpeg, point):
        with Image.open(io.BytesIO(jpeg)) as img:
            return color_only(img.convert("RGB"), point)

    made = {"color": lambda: color, "gemini": lambda: gemini_hold_finder(env),
            "sam_color": lambda: sam_color_finder(env), "meta": lambda: meta_hold_finder(env)}
    out = {}
    for name in names:
        f = made[name]() if name in made else None
        if f:
            out[name] = f
        else:
            print(f"{name}: not available (no settings, or unknown name)", file=sys.stderr)
    return out


def evaluate(items, methods, clock=time.monotonic):
    """Run every method on every problem and score it.
    Returns rows of {"problem", "method", "hits", "found", "truth", "seconds", "error"}."""
    rows = []
    for item in items:
        holds = holds_of(item["marks"])
        tap = item.get("tap") or pick_tap(holds, item["id"])
        width, height = item["marks"]["photo"]["width"], item["marks"]["photo"]["height"]
        for name, find in methods.items():
            started = clock()
            try:
                found = find(item["jpeg"], {"x": tap["x"], "y": tap["y"]}) or []
                error = ""
            except Exception as err:  # one failing problem should not stop the rest
                found, error = [], f"{type(err).__name__}: {str(err)[:120]}"
            hits, n_found, n_truth = match(holds, found, width, height)
            rows.append({"problem": item["id"], "method": name, "hits": hits, "found": n_found, "truth": n_truth,
                         "seconds": round(clock() - started, 1), "error": error})
    return rows


def summary(rows):
    """Totals per method (recall, precision, mean seconds)."""
    out = {}
    for name in dict.fromkeys(r["method"] for r in rows):
        mine = [r for r in rows if r["method"] == name]
        hits, found, truth = (sum(r[k] for r in mine) for k in ("hits", "found", "truth"))
        out[name] = {"problems": len(mine), "recall": round(hits / truth, 3) if truth else 0.0,
                     "precision": round(hits / found, 3) if found else 0.0,
                     "seconds": round(sum(r["seconds"] for r in mine) / len(mine), 1),
                     "errors": sum(1 for r in mine if r["error"])}
    return out


def by_scene(rows):
    """Totals per scene and method (sample wall problem ids are "scene-seed")."""
    out = {}
    for scene in dict.fromkeys(r["problem"].rsplit("-", 1)[0] for r in rows):
        out[scene] = summary([r for r in rows if r["problem"].rsplit("-", 1)[0] == scene])
    return out


def bench(env, names, scenes=None, seeds=(11, 12)):
    from topo_agent.baseline import bench as hold_bench
    methods = finders(env, names)
    if not methods:
        sys.exit("no method is available")
    rows = evaluate(hold_bench.items(scenes or hold_bench.NAMES, seeds), methods)
    for r in rows:
        print(f"{r['problem']:<10} {r['method']:<9} truth {r['truth']:>3}  found {r['found']:>3}  hits {r['hits']:>3}"
              f"  {r['seconds']:>5}s  {r['error']}", flush=True)
    print()
    for scene, per in by_scene(rows).items():
        for name, s in per.items():
            print(f"{scene:<8} {name:<9} recall {s['recall']:>4.0%}  precision {s['precision']:>4.0%}"
                  f"  mean {s['seconds']}s  errors {s['errors']}")
    print()
    for name, s in summary(rows).items():
        print(f"{'TOTAL':<8} {name:<9} recall {s['recall']:>4.0%}  precision {s['precision']:>4.0%}"
              f"  mean {s['seconds']}s  errors {s['errors']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score hold finders")
    ap.add_argument("--methods", default="color", help="comma separated: color,gemini,sam_color,meta")
    ap.add_argument("--bench", action="store_true", help="score on the hard sample walls (topo_agent/baseline/bench.py)")
    ap.add_argument("--scenes", default="", help="only these sample wall scenes (comma separated, e.g. light,twins)")
    ap.add_argument("--seeds", default="11,12", help="sample wall seeds (comma separated)")
    a = ap.parse_args(argv)
    if not a.bench:
        sys.exit("only --bench for now (real problems: mirai-climbing-v2's server/hold_eval.py)")
    names = [x.strip() for x in a.scenes.split(",") if x.strip()] or None
    seeds = tuple(int(x) for x in a.seeds.split(",") if x.strip())
    bench(dict(os.environ), [x.strip() for x in a.methods.split(",") if x.strip()], names, seeds)


if __name__ == "__main__":
    main()
