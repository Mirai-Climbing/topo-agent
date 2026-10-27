# SPDX-License-Identifier: AGPL-3.0-or-later
"""ホールド探しの点数を測る(mirai-climbing-v2 の server/hold_eval.py の、データベースを使わない部分)。

印の1つを押したことにして、方法ごとの答えと比べる。
- 見つけた割合(recall):答えのホールドのうち、囲めた物
- 当たりの割合(precision):囲んだ物のうち、答えに入っていた物
- かかった秒数

難しい見本の壁(--bench、topo_agent/baseline/bench.py):描いた絵なので、外の AI に送ってよい。

    python -m topo_agent.eval --bench --methods color,gemini,sam_color,meta

- color      色だけ(ハッカソン前。AI なし)
- gemini     Gemini のエージェント(ハッカソン前)
- sam_color  SAM 3.1 の候補を、色だけで絞る(AI の判断なし)
- meta       トポエージェント(SAM 3.1 + Muse Spark)

本物の課題(アプリのデータベース)で測るのは、mirai-climbing-v2 の server/hold_eval.py(同意した人の課題だけを外の AI に送る)。
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
    """トポの印のうち、ホールドの印(スタート・ゴールも含む。線は外す)。"""
    return [m for m in (marks or {}).get("marks", []) if m.get("kind", "hold") == "hold"]


def pick_tap(holds, seed):
    """押したことにするホールド。ふつうのホールド(any)から選ぶ(なければ、どれか)。毎回同じになるよう seed で決める。"""
    plain = [h for h in holds if h.get("use") == "any"] or holds
    return random.Random(seed).choice(plain)


def match(truth, found, width, height):
    """答え(truth)と見つけた物(found)を1つずつ組にする。組になる条件:片方の真ん中が、もう片方の中にある。
    (組になった数, 見つけた数, 答えの数) を返す。"""
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
    """測る方法の名前 → 関数(JPEG, point → 候補の並び)。設定がなくて使えない方法は外す。"""
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
            print(f"{name}:使えません(設定がない、または知らない名前)", file=sys.stderr)
    return out


def evaluate(items, methods, clock=time.monotonic):
    """課題ごとに、それぞれの方法で探して点数を付ける。
    返すのは、行の並び({"problem", "method", "hits", "found", "truth", "seconds", "error"})。"""
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
            except Exception as err:  # 1つの課題で止まっても、ほかは測る
                found, error = [], f"{type(err).__name__}: {str(err)[:120]}"
            hits, n_found, n_truth = match(holds, found, width, height)
            rows.append({"problem": item["id"], "method": name, "hits": hits, "found": n_found, "truth": n_truth,
                         "seconds": round(clock() - started, 1), "error": error})
    return rows


def summary(rows):
    """方法ごとの合計(見つけた割合・当たりの割合・平均の秒数)。"""
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
    """場面ごと・方法ごとの合計(見本の壁の課題の番号は「場面-seed」)。"""
    out = {}
    for scene in dict.fromkeys(r["problem"].rsplit("-", 1)[0] for r in rows):
        out[scene] = summary([r for r in rows if r["problem"].rsplit("-", 1)[0] == scene])
    return out


def bench(env, names, scenes=None, seeds=(11, 12)):
    from topo_agent.baseline import bench as hold_bench
    methods = finders(env, names)
    if not methods:
        sys.exit("使える方法がありません")
    rows = evaluate(hold_bench.items(scenes or hold_bench.NAMES, seeds), methods)
    for r in rows:
        print(f"{r['problem']:<10} {r['method']:<9} 答え{r['truth']:>3}  見つけた{r['found']:>3}  当たり{r['hits']:>3}"
              f"  {r['seconds']:>5}秒  {r['error']}", flush=True)
    print()
    for scene, per in by_scene(rows).items():
        for name, s in per.items():
            print(f"{scene:<8} {name:<9} 見つけた割合 {s['recall']:>4.0%}  当たりの割合 {s['precision']:>4.0%}"
                  f"  平均 {s['seconds']}秒  エラー {s['errors']}")
    print()
    for name, s in summary(rows).items():
        print(f"{'合計':<7} {name:<9} 見つけた割合 {s['recall']:>4.0%}  当たりの割合 {s['precision']:>4.0%}"
              f"  平均 {s['seconds']}秒  エラー {s['errors']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="ホールド探しの点数を測る")
    ap.add_argument("--methods", default="color", help="color,gemini,sam_color,meta から(カンマで区切る)")
    ap.add_argument("--bench", action="store_true", help="難しい見本の壁(topo_agent/baseline/bench.py)で測る")
    ap.add_argument("--scenes", default="", help="見本の壁の場面だけを測る(カンマで区切る。light,twins など)")
    ap.add_argument("--seeds", default="11,12", help="見本の壁の seed(カンマで区切る)")
    a = ap.parse_args(argv)
    if not a.bench:
        sys.exit("今は --bench だけです(本物の課題は mirai-climbing-v2 の server/hold_eval.py)")
    names = [x.strip() for x in a.scenes.split(",") if x.strip()] or None
    seeds = tuple(int(x) for x in a.seeds.split(",") if x.strip())
    bench(dict(os.environ), [x.strip() for x in a.methods.split(",") if x.strip()], names, seeds)


if __name__ == "__main__":
    main()
