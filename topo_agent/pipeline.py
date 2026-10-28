# SPDX-License-Identifier: AGPL-3.0-or-later
"""Three-stage route pipeline: SAM segmentation, CV color grouping, single-pass Muse Spark.

A selectable alternative to the ReAct agent in agent.py (kept as the `meta` baseline).
Each photo goes through exactly three stages:

1. SAM 3.1 segments every hold (agent.prepare: the `climbing hold` / `climbing volume`
   concept prompts plus a point prompt at the tap, merged into numbered candidates).
2. Computer vision classifies each hold's color from the photo: HSV + CIELAB features
   per hold, mask-boundary pixels excluded, highlights and shadows down-weighted.
   Holds are grouped by color similarity and same-group spatial neighbors are listed.
3. Muse Spark finds the tapped hold's route in exactly one model call (no tools, no loop)
   and answers with JSON only.

The trace records the latency of each stage separately (sam, color, spark, post) plus
the API call counts and token usage when the API reports it. There are no retries:
on a Spark failure or invalid JSON the answer falls back to the tapped hold's CV
color group, and on a SAM failure to the pre-hackathon color-only method.
"""
import io
import json
import logging
import math
import time

from PIL import Image

from topo_agent import agent as agent_mod
from topo_agent import sam as sam_mod
from topo_agent.baseline.shape import _covers, _hsv, _inside, _mean

log = logging.getLogger(__name__)

SHRINK = 0.85  # outlines are shrunk toward their center for color sampling (excludes mask-boundary pixels)
MIN_PIXELS = 5  # fewer inside pixels than this: sample the unshrunk outline instead
TEMPERATURE = 0.2  # same as the ReAct agent (agent.py); only the reasoning path differs, not the sampling

SYSTEM3 = """You are a climbing route grouping engine, not an exploratory agent.

Task: find every hold of the same route as the tapped hold (marked * in the table and outlines).

Input: hold ids, positions (x, y, size on a 0-1000 scale), precomputed colors (names, HSV and
CIELAB features you must trust, not recompute), color similarity groups, same-group spatial
neighbors, the wall photo, a second image where every hold is outlined and numbered, and a
close-up of the tapped hold.

Instructions:
1. Answer in a single inference pass. Find ONLY the tapped hold's route, not the other routes.
2. Do not perform color detection or segmentation again.
3. First look at the close-up: a strip of tape next to the tapped hold means a tape route
   (holds of any color with the same tape; precomputed colors do not help). Without tape,
   the route is the holds of the same color on the same wall.
4. Do not assume that all holds of the same color belong to one route: a same-color group
   across a gap or on another wall is usually another route.
5. Never invent hold ids; use only the ids in the table. Always include the tapped hold.
6. When in doubt about a same-color hold on the same wall, include it. Exclude holds that
   are clearly a different color or on a different wall.
7. Return only valid JSON, without explanations or intermediate reasoning:
{"hold_ids": [1, 4, 7], "confidence": 0.95}"""


def rgb_to_lab(rgb):
    """sRGB (0-255) → CIELAB (D65). Plain Python, no numpy."""
    def lin(v):
        v /= 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(v) for v in rgb)
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return (round(116 * fy - 16, 1), round(500 * (fx - fy), 1), round(200 * (fy - fz), 1))


def _shrunk(pts, factor=SHRINK):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return [[cx + (x - cx) * factor, cy + (y - cy) * factor] for x, y in pts]


def classify_colors(img, cands, side=agent_mod.COLOR_SIDE):
    """Set c["hsv"], c["lab"] and c["color_name"] for each candidate from the photo.

    Pixels are sampled inside the outline shrunk toward its center, so wall pixels
    leaking in at the mask boundary do not shift the color. Within the sample, washed-out
    highlights and dark shadows are down-weighted: saturated holds average the hue of the
    saturated pixels only, pale holds average the half farthest from the wall brightness.
    CIELAB of the sample mean is kept as numeric features for grouping."""
    small = img.convert("RGB").copy()
    small.thumbnail((side, side))
    w, h = small.size
    px = small.load()
    hsv = [[_hsv(px[x, y]) for x in range(w)] for y in range(h)]
    every = [c for row in hsv[::6] for c in row[::6]]
    wall_v = sorted(c[2] for c in every)[len(every) // 2]
    for c in cands:
        pts = c.get("points") or []
        shapes = [(_shrunk(pts), None), (pts, None)] if len(pts) >= 3 else [(None, SHRINK), (None, 1.0)]
        pixels = []
        for shape, rf in shapes:
            if shape is not None:
                xs, ys = [p[0] for p in shape], [p[1] for p in shape]
                box = (min(xs), min(ys), max(xs), max(ys))
                inside = lambda x, y: _inside(x, y, shape)  # noqa: E731
            else:
                probe = {"x": c["x"], "y": c["y"], "r": c["r"] * rf}
                box = (c["x"] - probe["r"], c["y"] - probe["r"] * w / h,
                       c["x"] + probe["r"], c["y"] + probe["r"] * w / h)
                inside = lambda x, y: _covers(probe, x, y, w, h)  # noqa: E731
            x0, y0 = max(0, int(box[0] * w)), max(0, int(box[1] * h))
            x1, y1 = min(w - 1, int(box[2] * w) + 1), min(h - 1, int(box[3] * h) + 1)
            pixels = [(px[x, y], hsv[y][x]) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
                      if inside((x + 0.5) / w, (y + 0.5) / h)]
            if len(pixels) >= MIN_PIXELS:
                break
        if not pixels:
            x, y = min(w - 1, int(c["x"] * w)), min(h - 1, int(c["y"] * h))
            pixels = [(px[x, y], hsv[y][x])]
        sats = sorted(p[1][1] for p in pixels)
        if sats[len(sats) // 2] >= 0.28:  # saturated hold: hue of the saturated pixels only
            sel = [p for p in pixels if p[1][1] >= max(0.2, sats[len(sats) // 2] * 0.5)] or pixels
        else:  # pale hold: the half farthest from the wall brightness
            sel = sorted(pixels, key=lambda p: abs(p[1][2] - wall_v), reverse=True)
            sel = sel[:max(1, len(sel) // 2)]
        col = _mean([p[1] for p in sel])
        mean_rgb = tuple(sum(p[0][k] for p in sel) / len(sel) for k in range(3))
        c["hsv"] = tuple(round(v, 3) for v in col)
        c["lab"] = rgb_to_lab(mean_rgb)
        c["color_name"] = agent_mod.color_name(col)
    return cands


def group_by_color(cands):
    """Group ids by color similarity (union-find over color_distance <= SIMILAR). Sets c["group"]."""
    parent = {c["id"]: c["id"] for c in cands}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k, a in enumerate(cands):
        for b in cands[k + 1:]:
            if agent_mod.color_distance(a["hsv"], b["hsv"]) <= agent_mod.SIMILAR:
                parent[root(a["id"])] = root(b["id"])
    labels = {}
    for c in cands:
        labels.setdefault(root(c["id"]), len(labels))
        c["group"] = labels[root(c["id"])]
    return cands


def add_neighbors(cands, width, height):
    """Set c["near"]: same-group ids within LINK of the photo's long side (spatial chain hints)."""
    side = max(width, height)
    for a in cands:
        a["near"] = sorted(b["id"] for b in cands
                           if b["id"] != a["id"] and b["group"] == a["group"]
                           and math.hypot((a["x"] - b["x"]) * width, (a["y"] - b["y"]) * height)
                           <= agent_mod.LINK * side)
    return cands


def fallback_ids(cands, tap_id):
    """Deterministic answer without AI: the tapped hold's CV color group (tap first)."""
    group = next(c["group"] for c in cands if c["id"] == tap_id)
    return agent_mod.clean_ids([c["id"] for c in cands if c["group"] == group],
                               {c["id"] for c in cands}, tap_id)


def prompt_table(cands, tap_id):
    """All holds as compact text: color features, groups and spatial neighbors."""
    rows = ["id | color | hue sat val | L a b | group | x y size | near"]
    for c in cands:
        h, s, v = c["hsv"]
        lab = " ".join(str(v) for v in c["lab"])
        rows.append(f"{c['id']}{'*' if c['id'] == tap_id else ''} | {c['color_name']} | "
                    f"{h} {s} {v} | {lab} | {c['group']} | "
                    f"{round(c['x'] * 1000)} {round(c['y'] * 1000)} {round(c['r'] * 1000)} | "
                    f"{','.join(map(str, c['near']))}")
    groups = {}
    for c in cands:
        groups.setdefault(c["group"], []).append(c["id"])
    names = {c["group"]: c["color_name"] for c in cands}
    detail = "; ".join(f"{g} ({names[g]}): {ids}" for g, ids in sorted(groups.items()))
    return ("\n".join(rows) + f"\nColor groups: {detail}\n"
            f"The user tapped hold {tap_id} (marked {tap_id}*). Return ONLY the holds of the tapped "
            f"hold's route as JSON.")


def single_call(post, model=agent_mod.MODEL, effort=agent_mod.EFFORT):
    """A one-shot model_call using Chat Completions (no tools). Returns {"text", "usage"}."""
    def call(messages):
        body = {"model": model, "messages": messages, "temperature": TEMPERATURE, "store": False}
        if effort:
            body["reasoning_effort"] = effort
        data = post(body)
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        return {"text": msg.get("content") or "", "usage": data.get("usage") or {}}
    return call


def parse_answer(text, known):
    """Route JSON → {"ids", "confidence", "error"}. Rejects unknown/duplicate ids and bad shapes."""
    body = text.strip()
    if body.startswith("```"):  # ```json ... ``` fences are unwrapped, not rejected
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {"ids": [], "confidence": 0, "error": "not valid JSON"}
    ids, conf = (data.get("hold_ids"), data.get("confidence")) if isinstance(data, dict) else (None, None)
    if (not isinstance(ids, list) or not ids
            or not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0 <= conf <= 1):
        return {"ids": [], "confidence": 0, "error": "missing hold_ids or confidence"}
    clean = []
    for i in ids:
        if not isinstance(i, int) or isinstance(i, bool) or i not in known:
            return {"ids": [], "confidence": 0, "error": f"unknown id {i!r}"}
        if i in clean:
            return {"ids": [], "confidence": 0, "error": f"hold {i} is assigned twice"}
        clean.append(i)
    return {"ids": clean, "confidence": conf, "error": ""}


def run_once(spark, img, cands, tap_id):
    """One Spark call for the tapped route. Returns (hold ids, info).

    info holds the raw reply, token usage and the fallback reason ("" when Spark answered).
    Never retries: on failure or invalid JSON the answer is the CV color group of the tap."""
    small = img.convert("RGB").copy()
    small.thumbnail((agent_mod.AI_SIDE, agent_mod.AI_SIDE))
    close_up, _ = agent_mod.Tools(img, cands, tap_id).look_closer([tap_id])
    content = [{"type": "text", "text": prompt_table(cands, tap_id)},
               agent_mod._image_part(agent_mod._jpeg(small)),
               agent_mod._image_part(agent_mod._jpeg(agent_mod.annotate(img, cands, tap_id)))]
    if close_up:
        content.append(agent_mod._image_part(close_up))
    messages = [{"role": "system", "content": SYSTEM3}, {"role": "user", "content": content}]
    try:
        reply = spark(messages)
    except Exception as err:
        return fallback_ids(cands, tap_id), {"text": "", "usage": {},
                                             "fallback": f"{type(err).__name__}: {str(err)[:200]}"}
    text, usage = reply.get("text") or "", reply.get("usage") or {}
    parsed = parse_answer(text, {c["id"] for c in cands})
    if parsed["error"]:
        return fallback_ids(cands, tap_id), {"text": text, "usage": usage, "fallback": parsed["error"]}
    ids = agent_mod.clean_ids(parsed["ids"], {c["id"] for c in cands}, tap_id)
    return ids, {"text": text, "usage": usage, "fallback": "", "confidence": parsed["confidence"]}


def find_with(sam, spark, jpeg, point, on_trace=None, clock=time.monotonic):
    """(JPEG, point → holds) with injected SAM and Spark. Trace steps: (sam), (color), (spark), (post)."""
    started = clock()
    try:
        img, cands, tap_id = agent_mod.prepare(sam, jpeg, point)
    except Exception as err:  # SAM did not answer: color only (pre-hackathon method)
        log.warning("route pipeline: SAM failed (%s: %s)", type(err).__name__, str(err)[:300])
        from topo_agent.baseline.gemini_agent import color_only
        with Image.open(io.BytesIO(jpeg)) as raw:
            holds = color_only(raw.convert("RGB"), point)
        trace = [{"tool": "(error)", "result": f"{type(err).__name__}: {str(err)[:200]}"},
                 {"tool": "(fallback: color)", "args": {"count": len(holds)}}]
        if on_trace:
            on_trace(trace)
        return holds
    trace = [{"tool": "(sam)", "args": {"count": len(cands), "sam_calls": 3},
              "ms": round((clock() - started) * 1000)}]
    if tap_id is None:
        if on_trace:
            on_trace(trace)
        return []
    started = clock()
    classify_colors(img, cands)
    group_by_color(cands)
    add_neighbors(cands, img.width, img.height)
    trace.append({"tool": "(color)", "args": {"groups": len({c["group"] for c in cands})},
                  "ms": round((clock() - started) * 1000)})
    started = clock()
    ids, info = run_once(spark, img, cands, tap_id)
    step = {"tool": "(spark)", "args": {"count": len(ids), "spark_calls": 1},
            "ms": round((clock() - started) * 1000)}
    if info.get("fallback"):
        step["args"]["fallback"] = info["fallback"]
    if info.get("confidence") is not None:
        step["args"]["confidence"] = info["confidence"]
    if info.get("usage"):
        step["args"]["tokens"] = {k: v for k, v in info["usage"].items() if "token" in k}
    trace.append(step)
    started = clock()
    holds = agent_mod._holds(cands, ids)
    trace.append({"tool": "(post)", "args": {"count": len(holds)}, "ms": round((clock() - started) * 1000)})
    log.info("route pipeline: %s", json.dumps(trace, ensure_ascii=False))
    if on_trace:
        on_trace(trace)
    return holds


def route_finder(env, on_trace=None):
    """Find holds with the three-stage pipeline (SAM + CV colors + single-pass Spark).

    (JPEG, point → holds), or None without a key. Pass on_trace(trace) here, or to each
    call as find(jpeg, point, on_trace=...), to receive the per-stage steps."""
    post, chat = sam_mod.http_post(env), agent_mod.http_chat(env)
    if not (post and chat):
        return None
    sam = sam_mod.Sam(post)
    spark = single_call(chat, env.get("META_AGENT_MODEL", agent_mod.MODEL),
                        env.get("META_AGENT_EFFORT", agent_mod.EFFORT))

    def find(jpeg, point, on_trace=on_trace):
        return find_with(sam, spark, jpeg, point, on_trace)
    return find
