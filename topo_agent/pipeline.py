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

The single-pass system prompt has variants for real gym light (META3_PROMPT=v1, v2
or v3; default v1 keeps the original rules): v2 verifies colors visually and rejects
wall-colored pale lookalikes, v3 additionally merges split color groups and follows
hold chains to recover shadowed routes.

Optionally (META3_EMBED=1), DINOv2 hold-crop embeddings add a per-hold top-similar
list to the Spark table as a second similarity signal next to color. torch stays a
lazy optional dependency: without it the pipeline silently runs without hints.
"""
import io
import json
import logging
import math
import os
import time

from PIL import Image, ImageDraw

from topo_agent import agent as agent_mod
from topo_agent import sam as sam_mod
from topo_agent.baseline.shape import _covers, _hsv, _inside, _mean

log = logging.getLogger(__name__)

SHRINK = 0.85  # outlines are shrunk toward their center for color sampling (excludes mask-boundary pixels)
MIN_PIXELS = 5  # fewer inside pixels than this: sample the unshrunk outline instead
TEMPERATURE = 0.2  # same as the ReAct agent (agent.py); only the reasoning path differs, not the sampling
EMBED_ENV = "META3_EMBED"  # set to "1" to add DINOv2 similarity hints to the Spark table (default off)
EMBED_MODEL = "dinov2_vits14"  # small DINOv2, fetched from the torch hub cache on first use
EMBED_SIZE = 224  # DINOv2 input side (pixels)
EMBED_TOP_K = 3  # similar holds listed per row (ids with cosine scores)
EMBED_MARGIN = 0.15  # crop grows past the outline by this fraction of its box
EMBED_MEAN = (0.485, 0.456, 0.406)  # DINOv2 input normalization
EMBED_STD = (0.229, 0.224, 0.225)
EMBED_BATCH = 32  # crops per forward pass

_DINO = None  # cached (model, device), loaded lazily so torch stays optional

SYSTEM3 = """You are a climbing route grouping engine, not an exploratory agent.

Task: find every hold of the same route as the tapped hold (marked * in the table and outlines).

Input: hold ids, positions (x, y, size on a 0-1000 scale), precomputed colors (names, HSV and
CIELAB features you must trust, not recompute), color similarity groups, same-group spatial
neighbors, optional visual-embedding hints (the embed column, when present), the wall photo,
a second image where every hold is outlined and numbered, and a close-up of the tapped hold.

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
{"hold_ids": [1, 4, 7], "confidence": 0.95}
8. When the embed column is present, treat it as a visual-similarity hint only: same-route
   holds usually rank each other high, but color, tape and wall position decide."""

SYSTEM3_V2 = """You are a climbing route grouping engine, not an exploratory agent.

Task: find every hold of the same route as the tapped hold (marked * in the table and outlines).

Input: hold ids, positions (x, y, size on a 0-1000 scale), precomputed colors (names, HSV and
CIELAB features), color similarity groups, same-group spatial neighbors, the wall photo, a second
image where every hold is outlined and numbered, and a close-up of the tapped hold.

The table is a cheap color sensor, not the truth: on real photos chalk dust washes colors out and
shadows darken them, so color names and groups are often wrong. Trust your eyes over the table.

Instructions:
1. Answer in a single inference pass. Find ONLY the tapped hold's route, not the other routes.
2. First look at the close-up: a strip of tape next to the tapped hold means a tape route
   (holds of any color with the same tape; colors and groups do not help). Without tape,
   the route is the holds that look the same color as the tapped hold IN THE PHOTOS.
3. Decide the tapped hold's true color from the close-up and the photos, not from its table row.
   If the table calls it gray, white or black but it looks red, orange, yellow, green, blue or
   pink, the table is wrong: collect the holds that visually match the tap, whatever group the
   table puts them in.
4. Judge matches by hue family in the photos and ignore brightness: a hold in shadow looks darker
   and a hold under a lamp looks washed out, but the hue stays the same. Shadow or glare alone
   never excludes a hold.
5. Pale gray, white and beige holds that look like the bare wall or chalk dust are NOT the route
   when the tapped hold is a saturated color: exclude them even if the table puts them in the
   tapped hold's group. Pale holds qualify only when the tapped hold is itself pale, and then only
   the ones that clearly share its tint on the same wall.
6. The route runs from the bottom to the top as one continuous line you can follow from the tapped
   hold. A same-color hold across a gap or on another wall is usually another route: exclude it.
7. Never invent hold ids; use only the ids in the table. Always include the tapped hold.
8. When in doubt, compare the hold with the tapped hold in the photos: include it only if the
   colors clearly match and it sits on the same wall and line. Exclude wall-colored lookalikes
   and holds that are clearly a different color.
9. Return only valid JSON, without explanations or intermediate reasoning:
{"hold_ids": [1, 4, 7], "confidence": 0.95}"""

SYSTEM3_V3 = """You are a climbing route grouping engine, not an exploratory agent.

Task: find every hold of the same route as the tapped hold (marked * in the table and outlines).

Input: hold ids, positions (x, y, size on a 0-1000 scale), precomputed colors (names, HSV and
CIELAB features), color similarity groups, same-group spatial neighbors, the wall photo, a second
image where every hold is outlined and numbered, and a close-up of the tapped hold.

The table is a cheap color sensor, not the truth: on real photos chalk dust washes colors out and
shadows darken them, so color names and groups are often wrong. Trust your eyes over the table.

Instructions:
1. Answer in a single inference pass. Find ONLY the tapped hold's route, not the other routes.
2. First look at the close-up: a strip of tape next to the tapped hold means a tape route
   (holds of any color with the same tape; colors and groups do not help). Without tape,
   the route is the holds that look the same color as the tapped hold IN THE PHOTOS.
3. Decide the tapped hold's true color from the close-up and the photos, not from its table row.
   If the table calls it gray, white or black but it looks red, orange, yellow, green, blue or
   pink, the table is wrong: collect the holds that visually match the tap, whatever group the
   table puts them in.
4. Judge matches by hue family in the photos and ignore brightness: a hold in shadow looks darker
   and a hold under a lamp looks washed out, but the hue stays the same. Shadow or glare alone
   never excludes a hold.
5. Pale gray, white and beige holds that look like the bare wall or chalk dust are NOT the route
   when the tapped hold is a saturated color: exclude them even if the table puts them in the
   tapped hold's group. Pale holds qualify only when the tapped hold is itself pale, and then only
   the ones that clearly share its tint on the same wall.
6. One route is often split across several table groups (shadow turns green to gray or black,
   chalk turns red to pink or white). After the visual matches in the tapped hold's group, scan
   the OTHER groups for holds that look like the tapped hold in the photos on the same wall and
   line, and include them too.
7. The near column lists only direct neighbors; the route continues through chains (tap to A to B
   to C). Include visual matches reachable through a chain of same-looking holds, not just the
   tapped hold's direct neighbors.
8. A problem usually has 4 to 15 holds. Fewer than 4 means shadowed or far holds were missed:
   look again at the other groups and at far holds on the same line before answering.
9. The route runs from the bottom to the top as one continuous line you can follow from the tapped
   hold. A same-color hold across a gap or on another wall is usually another route: exclude it.
10. Never invent hold ids; use only the ids in the table. Always include the tapped hold.
11. When in doubt, compare the hold with the tapped hold in the photos: include it only if the
    colors clearly match and it sits on the same wall and line. Exclude wall-colored lookalikes
    and holds that are clearly a different color.
12. Return only valid JSON, without explanations or intermediate reasoning:
{"hold_ids": [1, 4, 7], "confidence": 0.95}"""

PROMPTS = {"v1": SYSTEM3, "v2": SYSTEM3_V2, "v3": SYSTEM3_V3}
DEFAULT_PROMPT = "v1"


def system_prompt(name):
    """Variant name → (name, system text). Empty or unknown names fall back to v1 (current behavior)."""
    key = (name or "").strip().lower()
    if key in ("", DEFAULT_PROMPT, "default"):
        return DEFAULT_PROMPT, SYSTEM3
    if key in PROMPTS:
        return key, PROMPTS[key]
    log.warning("route pipeline: unknown META3_PROMPT %r, using v1", name)
    return DEFAULT_PROMPT, SYSTEM3


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


def embed_enabled(env=None):
    """True when META3_EMBED=1 (default off). An explicit env dict entry wins over os.environ."""
    if env is not None and EMBED_ENV in env:
        return str(env[EMBED_ENV]) == "1"
    return os.environ.get(EMBED_ENV) == "1"


def hold_crop(img, cand, margin=EMBED_MARGIN, masked=True):
    """One hold as a crop for visual embeddings: the outline box grown by margin.

    Masked crops keep only the hold (black background), so wall pixels do not leak
    into the vector. Circle candidates (no outline) fall back to an ellipse mask."""
    w, h = img.size
    pts = cand.get("points") or []
    if len(pts) >= 3:
        xs = [p[0] * w for p in pts]
        ys = [p[1] * h for p in pts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    else:
        r = cand.get("r", 0.02) * w
        x0, y0, x1, y1 = cand["x"] * w - r, cand["y"] * h - r, cand["x"] * w + r, cand["y"] * h + r
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    box = (max(0, int(x0 - dx)), max(0, int(y0 - dy)),
           min(w, int(x1 + dx) + 1), min(h, int(y1 + dy) + 1))
    if box[2] <= box[0] or box[3] <= box[1]:  # degenerate outline: a patch at the center
        cx, cy = int(cand["x"] * w), int(cand["y"] * h)
        box = (max(0, cx - 2), max(0, cy - 2), min(w, cx + 3), min(h, cy + 3))
    crop = img.convert("RGB").crop(box)
    if not masked:
        return crop
    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    if len(pts) >= 3:
        draw.polygon([(x * w - box[0], y * h - box[1]) for x, y in pts], fill=255)
    else:
        draw.ellipse([x0 - box[0], y0 - box[1], x1 - box[0], y1 - box[1]], fill=255)
    return Image.composite(crop, Image.new("RGB", crop.size, (0, 0, 0)), mask)


def _cosine(a, b):
    """Cosine similarity of two vectors (0.0 when either is zero)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def dino_embed(crops):
    """DINOv2 ViT-S/14 embeddings for hold crops, L2-normalized (one vector per crop).

    torch is imported lazily and the model is cached, so the first call loads the
    weights and later calls reuse them. Raises ImportError without torch."""
    import torch  # lazy: torch stays an optional local-only dependency

    global _DINO
    if not crops:
        return []
    if _DINO is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("facebookresearch/dinov2", EMBED_MODEL).to(device).eval()
        _DINO = (model, device)
    model, device = _DINO
    batch = []
    for crop in crops:
        px = crop.convert("RGB").resize((EMBED_SIZE, EMBED_SIZE))
        t = torch.tensor(list(px.getdata()), dtype=torch.float32).reshape(EMBED_SIZE, EMBED_SIZE, 3)
        t = t.permute(2, 0, 1) / 255.0
        for ch in range(3):
            t[ch] = (t[ch] - EMBED_MEAN[ch]) / EMBED_STD[ch]
        batch.append(t)
    out = []
    with torch.no_grad():
        stacked = torch.stack(batch).to(device)
        for i in range(0, len(stacked), EMBED_BATCH):
            out.append(model(stacked[i:i + EMBED_BATCH]).cpu())
    embs = torch.cat(out)
    return (embs / embs.norm(dim=1, keepdim=True)).tolist()


def add_embed_similar(img, cands, embedder=None, top_k=EMBED_TOP_K):
    """Set c["embed_top"] to [(id, cosine)...] of the top_k most similar other holds.

    The default embedder is DINOv2 over masked hold crops; pass a fake embedder
    (crops → vectors) in tests. Without torch the default raises ImportError and
    every list stays empty, so the pipeline runs on color alone."""
    for c in cands:
        c["embed_top"] = []
    if len(cands) < 2:
        return cands
    run = embedder or dino_embed
    try:
        embs = run([hold_crop(img, c) for c in cands])
    except ImportError:
        if embedder is not None:
            raise
        log.warning("route pipeline: torch is not installed, embedding hints are off")
        return cands
    if len(embs) != len(cands):
        log.warning("route pipeline: embedder returned %d vectors for %d holds, hints are off",
                    len(embs), len(cands))
        return cands
    for i, c in enumerate(cands):
        ranked = sorted((_cosine(embs[i], embs[j]), cands[j]["id"])
                        for j in range(len(cands)) if j != i)
        top = ranked[-top_k:][::-1] if top_k > 0 else []
        c["embed_top"] = [(oid, round(score, 3)) for score, oid in top]
    return cands


def fallback_ids(cands, tap_id):
    """Deterministic answer without AI: the tapped hold's CV color group (tap first)."""
    group = next(c["group"] for c in cands if c["id"] == tap_id)
    return agent_mod.clean_ids([c["id"] for c in cands if c["group"] == group],
                               {c["id"] for c in cands}, tap_id)


def prompt_table(cands, tap_id):
    """All holds as compact text: color features, groups, spatial neighbors and embed hints."""
    show_embed = any(c.get("embed_top") for c in cands)
    rows = ["id | color | hue sat val | L a b | group | x y size | near" + (" | embed" if show_embed else "")]
    for c in cands:
        h, s, v = c["hsv"]
        lab = " ".join(str(v) for v in c["lab"])
        row = (f"{c['id']}{'*' if c['id'] == tap_id else ''} | {c['color_name']} | "
               f"{h} {s} {v} | {lab} | {c['group']} | "
               f"{round(c['x'] * 1000)} {round(c['y'] * 1000)} {round(c['r'] * 1000)} | "
               f"{','.join(map(str, c['near']))}")
        if show_embed:
            row += f" | {','.join(f'{i}:{s:.2f}' for i, s in c.get('embed_top') or [])}"
        rows.append(row)
    groups = {}
    for c in cands:
        groups.setdefault(c["group"], []).append(c["id"])
    names = {c["group"]: c["color_name"] for c in cands}
    detail = "; ".join(f"{g} ({names[g]}): {ids}" for g, ids in sorted(groups.items()))
    table = "\n".join(rows) + f"\nColor groups: {detail}\n"
    if show_embed:
        table += ("Embed lists the visually most similar holds per row as id:score "
                  "(higher is more similar); use it as a hint with color and position.\n")
    return (table + f"The user tapped hold {tap_id} (marked {tap_id}*). Return ONLY the holds of the tapped "
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


def run_once(spark, img, cands, tap_id, prompt=""):
    """One Spark call for the tapped route. Returns (hold ids, info).

    prompt selects the system prompt variant (v1, v2 or v3; default v1 keeps current
    behavior). info holds the raw reply, token usage, the resolved variant name and the
    fallback reason ("" when Spark answered). Never retries: on failure or invalid JSON
    the answer is the CV color group of the tap."""
    variant, system = system_prompt(prompt)
    small = img.convert("RGB").copy()
    small.thumbnail((agent_mod.AI_SIDE, agent_mod.AI_SIDE))
    close_up, _ = agent_mod.Tools(img, cands, tap_id).look_closer([tap_id])
    content = [{"type": "text", "text": prompt_table(cands, tap_id)},
               agent_mod._image_part(agent_mod._jpeg(small)),
               agent_mod._image_part(agent_mod._jpeg(agent_mod.annotate(img, cands, tap_id)))]
    if close_up:
        content.append(agent_mod._image_part(close_up))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    try:
        reply = spark(messages)
    except Exception as err:
        return fallback_ids(cands, tap_id), {"text": "", "usage": {}, "prompt": variant,
                                             "fallback": f"{type(err).__name__}: {str(err)[:200]}"}
    text, usage = reply.get("text") or "", reply.get("usage") or {}
    parsed = parse_answer(text, {c["id"] for c in cands})
    if parsed["error"]:
        return fallback_ids(cands, tap_id), {"text": text, "usage": usage, "prompt": variant,
                                             "fallback": parsed["error"]}
    ids = agent_mod.clean_ids(parsed["ids"], {c["id"] for c in cands}, tap_id)
    return ids, {"text": text, "usage": usage, "prompt": variant, "fallback": "",
                 "confidence": parsed["confidence"]}


def find_with(sam, spark, jpeg, point, on_trace=None, clock=time.monotonic,
              prompt="", embedder=None, use_embed=None):
    """(JPEG, point → holds) with injected SAM and Spark. Trace steps: (sam), (color), (spark), (post).

    prompt selects the single-pass system prompt variant (v1, v2 or v3; default v1).
    use_embed adds DINOv2 similarity hints (an extra (embed) trace step); None reads
    META3_EMBED from the environment (default off). embedder is injectable for tests
    (crops → vectors); embedding failures never break the pipeline, they skip the hints."""
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
    if use_embed if use_embed is not None else embed_enabled():
        started = clock()
        try:
            add_embed_similar(img, cands, embedder=embedder)
            if any(c.get("embed_top") for c in cands):
                trace.append({"tool": "(embed)",
                              "args": {"hints": sum(len(c.get("embed_top") or []) for c in cands)},
                              "ms": round((clock() - started) * 1000)})
        except ImportError:  # torch is optional: silently run without embedding hints
            pass
        except Exception as err:
            log.warning("route pipeline: embeddings failed (%s: %s)", type(err).__name__, str(err)[:300])
            for c in cands:
                c["embed_top"] = []
    started = clock()
    ids, info = run_once(spark, img, cands, tap_id, prompt=prompt)
    step = {"tool": "(spark)", "args": {"count": len(ids), "spark_calls": 1, "prompt": info["prompt"]},
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


def route_finder(env, on_trace=None, embedder=None):
    """Find holds with the three-stage pipeline (SAM + CV colors + single-pass Spark).

    (JPEG, point → holds), or None without a key. Pass on_trace(trace) here, or to each
    call as find(jpeg, point, on_trace=...), to receive the per-stage steps.
    META3_PROMPT in env selects the single-pass system prompt variant (v1, v2 or v3).
    META3_EMBED=1 in env adds DINOv2 similarity hints (torch stays optional);
    embedder (crops → vectors) is injectable for tests."""
    post, chat = sam_mod.http_post(env), agent_mod.http_chat(env)
    if not (post and chat):
        return None
    sam = sam_mod.Sam(post)
    spark = single_call(chat, env.get("META_AGENT_MODEL", agent_mod.MODEL),
                        env.get("META_AGENT_EFFORT", agent_mod.EFFORT))
    prompt = env.get("META3_PROMPT", "")
    use_embed = embed_enabled(env)

    def find(jpeg, point, on_trace=on_trace):
        return find_with(sam, spark, jpeg, point, on_trace, prompt=prompt,
                         embedder=embedder, use_embed=use_embed)
    return find
