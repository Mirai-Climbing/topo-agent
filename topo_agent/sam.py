# SPDX-License-Identifier: AGPL-3.0-or-later
"""Segment every hold in a photo with SAM 3.1 (Meta Model API).

- Concept prompt ("climbing hold", ...): every instance of that concept in the photo (box and mask)
- Point prompt (the tapped spot): the mask of the one object there

Uses the Responses API of Meta Model API (POST /v1/responses, model "sam-3.1"). The output is special-token text,
one line per frame (a photo is the single line <0f>). Each object is "id<|box;...|><|mask;...|>", comma separated.
A mask is a small binary raster covering only its box, decoded with Meta's meta-sam-parser.

Candidates use the same fractional shape as the rest of the app ({"x", "y", "r", "points"}, r is the radius
relative to the photo width), plus id (SAM object id), conf (confidence) and area (fraction of the photo).

The key (MODEL_API_KEY) is read from the environment. Photos are only sent, never kept (store: false).
"""
import base64
import io
import json
import re

from PIL import Image

from topo_agent.baseline.shape import _simplify, _trace

API = "https://api.meta.ai/v1"
MODEL = "sam-3.1"
SEND_SIDE = 1600  # long side of the photo sent to SAM (pixels)
CONCEPTS = ("climbing hold", "climbing volume")  # concept prompts (volumes are asked for separately)
MAX_POINTS = 32  # max points per outline
SAME = 0.6  # two candidates overlapping this much are the same object

_FRAME = re.compile(r"<(\d+)f>")
_RECORD = re.compile(r"(?P<id>\d+)<\|box;(?P<box>[^|]*)\|><\|mask;(?P<mask>.*?)\|>(?=,\d+<\|box;|\s*$)")


def _kv(s):
    return dict(p.split("=", 1) for p in s.split(";") if "=" in p)


def shrink(jpeg, side=SEND_SIDE):
    """JPEG resized for sending, with its width and height."""
    with Image.open(io.BytesIO(jpeg)) as raw:
        img = raw.convert("RGB")
    img.thumbnail((side, side))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=88)
    return out.getvalue(), img.width, img.height


def point_prompt(x, y, w, h):
    """The tapped spot (fractions) as a SAM point prompt (must be prefixed with the frame <0f> and an object id)."""
    return f"<0f>1<|point;x={round(x * w)};y={round(y * h)};w={w};h={h}|>"


def request_body(jpeg, prompt, model=MODEL):
    """Responses API request body. prompt is a noun phrase or a point_prompt."""
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    return {"model": model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt},
                                                   {"type": "input_image", "image_url": url}]}],
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
            "store": False}  # do not keep the response at Meta (the default is to keep it)


def output_text(response):
    """The output text of a Responses API response (JSON)."""
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts = []
    for item in response.get("output") or []:
        for c in item.get("content") or []:
            if c.get("type") == "output_text":
                parts.append(c.get("text") or "")
    return "".join(parts)


def parse(text):
    """Output text → objects [{"id", "frame", "box": (x1, y1, x2, y2), "w", "h", "conf", "mask": (H, W, payload)}].
    Empty when nothing matched (an empty output is a valid answer)."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        m = _FRAME.match(line)
        if not m:
            continue
        for r in _RECORD.finditer(line[m.end():]):
            box = _kv(r["box"])
            head, _, data = r["mask"].partition(";data=")
            meta = _kv(head)
            mh, mw, payload = data.split(",", 2)
            conf = box.get("c", meta.get("c"))
            out.append({"id": int(r["id"]), "frame": int(m.group(1)),
                        "box": tuple(int(float(box[k])) for k in ("x1", "y1", "x2", "y2")),
                        "w": int(box["w"]), "h": int(box["h"]),
                        "conf": float(conf) if conf is not None else None,
                        "mask": (int(mh), int(mw), payload)})
    return out


def decode(mask):
    """Mask token → rows of 0/1 (height H, width W)."""
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    mh, mw, payload = mask
    encoding = "one_bit" if payload.startswith("!") else "lossless"
    raster = decode_mask_to_raster(SegmentationMask(encoding=encoding, payload=payload, width=mw, height=mh))
    return [list(raster[y * mw:(y + 1) * mw]) for y in range(mh)]


def to_candidate(obj):
    """One SAM object → candidate (fractions), None if the mask is empty. The mask covers only its box, so place it there."""
    rows = decode(obj["mask"])
    mh, mw = len(rows), len(rows[0]) if rows else 0
    x1, y1, x2, y2 = obj["box"]
    W, H = obj["w"], obj["h"]
    sx, sy = (x2 - x1) / mw if mw else 0, (y2 - y1) / mh if mh else 0
    filled = sum(map(sum, rows))
    if not filled:
        return None
    cx = sum(x for row in rows for x, v in enumerate(row) if v) / filled
    cy = sum(y for y, row in enumerate(rows) for v in row if v) / filled
    edge = _simplify(_trace(rows, mw, mh), 0.8)[:MAX_POINTS * 4]
    if len(edge) > MAX_POINTS:
        edge = edge[::-(-len(edge) // MAX_POINTS)]
    points = [[round((x1 + (x + 0.5) * sx) / W, 4), round((y1 + (y + 0.5) * sy) / H, 4)] for x, y in edge]
    return {"id": obj["id"], "x": round((x1 + (cx + 0.5) * sx) / W, 4), "y": round((y1 + (cy + 0.5) * sy) / H, 4),
            "r": round(max(x2 - x1, (y2 - y1)) / 2 / W, 4), "points": points if len(points) >= 3 else [],
            "conf": obj["conf"], "area": round(filled * sx * sy / (W * H), 6), "box": [x1 / W, y1 / H, x2 / W, y2 / H]}


def _overlap(a, b):
    """Intersection of two boxes (fractions) divided by the smaller box area."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1e-9
    return ix * iy / small


def merge(groups):
    """Merge candidates from several prompts (keep the more confident of overlapping duplicates). Ids are renumbered."""
    out = []
    for c in sorted((c for g in groups for c in g), key=lambda c: -(c["conf"] or 0)):
        if all(_overlap(c["box"], o["box"]) < SAME for o in out):
            out.append(c)
    for i, c in enumerate(out):
        c["id"] = i
    return out


class Sam:
    """SAM 3.1 on Meta Model API. post(body) → response JSON (tests pass a fake)."""

    def __init__(self, post):
        self.post = post

    def segment(self, jpeg, prompt):
        """Candidates for prompt on jpeg (already resized for sending)."""
        objs = parse(output_text(self.post(request_body(jpeg, prompt))))
        return [c for c in (to_candidate(o) for o in objs) if c]

    def candidates(self, jpeg, concepts=CONCEPTS):
        """All hold candidates in the photo (one request per concept, merged)."""
        small, _, _ = shrink(jpeg)
        return merge([self.segment(small, c) for c in concepts])

    def at(self, jpeg, point):
        """The hold at the tapped spot (fractions), or None."""
        small, w, h = shrink(jpeg)
        found = self.segment(small, point_prompt(point["x"], point["y"], w, h))
        return max(found, key=lambda c: c["conf"] or 0) if found else None


def http_post(env, timeout=60):
    """A function posting to the real Meta Model API, or None without a key."""
    key = env.get("MODEL_API_KEY")
    if not key:
        return None
    import httpx
    base = env.get("META_API_BASE", API)

    def post(body):
        r = httpx.post(f"{base}/responses", json=body, timeout=timeout,
                       headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        if r.status_code >= 400:
            raise RuntimeError(f"SAM {r.status_code}: {r.text[:300]}")
        return r.json()
    return post


if __name__ == "__main__":
    # Try it: python -m topo_agent.sam photo.jpg [noun phrase | x,y (fractions)]
    import os
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python -m topo_agent.sam photo.jpg [noun phrase | x,y]")
    post = http_post(dict(os.environ))
    if not post:
        sys.exit("MODEL_API_KEY is not set")
    sam = Sam(post)
    jpeg = open(sys.argv[1], "rb").read()
    arg = sys.argv[2] if len(sys.argv) > 2 else ""
    if re.fullmatch(r"[\d.]+,[\d.]+", arg):
        x, y = map(float, arg.split(","))
        found = [sam.at(jpeg, {"x": x, "y": y})]
    else:
        small, _, _ = shrink(jpeg)
        found = sam.segment(small, arg) if arg else sam.candidates(jpeg)
    print(json.dumps([{k: c[k] for k in ("id", "x", "y", "r", "conf", "area")} for c in found if c],
                     ensure_ascii=False, indent=1))
    print(f"{len([c for c in found if c])} found", file=sys.stderr)
