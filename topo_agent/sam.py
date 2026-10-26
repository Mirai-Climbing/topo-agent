# SPDX-License-Identifier: AGPL-3.0-or-later
"""SAM 3.1(Meta Model API)で、写真のホールドを1つずつ切り出す。

- 言葉で頼む("climbing hold" など):写真の中の、その言葉に当たる物を全部(箱と形)
- 点で頼む(押した所):そこにある物1つの形

Meta Model API の Responses(POST /v1/responses、model "sam-3.1")を使う。答えは特殊な記号の文字列で、
1行が1枚の絵(写真なら <0f> の1行)。1つの物が「番号<|box;...|><|mask;...|>」で、カンマで並ぶ。
形(mask)は箱の中だけの小さな白黒の絵で、Meta の meta-sam-parser で元に戻す。

返す候補は、ほかと同じ割合の形({"x", "y", "r", "points"}、r は写真の幅に対する半径)に、
id(SAM の番号)・conf(確からしさ)・area(写真に対する面積の割合)を足した物。

鍵(MODEL_API_KEY)は環境変数から読む。写真は送るだけで、残さない。
"""
import base64
import io
import json
import re

from PIL import Image

from topo_agent.baseline.shape import _simplify, _trace

API = "https://api.meta.ai/v1"
MODEL = "sam-3.1"
SEND_SIDE = 1600  # SAM に送る写真の長い辺(画素)
CONCEPTS = ("climbing hold", "climbing volume")  # 言葉で頼むときの言葉(ハリボテも別に頼む)
MAX_POINTS = 32  # 輪郭の点の数の上限
SAME = 0.6  # 2つの候補が、この割合以上重なったら同じ物とする

_FRAME = re.compile(r"<(\d+)f>")
_RECORD = re.compile(r"(?P<id>\d+)<\|box;(?P<box>[^|]*)\|><\|mask;(?P<mask>.*?)\|>(?=,\d+<\|box;|\s*$)")


def _kv(s):
    return dict(p.split("=", 1) for p in s.split(";") if "=" in p)


def shrink(jpeg, side=SEND_SIDE):
    """送る大きさの JPEG と、その幅・高さ。"""
    with Image.open(io.BytesIO(jpeg)) as raw:
        img = raw.convert("RGB")
    img.thumbnail((side, side))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=88)
    return out.getvalue(), img.width, img.height


def point_prompt(x, y, w, h):
    """押した所(割合)を、SAM の点の記号にする。"""
    return f"<|point;x={round(x * w)};y={round(y * h)};w={w};h={h}|>"


def request_body(jpeg, prompt, model=MODEL):
    """Responses API に送る中身。prompt は言葉か、point_prompt の記号。"""
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    return {"model": model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt},
                                                   {"type": "input_image", "image_url": url}]}],
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"}}


def output_text(response):
    """Responses API の答え(JSON)から、出力の文字列を取り出す。"""
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts = []
    for item in response.get("output") or []:
        for c in item.get("content") or []:
            if c.get("type") == "output_text":
                parts.append(c.get("text") or "")
    return "".join(parts)


def parse(text):
    """出力の文字列 → 物の並び [{"id", "frame", "box": (x1, y1, x2, y2), "w", "h", "conf", "mask": (H, W, payload)}]。
    1つも見つからなければ空(空の答えも正しい答え)。"""
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
    """形の記号 → 行ごとの 0/1 の並び(高さ H、幅 W)。"""
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    mh, mw, payload = mask
    encoding = "one_bit" if payload.startswith("!") else "lossless"
    raster = decode_mask_to_raster(SegmentationMask(encoding=encoding, payload=payload, width=mw, height=mh))
    return [list(raster[y * mw:(y + 1) * mw]) for y in range(mh)]


def to_candidate(obj):
    """SAM の物1つ → 候補(割合)。形が空なら None。形は箱の中だけの絵なので、箱の位置に戻す。"""
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
    """2つの箱(割合)の重なり ÷ 小さいほうの面積。"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1e-9
    return ix * iy / small


def merge(groups):
    """いくつかの頼み方の候補をまとめる(同じ物が重なっていたら、確からしいほうを残す)。番号は付け直す。"""
    out = []
    for c in sorted((c for g in groups for c in g), key=lambda c: -(c["conf"] or 0)):
        if all(_overlap(c["box"], o["box"]) < SAME for o in out):
            out.append(c)
    for i, c in enumerate(out):
        c["id"] = i
    return out


class Sam:
    """Meta Model API の SAM 3.1。post(body) → 答えの JSON(テストでは偽物を渡す)。"""

    def __init__(self, post):
        self.post = post

    def segment(self, jpeg, prompt):
        """jpeg(送る大きさにした物)を prompt で切り出した候補の並び。"""
        objs = parse(output_text(self.post(request_body(jpeg, prompt))))
        return [c for c in (to_candidate(o) for o in objs) if c]

    def candidates(self, jpeg, concepts=CONCEPTS):
        """写真の中のホールドの候補を全部(言葉ごとに頼んで、まとめる)。"""
        small, _, _ = shrink(jpeg)
        return merge([self.segment(small, c) for c in concepts])

    def at(self, jpeg, point):
        """押した所(割合)のホールドの形。なければ None。"""
        small, w, h = shrink(jpeg)
        found = self.segment(small, point_prompt(point["x"], point["y"], w, h))
        return max(found, key=lambda c: c["conf"] or 0) if found else None


def http_post(env, timeout=60):
    """本物の Meta Model API に送る関数。鍵がなければ None。"""
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
    # 試す:python -m topo_agent.sam 写真.jpg [言葉 | x,y(割合)]
    import os
    import sys
    if len(sys.argv) < 2:
        sys.exit("使い方:python -m topo_agent.sam 写真.jpg [言葉 | x,y]")
    post = http_post(dict(os.environ))
    if not post:
        sys.exit("MODEL_API_KEY がありません")
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
    print(f"{len([c for c in found if c])} 個", file=sys.stderr)
