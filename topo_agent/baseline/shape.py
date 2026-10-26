# SPDX-License-Identifier: AGPL-3.0-or-later
"""ホールドの輪郭を、写真の色で見分ける(AI の輪郭は角ばっていて、押した所の形も知りたいため)。

やり方(numpy を使わず、Pillow と Python だけで。ホールド1つあたり 96 画素四方くらいの小さな絵で動かす):
1. ホールドのまわりを切り出して小さくし、真ん中の色を「ホールドの色」にする
2. 真ん中から、同じ色の画素を順にたどって広げる。色の濃いホールドは色合い(色相)で、白・黒・灰色のホールドは明るさで比べる
   (光の当たり方で明るさが変わっても、色合いはあまり変わらないため。壁の影やふちを拾いにくい)
3. かたまりの中の穴(光った所など)を埋め、ふちをなぞって点を減らし、写真全体の割合に直す
4. 切り出した四角のふちに大きくはみ出すときは、壁と同じ色につながっているので、使わない
"""
import colorsys
import math
from collections import deque

WORK = 96  # 切り出した絵の長い辺(画素)
MAX_POINTS = 32
SCAN = 640  # 同じ色のホールドを探すときの、写真の長い辺(画素。小さなホールドも拾えるように)
MAX_EXTRA = 40
MIN_BLOB = 6  # 見落としとして足す、かたまりの小ささの下限(探す写真の画素)
BLOB_HUE = 0.035  # かたまり全体の色合いの、押したホールドとの違いの上限


def _hsv(rgb):
    return colorsys.rgb_to_hsv(*(v / 255 for v in rgb))


def _hue_gap(a, b):
    return min(abs(a - b), 1 - abs(a - b))


def colored(c):
    """色の濃いホールドか(白・黒・灰色ではない)。"""
    return c[1] >= 0.28 and c[2] >= 0.15


def same(c, seed):
    """画素 c が、ホールドの色 seed と同じに見えるか。"""
    if colored(seed):
        # 光が当たって白っぽく見える所も拾えるよう、鮮やかさは控えめに比べる。暗い所(影・天井の梁)は外す
        return (_hue_gap(c[0], seed[0]) < 0.05 and c[1] >= max(0.2, seed[1] * 0.4)
                and c[2] >= max(0.12, seed[2] * 0.55))
    return c[1] < 0.25 and abs(c[2] - seed[2]) < 0.12


def _mean(colors):
    if not colors:
        return (0, 0, 0)
    # 色相は角度なので、そのまま足さず、向きを足して平均する
    x = sum(math.cos(c[0] * 2 * math.pi) * c[1] for c in colors)
    y = sum(math.sin(c[0] * 2 * math.pi) * c[1] for c in colors)
    h = (math.atan2(y, x) / (2 * math.pi)) % 1
    return (h, sum(c[1] for c in colors) / len(colors), sum(c[2] for c in colors) / len(colors))


def _trace(mask, w, h):
    """かたまりのふちを、時計回りになぞる(ムーアの近傍をたどるやり方)。"""
    start = next(((x, y) for y in range(h) for x in range(w) if mask[y][x]), None)
    if not start:
        return []
    dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    out, cur, back = [start], start, 4
    for _ in range(w * h * 4):
        for i in range(8):
            d = (back + 1 + i) % 8
            nx, ny = cur[0] + dirs[d][0], cur[1] + dirs[d][1]
            if 0 <= nx < w and 0 <= ny < h and mask[ny][nx]:
                cur, back = (nx, ny), (d + 4) % 8
                break
        else:
            break
        if cur == start:
            break
        out.append(cur)
    return out


def _simplify(points, tol):
    """点を減らす(ダグラス・ポイカー)。輪郭は閉じているので、いちばん遠い2点で分けてから減らす。"""
    def rdp(pts):
        if len(pts) < 3:
            return pts
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        dx, dy = x2 - x1, y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5 or 1
        far, idx = 0, 0
        for i, (x, y) in enumerate(pts[1:-1], 1):
            d = abs(dy * x - dx * y + x2 * y1 - y2 * x1) / norm
            if d > far:
                far, idx = d, i
        if far <= tol:
            return [pts[0], pts[-1]]
        return rdp(pts[:idx + 1])[:-1] + rdp(pts[idx:])
    if len(points) < 4:
        return points
    i = max(range(len(points)), key=lambda k: (points[k][0] - points[0][0]) ** 2 + (points[k][1] - points[0][1]) ** 2)
    return rdp(points[:i + 1])[:-1] + rdp(points[i:] + [points[0]])[:-1]


def _grow(hsv, w, h, sx, sy, seed):
    """(sx, sy) から、seed と同じ色の画素をたどったかたまり。"""
    mask = [[False] * w for _ in range(h)]
    if not same(hsv[sy][sx], seed):
        return mask, 0
    q, mask[sy][sx], count = deque([(sx, sy)]), True, 1
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not mask[ny][nx] and same(hsv[ny][nx], seed):
                mask[ny][nx] = True
                count += 1
                q.append((nx, ny))
    return mask, count


def _fill_holes(mask, w, h):
    """かたまりの中の穴(光って白く見える所など)を埋める。ふちから届かない所が穴。"""
    outside = [[False] * w for _ in range(h)]
    q = deque((x, y) for y in range(h) for x in range(w) if (x in (0, w - 1) or y in (0, h - 1)) and not mask[y][x])
    for x, y in q:
        outside[y][x] = True
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not mask[ny][nx] and not outside[ny][nx]:
                outside[ny][nx] = True
                q.append((nx, ny))
    return [[mask[y][x] or not outside[y][x] for x in range(w)] for y in range(h)]


def shape_at(img, x, y, half, seed=None):
    """(x, y)(割合)にあるホールドの輪郭(割合の点の並び)。half は切り出す四角の半分の幅(写真の幅に対する割合)。
    seed を渡さなければ、押した所のまわりの色をホールドの色にする。見分けられなければ None。"""
    W, H = img.size
    hx, hy = half * W, half * W
    box = (max(0, int(x * W - hx)), max(0, int(y * H - hy)), min(W, int(x * W + hx) + 1), min(H, int(y * H + hy) + 1))
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None
    crop = img.crop(box)
    scale = WORK / max(crop.size)
    w, h = max(4, int(crop.width * scale)), max(4, int(crop.height * scale))
    px = crop.resize((w, h)).load()
    hsv = [[_hsv(px[i, j]) for i in range(w)] for j in range(h)]
    cx = min(max(int((x * W - box[0]) / crop.width * w), 0), w - 1)
    cy = min(max(int((y * H - box[1]) / crop.height * h), 0), h - 1)
    near = [(hsv[j][i], i, j) for j in range(max(0, cy - 2), min(h, cy + 3)) for i in range(max(0, cx - 2), min(w, cx + 3))]
    if seed is None:
        seed = _mean([c for c, _, _ in near])
    # たどり始めは、真ん中の近くで、いちばんホールドの色らしい画素
    start = next(((i, j) for c, i, j in sorted(near, key=lambda n: abs(n[1] - cx) + abs(n[2] - cy)) if same(c, seed)), None)
    if not start:
        return None
    mask, count = _grow(hsv, w, h, start[0], start[1], seed)
    if count < 6 or count > 0.6 * w * h:
        return None
    border = [mask[j][i] for j in range(h) for i in range(w) if i in (0, w - 1) or j in (0, h - 1)]
    if sum(border) > 0.12 * len(border):  # 四角のふちに大きくはみ出す → 壁と同じ色につながっている
        return None
    mask = _fill_holes(mask, w, h)
    outline = _simplify(_trace(mask, w, h), tol=0.7)
    if len(outline) < 3:
        return None
    if len(outline) > MAX_POINTS:
        step = len(outline) / MAX_POINTS
        outline = [outline[int(k * step)] for k in range(MAX_POINTS)]
    return [[round((box[0] + (i + 0.5) / w * crop.width) / W, 4), round((box[1] + (j + 0.5) / h * crop.height) / H, 4)]
            for i, j in outline]


def refine(img, hold):
    """AI の候補(割合の {"x", "y", "r", "points"?})の輪郭を、写真の色で直した点の並び。直せなければ None。"""
    img = img.convert("RGB")
    size = hold["r"]
    if hold.get("points"):
        xs, ys = [p[0] for p in hold["points"]], [p[1] for p in hold["points"]]
        size = max(size, (max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2 * img.height / img.width)
    return shape_at(img, hold["x"], hold["y"], max(size * 2.2, 0.03))


def refine_all(img, holds):
    """候補すべての輪郭を、色で直せる物だけ直す(直せない物は AI の輪郭のまま)。"""
    img = img.convert("RGB")
    for hold in holds:
        better = refine(img, hold)
        if better:
            hold["points"] = better
    return holds


def outline_at(img, point):
    """押した所(割合)にあるホールドの輪郭。ホールドの大きさが分からないので、小さい四角から順に試し、
    ふちにはみ出さずに見分けられた最初の形を使う。見分けられなければ None。"""
    img = img.convert("RGB")
    for half in (0.04, 0.08, 0.14):
        pts = shape_at(img, point["x"], point["y"], half)
        if pts:
            return pts
    return None


def _inside(x, y, pts):
    hit = False
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def _covers(hold, x, y, W, H):
    """(x, y)(割合)が、その候補の中か。"""
    if hold.get("points"):
        return _inside(x, y, hold["points"])
    return ((hold["x"] - x) * W) ** 2 + ((hold["y"] - y) * H) ** 2 <= (hold["r"] * W) ** 2


CHAIN_ROUNDS = 6  # 見つけたホールドのまわりを、さらに探す回数(課題は、ホールドが鎖のようにつながって続くため)
LINK = 0.25  # 見つけたホールドから、この距離(写真の長い辺に対する割合)までのかたまりは、同じ課題として足す
PALE_GAP = 0.3
SMALL_PART = 12  # 押したホールドの大きさ(画素の数)の、この分の1より小さいかたまりは拾わない  # 白・黒・灰色のホールドを探すときの、壁との明るさの違いの下限


def _wall_color(hsv):
    """写真のいちばん多い色(壁)の、鮮やかさと明るさのまん中の値。"""
    flat = [c for row in hsv[::4] for c in row[::4]]
    s = sorted(c[1] for c in flat)
    v = sorted(c[2] for c in flat)
    return s[len(s) // 2], v[len(v) // 2]


def pale_ok(color, wall):
    """白・黒・灰色のホールドを、壁と見分けて探してよいか(壁と明るさが十分ちがうときだけ)。"""
    return abs(color[2] - wall[1]) >= PALE_GAP


def extend_by_color(img, holds, point, everywhere=False):
    """押したホールドと同じ色のかたまりを、写真の中から探して足す(AI が見落とした物を拾う)。
    探すのは、見つかった候補のまわり(候補全体を囲む四角を広げた所)だけ。足した物のまわりも、また探す
    (課題はホールドが続いているので、押した所から鎖のようにたどる)。everywhere なら写真全体
    (エージェントの道具。となりの壁の物も混ざるが、選ぶのはエージェント)。
    白・黒・灰色のように色の薄いホールドは、壁と明るさが十分ちがうときだけ、明るさで探す(白い壁の白は探さない)。"""
    img = img.convert("RGB")
    W, H = img.size
    seed_hold = next((h for h in holds if _covers(h, point["x"], point["y"], W, H)), None)
    if seed_hold is None:  # AI の候補が押した所から少しずれていても、押した所の色で探す
        pts = outline_at(img, point)
        if not pts:
            return holds
        seed_hold = {"x": point["x"], "y": point["y"], "r": 0.01, "points": pts,
                     "color": next((h.get("color", "") for h in holds), "")}
        holds = holds + [seed_hold]  # 押したホールド自身も、課題のホールドに入れる
    scale = SCAN / max(W, H)
    w, h = max(8, int(W * scale)), max(8, int(H * scale))
    px = img.resize((w, h)).load()
    hsv = [[_hsv(px[x, y]) for x in range(w)] for y in range(h)]
    inside = [hsv[y][x] for y in range(h) for x in range(w) if _covers(seed_hold, (x + 0.5) / w, (y + 0.5) / h, W, H)]
    if len(inside) < 3:
        return holds
    # 押したホールドよりずっと小さな点(テープの切れ端・ネジ・汚れ)は拾わない。本物の写真で、壁じゅうの赤い点を拾った
    min_blob = max(MIN_BLOB, len(inside) // SMALL_PART)
    # 押したホールドの色(輪郭の中の、色の濃い半分の平均。輪郭が少しはみ出して壁の画素が混ざっても大丈夫なように)
    inside.sort(key=lambda c: c[1], reverse=True)
    color = _mean(inside[:max(3, len(inside) // 2)])
    if not colored(color):
        # 白・黒・灰色:輪郭の中の、壁から遠い半分の平均の明るさ
        wall = _wall_color(hsv)
        inside.sort(key=lambda c: abs(c[2] - wall[1]), reverse=True)
        color = _mean(inside[:max(3, len(inside) // 2)])
        if colored(color) or not pale_ok(color, wall):
            return holds
        min_blob = max(MIN_BLOB, len(inside) // 6)  # ボルトの穴やホールドの影の小さな点を拾わないように
    near = [[same(hsv[y][x], color) for x in range(w)] for y in range(h)]
    seen = [[False] * w for _ in range(h)]
    blobs = []
    for sy in range(h):
        for sx in range(w):
            if not near[sy][sx] or seen[sy][sx]:
                continue
            blob, q = [], deque([(sx, sy)])
            seen[sy][sx] = True
            while q:
                x, y = q.popleft()
                blob.append((x, y))
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < w and 0 <= ny < h and near[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        q.append((nx, ny))
            if not (min_blob <= len(blob) <= 0.03 * w * h):  # 小さすぎる点・大きすぎる面は、ホールドではない
                continue
            # 画素ごとには色合いを広めに見ているので、かたまり全体の色合いでもう一度比べる(赤とオレンジを分ける)
            if colored(color) and _hue_gap(_mean([hsv[y][x] for x, y in blob])[0], color[0]) >= BLOB_HUE:
                continue
            bx = [p[0] for p in blob]
            by = [p[1] for p in blob]
            cx, cy = sum(bx) / len(bx) / w + 0.5 / w, sum(by) / len(by) / h + 0.5 / h
            r = max(max(bx) - min(bx) + 1, (max(by) - min(by) + 1) * W / H * h / w) / 2 / w
            blobs.append((cx, cy, r))
    extra = []
    for round_ in range(1 if everywhere else CHAIN_ROUNDS):
        xs = [hh["x"] for hh in holds + extra]
        ys = [hh["y"] for hh in holds + extra]
        # 候補全体を囲む四角を、幅・高さの4分の1ずつ広げる(候補が少ないときも、写真の 15% は見る。
        # 広げすぎると、となりの壁のほかの課題まで拾うため)
        # 四角の外でも、見つけたホールドのどれかに近ければ足す(課題は、ホールドが続いて上へ登るため)
        px_, py_ = max((max(xs) - min(xs)) / 4, 0.15), max((max(ys) - min(ys)) / 4, 0.15)
        area = (0, 0, 1, 1) if everywhere else (min(xs) - px_, min(ys) - py_, max(xs) + px_, max(ys) + py_)
        if round_:  # 2回目からは、見つけたホールドの近くだけ(四角は、ホールドが増えるほど広がって、壁じゅうを拾うため)
            area = (2, 2, 2, 2)
        side = max(W, H)
        added = 0
        for cx, cy, r in blobs:
            linked = any(math.hypot((cx - hh["x"]) * W, (cy - hh["y"]) * H) <= LINK * side for hh in holds + extra)
            if not (linked or area[0] <= cx <= area[2] and area[1] <= cy <= area[3]):
                continue
            if any(_covers(hh, cx, cy, W, H) for hh in holds + extra):
                continue
            hold = {"x": round(cx, 4), "y": round(cy, 4), "r": round(min(max(r, 0.008), 0.2), 4),
                    "color": seed_hold.get("color", "")}
            better = shape_at(img, cx, cy, max(r * 2.5, 0.03), seed=color)
            if better:
                hold["points"] = better
            extra.append(hold)
            added += 1
            if len(extra) >= MAX_EXTRA:
                return holds + extra
        if not added:
            break
    return holds + extra
