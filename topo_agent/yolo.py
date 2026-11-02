"""YOLO-seg masks as pipeline candidates (a drop-in for the SAM stage).

YoloSeg.segment has the same (jpeg, prompt) -> candidates shape as Sam, so
agent.prepare and pipeline.find_with run unchanged: concept prompts return
every detection, point prompts return only detections covering the tap.
Inference runs once per photo (prepare calls segment up to 3 times); the
result is cached under the JPEG bytes. torch/ultralytics stay lazy so the
rest of the package imports without them.
"""
import io
import re
import threading

from PIL import Image

MAX_POINTS = 32  # max outline points per candidate (mirrors sam.py)


def point_in_poly(x, y, poly):
    """Ray-casting containment for fractional points."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def parse_point_prompt(prompt):
    """(fx, fy) tap fractions from a point prompt, or None for concepts."""
    m = re.search(r"x=([\d.]+);y=([\d.]+);w=(\d+);h=(\d+)", prompt or "")
    if not m:
        return None
    x, y, w, h = float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))
    if w <= 0 or h <= 0:
        return None
    return (x / w, y / h)


def poly_to_candidate(poly, img_w, img_h, cand_id, conf):
    """Fractional polygon (+ image size) -> pipeline candidate dict."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    area = abs(sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i]
                   for i in range(len(xs)))) / 2
    pts = poly
    if len(pts) > MAX_POINTS:
        pts = pts[::-(-len(pts) // MAX_POINTS)]
    return {"id": cand_id, "x": round(sum(xs) / len(xs), 4), "y": round(sum(ys) / len(ys), 4),
            "r": round(max(x2 - x1, y2 - y1) / 2, 4),
            "points": [[round(x, 4), round(y, 4)] for x, y in pts] if len(pts) >= 3 else [],
            "conf": conf, "area": round(area, 6), "box": [x1, y1, x2, y2]}


class YoloSeg:
    """Ultralytics YOLO-seg behind the Sam interface (exact prompt strings ignored)."""

    def __init__(self, weights, conf=0.5, imgsz=640, device=0):
        from ultralytics import YOLO  # lazy: torch stays optional for the package
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.lock = threading.Lock()
        self.cache = {}

    def _infer(self, jpeg):
        with Image.open(io.BytesIO(jpeg)) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
        pred = self.model.predict(rgb, conf=self.conf, imgsz=self.imgsz,
                                  device=self.device, verbose=False)[0]
        out = []
        if pred.masks is not None:
            for seg, box in zip(pred.masks.xy, pred.boxes):
                poly = [(float(x) / w, float(y) / h) for x, y in seg]
                if len(poly) >= 3:
                    out.append((poly, float(box.conf[0])))
        return w, h, out

    def segment(self, jpeg, prompt):
        """Detections for the photo (all of them, or only tap-covering ones)."""
        with self.lock:
            if jpeg not in self.cache:
                self.cache.clear()
                self.cache[jpeg] = self._infer(jpeg)
            w, h, dets = self.cache[jpeg]
        tap = parse_point_prompt(prompt)
        if tap is not None:
            dets = [(p, c) for p, c in dets if point_in_poly(tap[0], tap[1], p)]
        return [poly_to_candidate(p, w, h, i, c) for i, (p, c) in enumerate(dets)]
