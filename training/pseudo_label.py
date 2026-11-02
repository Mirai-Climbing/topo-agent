"""Pseudo-label unlabeled Kaggle images with the converted Mask R-CNN teacher.

Keeps class-1 (hold) masks with score >= THRESH, writes YOLO-seg labels.
Images already labeled in VIA are skipped (gold GT lives in via2yolo output).

Usage:
    python pseudo_label.py --src ~/data/kaggle-climbing \
        --teacher /tmp/tv_hold_rcnn.pth --dst ~/data/yolo-hold-pseudo
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from d2teacher import build_model

SOURCES = ['bh', 'bh-phone', 'sm']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Kaggle dataset root')
    ap.add_argument('--teacher', required=True, help='converted torchvision weights')
    ap.add_argument('--dst', required=True, help='YOLO-seg pseudo dataset root')
    ap.add_argument('--thresh', type=float, default=0.6)
    ap.add_argument('--min-area', type=float, default=150, help='full-res px')
    ap.add_argument('--max-frac', type=float, default=0.25, help='of image area')
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    labeled = set()
    for name in SOURCES:
        d = json.loads((src / f'{name}-annotation.json').read_text())
        for m in d['_via_img_metadata'].values():
            if m['regions']:
                labeled.add(f'{name}/{m["filename"]}')

    jobs = []
    for name in SOURCES:
        for p in sorted((src / name).glob('*.jpg')):
            if f'{name}/{p.name}' not in labeled:
                jobs.append(p)
    print(f'unlabeled images: {len(jobs)}')

    model = build_model()
    model.load_state_dict(torch.load(args.teacher, map_location='cpu',
                                     weights_only=True), strict=True)
    model.cuda().eval()

    imgdir = dst / 'images' / 'train'
    lbldir = dst / 'labels' / 'train'
    imgdir.mkdir(parents=True, exist_ok=True)
    lbldir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_inst = 0
    for i, p in enumerate(jobs):
        tag = p.parent.name.replace('-', '')
        stem = f'{tag}_{p.stem}'
        img = Image.open(p).convert('RGB')
        w, h = img.size
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = model([x.cuda()])[0]
        keep = (out['labels'] == 1) & (out['scores'] >= args.thresh)
        lines = []
        for j in torch.where(keep)[0]:
            m = (out['masks'][j, 0].cpu().numpy() > 0.5).astype(np.uint8)
            area = m.sum()
            if area < args.min_area or area > args.max_frac * w * h:
                continue
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            approx = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            pts = ' '.join(f'{px / w:.6f} {py / h:.6f}' for px, py in approx)
            lines.append(f'0 {pts}')
        if lines:
            link = imgdir / f'{stem}.jpg'
            if not link.exists():
                link.symlink_to(p)
            (lbldir / f'{stem}.txt').write_text('\n'.join(lines) + '\n')
            n_inst += len(lines)
        if (i + 1) % 200 == 0:
            print(f'{i + 1}/{len(jobs)} {time.time() - t0:.0f}s inst={n_inst}',
                  flush=True)

    print(f'DONE {len(jobs)} images, {n_inst} instances in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
