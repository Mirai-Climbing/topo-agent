"""VIA (Kaggle climbing) annotations -> YOLO-seg dataset.

Single class (hold; volumes excluded). Optionally splits out a clean subset
that excludes eval photos, for leakage-free training.

Usage:
    python via2yolo.py --src ~/data/kaggle-climbing --dst ~/data/yolo-hold-all
    python via2yolo.py --src ... --dst ... --exclude bh_0003 bh_0074 ...
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image

SOURCES = ['bh', 'bh-phone', 'sm']


def load_items(src: Path):
    items = []  # (image_path, [(xs, ys)], tag)
    for name in SOURCES:
        d = json.loads((src / f'{name}-annotation.json').read_text())
        for m in d['_via_img_metadata'].values():
            regs = []
            for r in m['regions']:
                sa = r['shape_attributes']
                if sa.get('name') != 'polygon':
                    continue
                if r.get('region_attributes', {}).get('hold_type', 'hold') != 'hold':
                    continue
                xs, ys = sa['all_points_x'], sa['all_points_y']
                if len(xs) < 3:
                    continue
                regs.append((xs, ys))
            if regs:
                tag = name.replace('-', '')
                items.append((src / name / m['filename'], regs, tag))
    return items


def write_split(items, dst: Path, split: str):
    imgdir = dst / 'images' / split
    lbldir = dst / 'labels' / split
    imgdir.mkdir(parents=True, exist_ok=True)
    lbldir.mkdir(parents=True, exist_ok=True)
    for src_img, regs, tag in items:
        stem = f'{tag}_{src_img.stem}'
        link = imgdir / f'{stem}.jpg'
        if not link.exists():
            link.symlink_to(src_img)
        w, h = Image.open(src_img).size
        lines = []
        for xs, ys in regs:
            pts = ' '.join(f'{x / w:.6f} {y / h:.6f}' for x, y in zip(xs, ys))
            lines.append(f'0 {pts}')
        (lbldir / f'{stem}.txt').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Kaggle dataset root')
    ap.add_argument('--dst', required=True, help='YOLO dataset root to create')
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='stems (e.g. bh_0917) to leave out entirely')
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    src, dst = Path(args.src), Path(args.dst)
    items = [it for it in load_items(src)
             if f'{it[2]}_{it[0].stem}' not in args.exclude]
    print(f'labeled images (after exclusions): {len(items)}')

    by_src = {}
    for it in items:
        by_src.setdefault(it[2], []).append(it)
    train, val = [], []
    for lst in by_src.values():
        random.shuffle(lst)
        n_val = max(1, int(len(lst) * args.val_frac)) if len(lst) > 1 else 0
        val += lst[:n_val]
        train += lst[n_val:]
    print(f'train: {len(train)} val: {len(val)}')

    write_split(train, dst, 'train')
    write_split(val, dst, 'val')
    (dst / 'data.yaml').write_text(
        f'path: {dst}\ntrain: images/train\nval: images/val\n'
        f'names:\n  0: hold\n')
    print(f'wrote {dst}')


if __name__ == '__main__':
    main()
