"""Train YOLO11-seg (n/s/m) on GT + pseudo-labeled climbing holds.

Hyperparameters reproduce the y11n-seg-b run (Mask mAP50 0.70 on gold val).
Batch sizes are sized for a 12 GB GPU; shrink --batch if OOM on s/m.

Usage:
    python train.py --size n --gt ~/data/yolo-hold-clean \
        --pseudo ~/data/yolo-hold-pseudo --runs ~/data/runs-hold
    python train.py --size s --batch 8 ...
    python train.py --size m --batch 8 --epochs 120 ...
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

BATCH_DEFAULT = {'n': 16, 's': 16, 'm': 8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', choices=['n', 's', 'm'], default='n')
    ap.add_argument('--gt', required=True, help='YOLO dataset with images/{train,val}')
    ap.add_argument('--pseudo', default='',
                    help='optional pseudo dataset root (uses images/train)')
    ap.add_argument('--runs', required=True, help='ultralytics project dir')
    ap.add_argument('--name', default='', help='run name (default y11{size}-seg-clean)')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=-1)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    gt = Path(args.gt)
    data_yaml = gt / 'data-train.yaml'
    trains = [str(gt / 'images' / 'train')]
    if args.pseudo:
        trains.append(str(Path(args.pseudo) / 'images' / 'train'))
    data_yaml.write_text(
        'train:\n' + ''.join(f'  - {t}\n' for t in trains) +
        f'val: {gt / "images" / "val"}\nnames:\n  0: hold\n')

    batch = args.batch if args.batch > 0 else BATCH_DEFAULT[args.size]
    model = YOLO(f'yolo11{args.size}-seg.pt')
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=4,
        project=args.runs,
        name=args.name or f'yolo11{args.size}-seg-clean',
        exist_ok=True,
        patience=40,
        save_period=25,
        cos_lr=True,
        warmup_epochs=5,
        close_mosaic=15,
        copy_paste=0.2,
        mixup=0.15,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=15,
        seed=args.seed,
        verbose=True,
    )


if __name__ == '__main__':
    main()
