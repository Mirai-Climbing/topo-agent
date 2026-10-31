"""Export a trained YOLO-seg checkpoint to ExecuTorch .pte.

Targets the on-device runtime (ExecuTorch 0.6, torch 2.7): the ultralytics
exporter's torch>=2.9 gate is bypassed, and the bundled flatc is prepended to
PATH when the system one is missing.

Usage:
    python export_et.py --ckpt ~/data/runs-hold/.../best.pt
"""
import argparse
import os
import shutil
import sys

import ultralytics.engine.exporter as E

E.TORCH_2_9 = True  # ET 0.6 pairs with torch 2.7; bypass the version gate

from ultralytics import YOLO  # noqa: E402


def ensure_flatc():
    if shutil.which('flatc'):
        return
    try:
        import executorch
        bundled = os.path.join(os.path.dirname(executorch.__file__),
                               'data', 'bin')
        if os.path.exists(os.path.join(bundled, 'flatc')):
            os.environ['PATH'] = bundled + os.pathsep + os.environ['PATH']
            print(f'using bundled flatc from {bundled}', file=sys.stderr)
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()
    ensure_flatc()
    model = YOLO(args.ckpt)
    print('exported:', model.export(format='executorch'))


if __name__ == '__main__':
    main()
