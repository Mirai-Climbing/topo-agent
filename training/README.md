# Hold detector training

One-shot climbing-hold segmentation (YOLO11-seg) distilled from the Kaggle
Detectron2 weights. Only 19 of 2273 Kaggle images are labeled, so the pipeline
is teacher-student: convert the public Mask R-CNN to torchvision, pseudo-label
the rest, train nano/small/medium.

This directory holds training entry points only. Runtime agent code lives in
`topo_agent/`; the Android app lives outside this repo (`~/et-build/hold-app`).

## Requirements

A CUDA venv with torch (see `requirements-train.txt`). Training env and the
ExecuTorch 0.6 export env differ; see step 4.

## Pipeline

1. Convert VIA annotations (volumes excluded):
   `python via2yolo.py --src ~/data/kaggle-climbing --dst ~/data/yolo-hold-all`
2. Convert the teacher (no detectron2 package needed):
   `python d2teacher.py --ckpt ~/data/kaggle-climbing/model/model_final.pth --out /tmp/tv_hold_rcnn.pth`
3. Pseudo-label the unlabeled images (~25 min on a 3080 Ti):
   `python pseudo_label.py --src ~/data/kaggle-climbing --teacher /tmp/tv_hold_rcnn.pth --dst ~/data/yolo-hold-pseudo`
4. Train (clean split excludes the 8 scoreboard photos; see below):
   `python train.py --size n --gt ~/data/yolo-hold-clean --pseudo ~/data/yolo-hold-pseudo --runs ~/data/runs-hold`
5. Export for the ET 0.6 on-device runtime, then validate the `.pte`:
   `python export_et.py --ckpt ~/data/runs-hold/<run>/weights/best.pt`
   `yolo val task=segment model=<...>/best_executorch_model data=<gt>/data-train.yaml`

## Leakage rule

The 8 scoreboard photos (68 route cases in `~/data/realpairs`) must not appear
in GT train/val or pseudo-label targets. Build the clean split with
`--exclude bh_0003 bh_0074 bh_0075 bh_0457 bh_0502 bh_0518 bh_0530 bh_0917`
(the pseudo script skips VIA-labeled images automatically). The `y11n-seg-b`
weights saw all 8; use them for demos only, never for board numbers.

## Results so far

| Run | Train | Val | Box mAP50 | Mask mAP50 |
|---|---|---|---|---|
| y11n-seg-a | GT 14 | gold 5 | 0.096 | 0.100 |
| y11n-seg-b | GT 14 + pseudo 2254 | gold 5 | 0.753 | 0.700 |

On-device (Galaxy S23, XNNPACK): 160-280 ms per photo.
