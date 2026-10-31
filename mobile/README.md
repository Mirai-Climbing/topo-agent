# Mobile hold detector

On-device climbing-hold segmentation. One YOLO11n-seg ExecuTorch model, two
thin native apps. No network, no API calls.

## Layout

- `android/` — verified app (Galaxy S23): sample photo, camera capture,
  tap-to-inspect, before/after toggle. See its README.
- `ios/` — SwiftUI scaffold porting the same pipeline (needs an Xcode build;
  see its README).
- `assets/hold_yolo.pte` — 40 MB model (YOLO11s-seg, ExecuTorch 0.6 format),
  shared by both apps. Reproduce or retrain with `training/`.

## Model contract

- Input: `[1,3,640,640]` letterboxed RGB / 255, pad value 114.
- Output 0: `[1,37,8400]` — rows 0-3 cx/cy/w/h, row 4 confidence,
  rows 5-36 mask coefficients.
- Output 1: `[1,32,160,160]` mask prototypes; mask = sigmoid(coefs . proto).
- Post: conf >= 0.5, NMS IoU 0.5, mask threshold 0.5.

## Data honesty

This `.pte` (run `yolo11s-seg-clean`) never saw the 8 scoreboard photos:
Box mAP50 0.822 / Mask mAP50 0.735 held-out. An earlier nano build
(`y11n-seg-b`, 0.807/0.751) did train on those photos and is kept for
reference only — see `docs/experiments.md` for the leakage analysis.
