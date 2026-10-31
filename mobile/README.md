# Mobile hold detector

On-device climbing-hold segmentation. One YOLO11n-seg ExecuTorch model, two
thin native apps. No network, no API calls.

## Layout

- `android/` — verified app (Galaxy S23): sample photo, camera capture,
  tap-to-inspect, before/after toggle. See its README.
- `ios/` — SwiftUI scaffold porting the same pipeline (needs an Xcode build;
  see its README).
- `assets/hold_yolo.pte` — 11 MB model, ExecuTorch 0.6 format, shared by both
  apps. Reproduce or retrain with `training/`.

## Model contract

- Input: `[1,3,640,640]` letterboxed RGB / 255, pad value 114.
- Output 0: `[1,37,8400]` — rows 0-3 cx/cy/w/h, row 4 confidence,
  rows 5-36 mask coefficients.
- Output 1: `[1,32,160,160]` mask prototypes; mask = sigmoid(coefs . proto).
- Post: conf >= 0.5, NMS IoU 0.5, mask threshold 0.5.

## Data honesty

This `.pte` (run `y11n-seg-b`) saw the 8 scoreboard photos during training
(Mask mAP50 0.70, optimistic). It is a demo/model-quality build, not a board
entry. For board numbers, train the clean split (`training/README.md`,
leakage rule) and re-export.
