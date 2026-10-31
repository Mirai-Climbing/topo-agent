# Hold Detector (iOS scaffold)

SwiftUI port of the Android hold detector. Sources are complete but were
written without Xcode access, so the teammate building this should expect
small API adjustments against the installed ExecuTorch version.

## Project setup (Xcode 15+, iOS 17+)

1. New App project (SwiftUI, Swift) named `HoldDetector`.
2. Add the ExecuTorch Swift package:
   `https://github.com/pytorch/executorch` — use a release matching the
   `.pte` exporter (0.6; see [ExecuTorch iOS docs](https://docs.pytorch.org/executorch/stable/using-executorch-ios.html)).
3. Copy `HoldDetector/*.swift` into the project.
4. Add `mobile/assets/hold_yolo.pte` to the app bundle (target membership on).
5. Merge `HoldDetector/Info.plist` camera/photo keys into the target's Info.

## What the code does

- `YoloSeg.swift`: letterbox preprocess, `Module.forward`, NMS, per-instance
  masks — a direct port of the verified Android pipeline:
  - input `[1,3,640,640]`, pad 114, RGB / 255
  - output 0 `[1,37,8400]` (cx/cy/w/h, conf, 32 coefs), output 1 `[1,32,160,160]`
  - conf 0.5, NMS IoU 0.5, mask sigmoid threshold 0.5
- `ContentView.swift`: photo library + camera capture, before/after toggle,
  tap-to-inspect (smallest containing box wins).

## Notes

- If `Module`/`Tensor`/`Value` names differ in your ExecuTorch version, adapt
  the three call sites in `YoloSeg.swift` (`init`, `preprocess` tensor build,
  `detect` forward + `data()` reads).
- First run on simulator is fine for UI; measure inference on a real device.
- Retrain/re-export with `training/` (`export_et.py`).
