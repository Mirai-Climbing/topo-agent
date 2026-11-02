# Hold Detector (iOS scaffold)

SwiftUI port of the Android hold detector. Built and run on an iPhone with
Xcode and ExecuTorch 0.6 (`swiftpm-0.6.0`): 14 holds in about 220 ms on a gym
photo (debug build).

## Project setup (Xcode 15+, iOS 17+)

1. New App project (SwiftUI, Swift) named `HoldDetector`.
2. Add the ExecuTorch Swift package:
   `https://github.com/pytorch/executorch`, branch `swiftpm-0.6.0` (matches the
   `.pte` exporter; see [ExecuTorch iOS docs](https://docs.pytorch.org/executorch/stable/using-executorch-ios.html)).
   Link these products to the app target:
   - `executorch`
   - `backend_xnnpack`
   - `kernels_portable` only. With only `kernels_optimized`, many ops are
     undefined at link time; with both, the app crashes at launch (SIGABRT),
     probably from duplicate kernel registration.
3. Build Settings → Other Linker Flags: `-all_load` and `-lc++`.
4. Build Phases → Link Binary With Libraries: add `Accelerate.framework`
   (for `sgemm_` / `dgemm_`).
5. Copy `HoldDetector/*.swift` into the project.
6. Add `mobile/assets/hold_yolo.pte` to the app bundle (target membership on).
7. Add the camera and photo library usage descriptions from
   `HoldDetector/Info.plist` to the target's Info.

## What the code does

- `YoloSeg.swift`: letterbox preprocess, `Module.forward`, NMS, per-instance
  masks — a direct port of the verified Android pipeline:
  - input `[1,3,640,640]`, pad 114, RGB / 255
  - output 0 `[1,37,8400]` (cx/cy/w/h, conf, 32 coefs), output 1 `[1,32,160,160]`
  - conf 0.5, NMS IoU 0.5, mask sigmoid threshold 0.5
- `ContentView.swift`: photo library + camera capture, before/after toggle,
  tap-to-inspect (smallest containing box wins).

## Notes

- The code targets the ExecuTorch 0.6 Swift API: `Module(filePath:)` does not
  throw, the input is `Tensor(data:shape:dataType:)` with an `[NSNumber]` shape,
  and outputs are read with `value.tensor` and `tensor.bytes` (the `floats(_:)`
  helper). Other versions may need changes at these call sites.
- First run on simulator is fine for UI; measure inference on a real device.
- Retrain/re-export with `training/` (`export_et.py`).
