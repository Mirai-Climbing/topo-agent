# Hold Detector (Android)

One-shot climbing-hold segmentation on the phone. A YOLO11n-seg model
(ExecuTorch `.pte`) runs once per photo; every hold mask is drawn immediately.
No network, no API calls.

## Features

- Sample photo (`gym.jpg` asset, optional) detected at launch
- Camera capture with EXIF rotation handling
- Tap a hold to inspect (index + confidence)
- Before/after toggle

## Build

Prerequisites: JDK 17, Android SDK 34+, `org.pytorch:executorch-android:0.6.0`
from Maven (declared in `app/build.gradle`).

```bash
cd mobile/android
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Model

Push `mobile/assets/hold_yolo.pte` to the app's files dir (the app shows the
exact path when it is missing):

```bash
adb push ../assets/hold_yolo.pte \
  /sdcard/Android/data/io.github.johnrocky.etsamdemo/files/hold_yolo.pte
```

The `.pte` must match the ExecuTorch 0.6 runtime. To retrain or re-export, see
`training/` (export with `export_et.py`).

## Sample photo (optional)

Drop any climbing photo at `app/src/main/assets/gym.jpg` to have it detected
at launch. Without it the app starts in camera-only mode. Real gym photos are
never committed; see the repo `.gitignore`.

## Measured on Galaxy S23

- Inference: 160-370 ms per photo (XNNPACK)
- Render (Kotlin NMS + mask assembly): 15-340 ms for 40-100 holds
