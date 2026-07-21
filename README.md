# TaoSL Gesture ML — YOLO11s Two-Hand Classifiers

FYP-26-S2-14 — **Gaming with Bare Hands (TaoSL)**

This repo trains and deploys **two separate YOLO11s image-classification models**:

| Hand | Role | Classes | Deploy artifact |
|------|------|---------|-----------------|
| **Left** | Movement / dash | 9 | `left_gesture_model.onnx` |
| **Right** | Mudra / skill | 8 | `right_gesture_model.onnx` |

**Pipeline overview**

```
Webcam frame
  → MediaPipe Hands (detect + crop)
  → Left/Right YOLO-cls ONNX (classify crop)
  → gesture_array = [left_id, right_id]   # for Unity / game logic
```

Training is done on **Kaggle GPU** via `gesture_training_export.ipynb`. Live inference runs **locally on CPU** with `onnxruntime` (no PyTorch required at runtime).

---

## What each folder / file is for

```
Manoj FYP ML version/
├── README.md                          ← this file
├── gesture_training_export.ipynb      ← train + val + ONNX export (Kaggle)
├── two_hand_id_datasets.zip           ← cropped hand-image datasets (train/val)
├── gesture_models_export.zip          ← local zip of exports (gitignored if >100MB)
└── gesture_models_export/             ← production-ready models + demos
    ├── left_gesture_model.onnx        ← left classifier (Unity / demo)
    ├── right_gesture_model.onnx       ← right classifier (Unity / demo)
    ├── left_class_names.json          ← ONNX output index → class name
    ├── right_class_names.json
    ├── left_train/                    ← Ultralytics training run (plots, weights, metrics)
    │   ├── weights/best.pt|.onnx
    │   ├── results.csv|.png
    │   └── confusion_matrix*.png
    ├── right_train/                   ← same for right hand
    ├── two_hand_realtime_demo_onnx.py           ← live demo (ONNX + MediaPipe)
    ├── two_hand_realtime_demo_onnx_threaded.py  ← same + threaded camera capture
    ├── two_hand_realtime_demo_profiled.py       ← Ultralytics/.pt profiled demo
    ├── benchmark_onnx_latency.py      ← pure inference latency benchmark
    ├── check_camera_fps.py            ← measure webcam FPS
    ├── inference.py                   ← shared helpers (e.g. square crop)
    └── files/                         ← copies of demo scripts (run from parent folder)
```

| Path | Purpose |
|------|---------|
| `gesture_training_export.ipynb` | End-to-end train → validate → export ONNX → save class JSONs |
| `gesture_models_export/*.onnx` | Models to hand to Unity (Sentis) or run in the Python demos |
| `*_class_names.json` | Maps softmax index → gesture folder name |
| `left_train/` / `right_train/` | Training artifacts for reports (accuracy curves, confusion matrices) |
| `two_hand_id_datasets.zip` | Image folders used for classification training |

> **Important:** Run demos from `gesture_models_export/` (where the `.onnx` files live), not from `files/`.

---

## Environment setup (do this first)

### 1. Create / activate the conda env

```powershell
conda create -n gesture python=3.11 -y
conda activate gesture
```

### 2. Install runtime dependencies (live demo + latency)

```powershell
pip install ultralytics onnxruntime opencv-python mediapipe numpy pillow
```

Notes:

- **Live ONNX demo** needs: `onnxruntime`, `opencv-python`, `mediapipe`, `numpy`
- **Retrain / export locally** also needs: `ultralytics`, `onnx`, `onnxslim`
- Prefer MediaPipe builds that expose `mediapipe.solutions` (classic API used by the demos)

### 3. Confirm ONNX models load

```powershell
cd "gesture_models_export"
python -c "import onnxruntime as ort; s=ort.InferenceSession('left_gesture_model.onnx'); print(s.get_inputs()[0].shape)"
```

Expected input shape: `(1, 3, 224, 224)` — batch × RGB × H × W.

### 4. (Optional) Check camera FPS

```powershell
python check_camera_fps.py
```

---

## How to train the model

Training uses **Ultralytics YOLO11s-cls** (`yolo11s-cls.pt` ImageNet pretrained), transfer-learned on cropped hand images.

### Dataset layout

```
<dataset-root>/
  left/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
  right/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
```

This project’s trained runs used Kaggle paths similar to:

- Left: `dataset_left9_id_crop` (9 classes)
- Right: `dataset_right8_id_crop` (8 classes)

You can also unpack `two_hand_id_datasets.zip` and point training at those folders.

### Recommended: train on Kaggle (GPU)

1. Upload the notebook `gesture_training_export.ipynb` to Kaggle.
2. Add your dataset under **Add Data**.
3. Enable GPU: **Settings → Accelerator → GPU**.
4. Edit the Config cell (`DATASET_ROOT`, epochs, batch, etc.).
5. Run all cells: train left → train right → val → export ONNX → zip download.

### Key training hyperparameters (from the notebook / `args.yaml`)

| Setting | Value | Notes |
|---------|-------|-------|
| Base model | `yolo11s-cls.pt` | YOLO11 **Small** classifier |
| Task | `classify` | Image classification (not detect) |
| Image size | `224` | Must match ONNX export |
| Epochs | `100` (ceiling) | Early stopping `patience=20` |
| Batch | `64` | Reduce if OOM |
| ONNX opset | `15` | Unity Sentis-friendly range |
| Export | `dynamic=False`, `simplify=True` | Fixed `224×224` input |

### Train locally (optional)

```powershell
conda activate gesture
pip install ultralytics onnx onnxslim

yolo classify train model=yolo11s-cls.pt data=path/to/left imgsz=224 epochs=100 patience=20 batch=64 name=left_train
yolo classify train model=yolo11s-cls.pt data=path/to/right imgsz=224 epochs=100 patience=20 batch=64 name=right_train
```

Then export:

```powershell
yolo export model=runs/classify/left_train/weights/best.pt format=onnx imgsz=224 opset=15 simplify=True
yolo export model=runs/classify/right_train/weights/best.pt format=onnx imgsz=224 opset=15 simplify=True
```

Copy the resulting `.onnx` files to `gesture_models_export/` as `left_gesture_model.onnx` / `right_gesture_model.onnx`, and refresh the `*_class_names.json` files from `model.names`.

---

## How to evaluate the model

### During / after training (Ultralytics)

In the notebook (or CLI):

```python
from ultralytics import YOLO
m = YOLO("gesture_models_export/left_train/weights/best.pt")
metrics = m.val(data="path/to/left", imgsz=224)
print(metrics.top1, metrics.top5)
```

Artifacts already in this repo:

| File | What it shows |
|------|----------------|
| `left_train/results.csv` / `results.png` | Loss + top-1 / top-5 over epochs |
| `left_train/confusion_matrix*.png` | Per-class errors |
| Same under `right_train/` | Right-hand metrics |

Approximate final val top-1 from the shipped runs (see last rows of `results.csv`):

- **Left:** ~0.99 top-1
- **Right:** ~0.99 top-1

Always re-validate after any retrain or quantization.

### ONNX sanity check

The notebook includes a random-val-image check with `onnxruntime`. Locally you can also use:

```powershell
cd gesture_models_export
python benchmark_onnx_latency.py
```

That script loads both ONNX models, times inference, and prints a sample prediction.

### Live qualitative eval

Run the realtime demo (next section) and watch:

- Bounding boxes + predicted labels
- Console `gesture_array=[left_id, right_id]`
- `[PROFILE]` lines every N frames (`--profile-every`)

---

## How to run production (live inference)

Production path for this FYP: **MediaPipe detect → ONNX classify → integer IDs**.

### Prerequisites

1. `conda activate gesture`
2. Working webcam
3. Be in `gesture_models_export/` so default model paths resolve

### Standard live demo

```powershell
cd "C:\Users\Fukutaro\Desktop\CSIT321 FYP\Manoj FYP ML version\gesture_models_export"
conda activate gesture

python two_hand_realtime_demo_onnx.py --mirror --crop-margin 0.25 --cls-conf 0.25
```

### Faster capture (recommended)

Overlaps camera wait with compute:

```powershell
python two_hand_realtime_demo_onnx_threaded.py --mirror --crop-margin 0.25 --cls-conf 0.25
```

### Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--mirror` | off | Flip for selfie-style view (common for demos) |
| `--crop-margin` | `0.18` | Expand MediaPipe box before classify |
| `--cls-conf` | `0.45` | Min classifier confidence; else id = `-1` |
| `--det-conf` | `0.35` | MediaPipe hand detection confidence |
| `--every` | `4` | Run classifier every N frames (reuse last pred) |
| `--detect-every` | `2` | Run MediaPipe every N frames (reuse boxes) |
| `--swap-handedness` | off | Swap Left/Right labels if mirror confuses them |
| `--left-onnx` / `--right-onnx` | `*_gesture_model.onnx` | Override model paths |
| `--camera` | `0` | Webcam index |

### Output contract for Unity / game

Each frame prints:

```text
gesture_array = [left_id, right_id]
```

- Valid IDs: see **Gesture class IDs** below  
- Missing / low-confidence hand: `-1`

### Latency reference (CPU `onnxruntime`, measured on this machine)

| Model | Pure `session.run` | Preprocess + infer |
|-------|--------------------|--------------------|
| Left | ~8.1 ms | ~8.5 ms |
| Right | ~8.3 ms | ~8.5 ms |
| Both same frame | — | ~17 ms |

With `--every 4`, average classify cost per display frame is much lower.

---

## Common issues and fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `NO_SUCHFILE ... left_gesture_model.onnx` | Ran demo from `files/` or wrong cwd | `cd gesture_models_export` then run again |
| `No module named 'onnxruntime'` | Wrong / empty env | `conda activate gesture` then `pip install onnxruntime` |
| `mediapipe` has no `solutions` | Incompatible MediaPipe build | Use the `gesture` env; install a classic `mediapipe` with `mp.solutions` |
| Camera won’t open | Wrong index / in use | Try `--camera 1`; close Zoom/Teams; on Windows demos use `CAP_DSHOW` |
| Left/Right swapped | Mirror + MediaPipe handedness | Add `--swap-handedness` and/or toggle `--mirror` |
| Labels flicker | Borderline confidence / motion blur | Raise `--cls-conf`; increase `--crop-margin`; keep temporal majority vote (already in demo) |
| Low FPS | MediaPipe + classify every frame | Use threaded demo; raise `--every` / `--detect-every` |
| ONNX slow on CPU | FP32 YOLO11s | Consider YOLO11n retrain, GPU ORT, or INT8 quantization (same architecture, lower precision — **not** a different ONNX opset “version”) |
| Unity class mismatch | JSON index ≠ game ID for right hand | Use name→ID maps in the demo (or the tables below), don’t assume ONNX index == game ID for Mudras |
| Git push rejected (>100MB) | `gesture_models_export.zip` | Already gitignored; push unpacked `gesture_models_export/` instead |

---

## Gesture class IDs

There are **two related ID spaces**:

1. **ONNX / Ultralytics class index** — softmax argmax from the model (`*_class_names.json`)
2. **Game / Unity ID** — integer emitted in `gesture_array` (after name normalization in the demo)

For the **left** model these usually match. For the **right** model, folder/ONNX order ≠ game ID order — the demo remaps by class **name**.

### Left hand (movement) — 9 classes

| Game ID | Class name | ONNX index (`left_class_names.json`) |
|--------:|------------|--------------------------------------|
| 0 | Left_Stop | 0 (`0_Left_Stop`) |
| 1 | Left_Front | 1 |
| 2 | Left_Behind | 2 |
| 3 | Left_Left | 3 |
| 4 | Left_Right | 4 |
| 5 | Left_FrontDash | 5 |
| 6 | Left_BehindDash | 6 |
| 7 | Left_LeftDash | 7 |
| 8 | Left_RightDash | 8 |
| -1 | none / low confidence | — |

### Right hand (Mudra / skill) — 8 classes

| Game ID | Gesture | ONNX index (`right_class_names.json`) |
|--------:|---------|----------------------------------------|
| 0 | Stop / Fist | 6 (`6_Stop`) |
| 1 | Iron | 4 (`4_Iron`) |
| 2 | Young | 7 (`7_Young`) |
| 3 | Flow | 2 (`2_Flow`) |
| 4 | Burst | 0 (`0_Burst`) |
| 5 | Ground | 3 (`3_Ground`) |
| 6 | Like / Confirm | 5 (`5_Like`) |
| 7 | Dislike / Cancel | 1 (`1_Dislike`) |
| -1 | none / low confidence | — |

ONNX softmax order (for debugging model outputs directly):

| ONNX idx | Folder / JSON name |
|---------:|--------------------|
| 0 | `0_Burst` |
| 1 | `1_Dislike` |
| 2 | `2_Flow` |
| 3 | `3_Ground` |
| 4 | `4_Iron` |
| 5 | `5_Like` |
| 6 | `6_Stop` |
| 7 | `7_Young` |

### Example `gesture_array`

```text
[1, 4]   → left moving Front, right Burst
[0, 6]   → left Stop, right Like
[-1, 1]  → no confident left hand, right Iron
```

---

## Model card (quick reference)

| Item | Detail |
|------|--------|
| Architecture | YOLO11s-cls (Ultralytics) |
| Input | RGB `224×224`, values in `[0, 1]`, NCHW |
| Runtime (laptop demo) | ONNX Runtime CPU |
| Downstream (game) | Unity Sentis / Inference Engine (opset 15) |
| Hand detection | MediaPipe Hands (`max_num_hands=2`) |
| Repo | https://github.com/fukutarosie/taoSL-dinov2-model-v2 |

---

## Suggested workflow checklist

1. ✅ Set up `gesture` conda env  
2. ✅ Unzip / verify `left_gesture_model.onnx` + `right_gesture_model.onnx`  
3. ✅ `python benchmark_onnx_latency.py`  
4. ✅ `python two_hand_realtime_demo_onnx_threaded.py --mirror`  
5. ✅ Confirm `gesture_array` IDs match Unity expectations  
6. ✅ Hand `.onnx` + `*_class_names.json` (and this ID table) to the Unity teammate  
