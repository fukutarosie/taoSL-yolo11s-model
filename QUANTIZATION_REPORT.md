# Quantization Report — YOLO11s Gesture Classifiers

**Date:** 21 July 2026  
**Method:** ONNX Runtime static INT8 quantization (QDQ)  
**Script:** `gesture_models_export/quantize_onnx_int8.py`  
**Goal:** Reduce CPU inference latency without retraining

---

## What changed

| | FP32 (baseline) | INT8 (quantized) |
|--|-----------------|------------------|
| Numeric format | 32-bit float | 8-bit integer (+ scale) |
| Files | `left_gesture_model.onnx`, `right_gesture_model.onnx` | `left_gesture_model_int8.onnx`, `right_gesture_model_int8.onnx` |
| Architecture | YOLO11s-cls | **Same** (weights/activations compressed) |
| ONNX opset | 15 | 15 (not a version change — a precision change) |

Pipeline:

```text
Train (.pt FP32)
  → Export ONNX (FP32)
  → Calibrate on hand-crop images from two_hand_id_datasets.zip
  → Static INT8 quantize
  → Compare size, top-1 accuracy, latency vs FP32
```

---

## Measurement setup

| Item | Setting |
|------|---------|
| Runtime | ONNX Runtime CPU (`CPUExecutionProvider`) |
| Calibration | Train split, 8 images per class (`MinMax`) |
| Accuracy eval | Val split sample (20 images per class) |
| Latency | Pure `session.run`, warmup 15, timed iterations 100 |
| Input | `1×3×224×224`, RGB, values in `[0, 1]` |

Re-run:

```powershell
cd gesture_models_export
conda activate gesture
pip install onnx   # if needed
python quantize_onnx_int8.py --calib-per-class 8 --eval-per-class 20
```

---

## Results (this machine)

| Model | Size FP32 → INT8 | Top-1 FP32 → INT8 | FP32↔INT8 agreement | Mean latency FP32 → INT8 | Speedup |
|-------|------------------|-------------------|---------------------|--------------------------|---------|
| **Left** (9 classes) | 20.8 → **5.5 MB** | 100.0% → **99.4%** | 99.4% | ~14.0 → **~8.2 ms** | **~1.71×** |
| **Right** (8 classes) | 20.8 → **5.5 MB** | 99.4% → **99.4%** | 100.0% | ~9.6 → **~6.6 ms** | **~1.46×** |

### Takeaways

1. **Latency decreased** on CPU for both models (~1.5–1.7×).
2. **File size** fell to ~26% of FP32 (~21 MB → ~5.5 MB each).
3. **Accuracy trade-off was small** on the val sample (left −0.6 pp; right unchanged).
4. Quantization is a valid **CPU optimisation** when GPU is unavailable and retraining to YOLO11n is not required.

---

## How to verify latency reduction yourself

**A. Script summary (recommended)**  
`python quantize_onnx_int8.py` prints FP32 vs INT8 size, top-1, agreement, and ms speedup.

**B. Side-by-side benchmark**

```powershell
python benchmark_onnx_latency.py --left-onnx left_gesture_model.onnx --right-onnx right_gesture_model.onnx
python benchmark_onnx_latency.py --left-onnx left_gesture_model_int8.onnx --right-onnx right_gesture_model_int8.onnx
```

Compare **Model inference only → mean**.

**C. Live demo profile**

```powershell
python two_hand_realtime_demo_onnx_threaded.py --mirror --profile-every 30 `
  --left-onnx left_gesture_model_int8.onnx `
  --right-onnx right_gesture_model_int8.onnx
```

Watch `[PROFILE] classify=...ms` (camera/MediaPipe may still dominate total FPS).

---

## Deploy notes

- **Python laptop demo:** prefer INT8 for lower classify latency.
- **Unity Sentis:** confirm QDQ INT8 import works; if not, ship **FP32** ONNX to Unity and keep INT8 for local testing.
- Class IDs / `*_class_names.json` are **unchanged** between FP32 and INT8.
