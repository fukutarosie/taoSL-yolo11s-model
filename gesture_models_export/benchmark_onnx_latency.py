import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(img_bgr, size=224):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    arr = img.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr


def benchmark_model(onnx_path, class_names_path, label, sample_image_path=None,
                     img_size=224, warmup=15, iterations=200):
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {label}  ({onnx_path})")
    print(f"{'=' * 60}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    with open(class_names_path) as f:
        raw = json.load(f)
    class_names = {int(k): v for k, v in raw.items()}

    # Use a real sample image if given, otherwise synthetic noise of the right shape.
    # Model architecture cost doesn't depend on image content, only shape - so
    # this is a fair timing test either way, but a real image lets us sanity
    # check the prediction alongside the timing.
    if sample_image_path and Path(sample_image_path).exists():
        img_bgr = cv2.imread(str(sample_image_path))
        source = f"real image ({sample_image_path})"
    else:
        img_bgr = (np.random.rand(img_size, img_size, 3) * 255).astype(np.uint8)
        source = "synthetic noise (no sample image given)"

    print(f"Test input source: {source}")

    # --- Warmup (first few runs are always slower - session init, cache warming) ---
    for _ in range(warmup):
        x = preprocess(img_bgr, img_size)
        session.run(None, {input_name: x})

    # --- Timed: preprocessing + inference combined (matches real predict() cost) ---
    full_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        x = preprocess(img_bgr, img_size)
        outputs = session.run(None, {input_name: x})
        full_times.append((time.perf_counter() - t0) * 1000.0)

    # --- Timed: pure model inference only (pre-built input, no preprocessing) ---
    x_fixed = preprocess(img_bgr, img_size)
    model_only_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: x_fixed})
        model_only_times.append((time.perf_counter() - t0) * 1000.0)

    probs = outputs[0][0]
    top_idx = int(np.argmax(probs))
    pred_label = class_names[top_idx]
    pred_conf = float(probs[top_idx])

    def stats(times, name):
        arr = np.array(times)
        print(f"\n  [{name}]")
        print(f"    mean:   {arr.mean():6.3f} ms")
        print(f"    median: {np.median(arr):6.3f} ms")
        print(f"    std:    {arr.std():6.3f} ms")
        print(f"    min:    {arr.min():6.3f} ms")
        print(f"    max:    {arr.max():6.3f} ms")
        print(f"    p95:    {np.percentile(arr, 95):6.3f} ms")

    stats(model_only_times, "Model inference only (session.run)")
    stats(full_times, "Preprocessing + inference (real predict() cost)")

    print(f"\n  Sample prediction: {pred_label}  (confidence: {pred_conf:.3f})")

    return {
        "model_only_mean_ms": float(np.mean(model_only_times)),
        "full_mean_ms": float(np.mean(full_times)),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark pure ONNX inference latency.")
    parser.add_argument("--left-onnx", default="left_gesture_model.onnx")
    parser.add_argument("--right-onnx", default="right_gesture_model.onnx")
    parser.add_argument("--left-names", default="left_class_names.json")
    parser.add_argument("--right-names", default="right_class_names.json")
    parser.add_argument("--left-sample", default=None, help="Optional path to a real left-hand crop image")
    parser.add_argument("--right-sample", default=None, help="Optional path to a real right-hand crop image")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=15)
    args = parser.parse_args()

    left_results = benchmark_model(
        args.left_onnx, args.left_names, "LEFT (movement) model",
        sample_image_path=args.left_sample,
        warmup=args.warmup, iterations=args.iterations,
    )
    right_results = benchmark_model(
        args.right_onnx, args.right_names, "RIGHT (Mudra) model",
        sample_image_path=args.right_sample,
        warmup=args.warmup, iterations=args.iterations,
    )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Left  model - pure inference: {left_results['model_only_mean_ms']:.3f} ms  |  full predict(): {left_results['full_mean_ms']:.3f} ms")
    print(f"Right model - pure inference: {right_results['model_only_mean_ms']:.3f} ms  |  full predict(): {right_results['full_mean_ms']:.3f} ms")
    total_full = left_results['full_mean_ms'] + right_results['full_mean_ms']
    print(f"\nIf both hands classified every frame: ~{total_full:.2f} ms combined")
    print(f"(In your live demo, classification is skipped most frames via --every, so real average cost is lower)")
