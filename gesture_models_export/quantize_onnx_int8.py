"""
Static INT8 quantization for TaoSL YOLO-cls ONNX models (ONNX Runtime).

Rationale
---------
Your shipped models are FP32 ONNX (~8 ms/hand on CPU). Static INT8 quantization
keeps the same graph / opset, but stores most weights and activations as 8-bit
integers so CPU inference is typically faster and the file is smaller.

This is NOT changing the ONNX "version" (opset). It is a post-export precision
optimization:

  FP32 .onnx  --(calibrate on hand crops)-->  INT8 .onnx

Steps this script runs
----------------------
1. Sample calibration images from two_hand_id_datasets.zip (train/val crops)
2. Optional ORT quant_pre_process (shape inference / model prep)
3. Static quantization (QDQ, QUInt8 activations, QInt8 weights)
4. Top-1 accuracy on a val subset: FP32 vs INT8
5. Latency benchmark: FP32 vs INT8

Usage (from gesture_models_export/)
-----------------------------------
  conda activate gesture
  pip install onnx   # once, if missing

  python quantize_onnx_int8.py
  python quantize_onnx_int8.py --calib-per-class 8 --eval-per-class 20

Then run the live demo with the INT8 files:
  python two_hand_realtime_demo_onnx.py --mirror ^
    --left-onnx left_gesture_model_int8.onnx ^
    --right-onnx right_gesture_model_int8.onnx
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_bgr(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """Match the live demo / benchmark preprocessing."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    arr = img.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0)


def list_zip_images(zf: zipfile.ZipFile, dataset_root: str, split: str) -> dict[str, list[str]]:
    """Return {class_folder: [zip member paths]} for a dataset split."""
    prefix = f"{dataset_root}/{split}/"
    by_class: dict[str, list[str]] = {}
    for name in zf.namelist():
        if not name.startswith(prefix):
            continue
        rel = name[len(prefix) :]
        parts = rel.split("/")
        if len(parts) != 2:
            continue
        class_name, filename = parts
        if Path(filename).suffix.lower() not in IMG_EXTS:
            continue
        by_class.setdefault(class_name, []).append(name)
    return by_class


def sample_members(
    by_class: dict[str, list[str]],
    per_class: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    picked: list[str] = []
    for class_name in sorted(by_class):
        files = list(by_class[class_name])
        rng.shuffle(files)
        picked.extend(files[:per_class])
    rng.shuffle(picked)
    return picked


def extract_members(zf: zipfile.ZipFile, members: list[str], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for member in members:
        data = zf.read(member)
        dest = out_dir / Path(member).name
        # Avoid collisions across classes by prefixing with parent folder name
        class_name = Path(member).parts[-2]
        dest = out_dir / f"{class_name}__{Path(member).name}"
        dest.write_bytes(data)
        paths.append(dest)
    return paths


class CropCalibrationDataReader(CalibrationDataReader):
    """Feeds preprocessed hand-crop tensors to ORT static quantization."""

    def __init__(self, image_paths: list[Path], input_name: str, img_size: int = 224):
        self.input_name = input_name
        self.img_size = img_size
        self.image_paths = image_paths
        self._enum = None

    def get_next(self):
        if self._enum is None:
            self._enum = self._generator()
        return next(self._enum, None)

    def _generator(self):
        for path in self.image_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            yield {self.input_name: preprocess_bgr(img, self.img_size)}


def get_input_name(onnx_path: Path) -> str:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session.get_inputs()[0].name


def quantize_model(
    fp32_path: Path,
    int8_path: Path,
    calib_images: list[Path],
    img_size: int,
    calib_method: CalibrationMethod,
) -> Path:
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ort_quant_") as tmp:
        prepared = Path(tmp) / f"{fp32_path.stem}_prep.onnx"
        print(f"  [prep] quant_pre_process -> {prepared.name}")
        quant_pre_process(
            input_model_path=str(fp32_path),
            output_model_path=str(prepared),
            skip_optimization=False,
        )

        input_name = get_input_name(prepared)
        reader = CropCalibrationDataReader(calib_images, input_name, img_size)

        print(f"  [quant] static INT8  calib_images={len(calib_images)}  method={calib_method.name}")
        quantize_static(
            model_input=str(prepared),
            model_output=str(int8_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=calib_method,
            per_channel=True,
            reduce_range=False,
        )

    return int8_path


def evaluate_top1(
    onnx_path: Path,
    image_paths: list[Path],
    class_names: dict[int, str],
    img_size: int,
) -> dict:
    """
    Ground-truth class is inferred from the filename prefix we wrote:
      {class_folder}__{original_name}.jpg
    """
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Map folder/class name -> index using the JSON names
    name_to_idx = {v: int(k) for k, v in class_names.items()}

    correct = 0
    total = 0
    agree_skipped = 0

    for path in image_paths:
        stem = path.name
        class_folder = stem.split("__", 1)[0]
        if class_folder not in name_to_idx:
            agree_skipped += 1
            continue
        gt = name_to_idx[class_folder]
        img = cv2.imread(str(path))
        if img is None:
            continue
        x = preprocess_bgr(img, img_size)
        probs = session.run(None, {input_name: x})[0][0]
        pred = int(np.argmax(probs))
        correct += int(pred == gt)
        total += 1

    acc = (correct / total) if total else 0.0
    return {
        "correct": correct,
        "total": total,
        "top1": acc,
        "skipped_unknown_class": agree_skipped,
    }


def compare_agreement(
    fp32_path: Path,
    int8_path: Path,
    image_paths: list[Path],
    img_size: int,
) -> dict:
    """How often INT8 matches FP32 argmax (independent of labels)."""
    s_fp = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    s_i8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    in_fp = s_fp.get_inputs()[0].name
    in_i8 = s_i8.get_inputs()[0].name

    match = 0
    total = 0
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        x = preprocess_bgr(img, img_size)
        p_fp = int(np.argmax(s_fp.run(None, {in_fp: x})[0][0]))
        p_i8 = int(np.argmax(s_i8.run(None, {in_i8: x})[0][0]))
        match += int(p_fp == p_i8)
        total += 1
    return {"match": match, "total": total, "agree": (match / total) if total else 0.0}


def benchmark_latency(onnx_path: Path, sample_bgr: np.ndarray, img_size: int, warmup: int, iters: int) -> dict:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    x = preprocess_bgr(sample_bgr, img_size)

    for _ in range(warmup):
        session.run(None, {input_name: x})

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: x})
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def load_class_names(path: Path) -> dict[int, str]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def process_side(
    label: str,
    fp32_path: Path,
    int8_path: Path,
    names_path: Path,
    zip_path: Path,
    dataset_root: str,
    calib_split: str,
    eval_split: str,
    calib_per_class: int,
    eval_per_class: int,
    img_size: int,
    seed: int,
    calib_method: CalibrationMethod,
    warmup: int,
    iters: int,
    work_dir: Path,
) -> dict:
    print(f"\n{'=' * 64}")
    print(f"{label}")
    print(f"{'=' * 64}")
    print(f"  FP32 : {fp32_path}")
    print(f"  INT8 : {int8_path}")

    if not fp32_path.exists():
        raise SystemExit(f"Missing FP32 model: {fp32_path}")

    class_names = load_class_names(names_path)

    with zipfile.ZipFile(zip_path) as zf:
        calib_by = list_zip_images(zf, dataset_root, calib_split)
        eval_by = list_zip_images(zf, dataset_root, eval_split)
        if not calib_by:
            raise SystemExit(f"No calibration images found under {dataset_root}/{calib_split}")
        if not eval_by:
            raise SystemExit(f"No eval images found under {dataset_root}/{eval_split}")

        print(f"  Classes in calib split: {len(calib_by)}")
        calib_members = sample_members(calib_by, calib_per_class, seed)
        eval_members = sample_members(eval_by, eval_per_class, seed + 1)

        calib_dir = work_dir / f"{label.lower()}_calib"
        eval_dir = work_dir / f"{label.lower()}_eval"
        calib_images = extract_members(zf, calib_members, calib_dir)
        eval_images = extract_members(zf, eval_members, eval_dir)

    print(f"  Calibration images: {len(calib_images)}")
    print(f"  Eval images:        {len(eval_images)}")

    quantize_model(fp32_path, int8_path, calib_images, img_size, calib_method)

    fp32_size = fp32_path.stat().st_size / (1024 * 1024)
    int8_size = int8_path.stat().st_size / (1024 * 1024)
    print(f"  Size FP32={fp32_size:.2f} MB  INT8={int8_size:.2f} MB  "
          f"({100.0 * int8_size / fp32_size:.1f}% of FP32)")

    print("  Evaluating top-1 accuracy on val sample...")
    acc_fp32 = evaluate_top1(fp32_path, eval_images, class_names, img_size)
    acc_int8 = evaluate_top1(int8_path, eval_images, class_names, img_size)
    agree = compare_agreement(fp32_path, int8_path, eval_images, img_size)

    print(f"  FP32 top-1: {acc_fp32['top1']*100:.2f}%  ({acc_fp32['correct']}/{acc_fp32['total']})")
    print(f"  INT8 top-1: {acc_int8['top1']*100:.2f}%  ({acc_int8['correct']}/{acc_int8['total']})")
    print(f"  INT8 vs FP32 agreement: {agree['agree']*100:.2f}%  ({agree['match']}/{agree['total']})")

    sample = cv2.imread(str(eval_images[0]))
    print("  Benchmarking latency (CPU)...")
    lat_fp32 = benchmark_latency(fp32_path, sample, img_size, warmup, iters)
    lat_int8 = benchmark_latency(int8_path, sample, img_size, warmup, iters)
    speedup = lat_fp32["mean_ms"] / lat_int8["mean_ms"] if lat_int8["mean_ms"] > 0 else float("inf")

    print(f"  FP32 mean: {lat_fp32['mean_ms']:.3f} ms")
    print(f"  INT8 mean: {lat_int8['mean_ms']:.3f} ms")
    print(f"  Speedup:   {speedup:.2f}x")

    return {
        "label": label,
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path),
        "fp32_mb": fp32_size,
        "int8_mb": int8_size,
        "acc_fp32": acc_fp32,
        "acc_int8": acc_int8,
        "agree": agree,
        "lat_fp32": lat_fp32,
        "lat_int8": lat_int8,
        "speedup": speedup,
    }


def parse_args():
    here = Path(__file__).resolve().parent
    repo = here.parent

    parser = argparse.ArgumentParser(
        description="Quantize left/right YOLO-cls ONNX models to INT8 with ORT static quantization."
    )
    parser.add_argument("--left-onnx", type=Path, default=here / "left_gesture_model.onnx")
    parser.add_argument("--right-onnx", type=Path, default=here / "right_gesture_model.onnx")
    parser.add_argument("--left-int8", type=Path, default=here / "left_gesture_model_int8.onnx")
    parser.add_argument("--right-int8", type=Path, default=here / "right_gesture_model_int8.onnx")
    parser.add_argument("--left-names", type=Path, default=here / "left_class_names.json")
    parser.add_argument("--right-names", type=Path, default=here / "right_class_names.json")
    parser.add_argument(
        "--dataset-zip",
        type=Path,
        default=repo / "two_hand_id_datasets.zip",
        help="Zip with dataset_left9_id_crop / dataset_right8_id_crop",
    )
    parser.add_argument("--left-root", default="dataset_left9_id_crop")
    parser.add_argument("--right-root", default="dataset_right8_id_crop")
    parser.add_argument("--calib-split", default="train", help="Split used for calibration crops")
    parser.add_argument("--eval-split", default="val", help="Split used for accuracy check")
    parser.add_argument("--calib-per-class", type=int, default=8, help="Calibration images per class")
    parser.add_argument("--eval-per-class", type=int, default=25, help="Eval images per class")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calib-method",
        choices=["minmax", "entropy", "percentile"],
        default="minmax",
        help="ORT calibration method (minmax is fast/stable for cls models)",
    )
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument(
        "--keep-calib-dir",
        type=Path,
        default=None,
        help="If set, keep extracted calib/eval images here instead of a temp dir",
    )
    parser.add_argument("--left-only", action="store_true")
    parser.add_argument("--right-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    method_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    calib_method = method_map[args.calib_method]

    if not args.dataset_zip.exists():
        raise SystemExit(
            f"Dataset zip not found: {args.dataset_zip}\n"
            "Place two_hand_id_datasets.zip next to gesture_models_export/ "
            "or pass --dataset-zip PATH"
        )

    do_left = not args.right_only
    do_right = not args.left_only

    results = []
    if args.keep_calib_dir:
        work_dir = args.keep_calib_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        tmp = tempfile.TemporaryDirectory(prefix="taosl_int8_calib_")
        work_dir = Path(tmp.name)
        cleanup = tmp

    try:
        if do_left:
            results.append(
                process_side(
                    "LEFT",
                    args.left_onnx,
                    args.left_int8,
                    args.left_names,
                    args.dataset_zip,
                    args.left_root,
                    args.calib_split,
                    args.eval_split,
                    args.calib_per_class,
                    args.eval_per_class,
                    args.img_size,
                    args.seed,
                    calib_method,
                    args.warmup,
                    args.iterations,
                    work_dir,
                )
            )
        if do_right:
            results.append(
                process_side(
                    "RIGHT",
                    args.right_onnx,
                    args.right_int8,
                    args.right_names,
                    args.dataset_zip,
                    args.right_root,
                    args.calib_split,
                    args.eval_split,
                    args.calib_per_class,
                    args.eval_per_class,
                    args.img_size,
                    args.seed + 17,
                    calib_method,
                    args.warmup,
                    args.iterations,
                    work_dir,
                )
            )
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    print(f"\n{'=' * 64}")
    print("SUMMARY")
    print(f"{'=' * 64}")
    for r in results:
        drop = (r["acc_fp32"]["top1"] - r["acc_int8"]["top1"]) * 100
        print(
            f"{r['label']:5s}  "
            f"size {r['fp32_mb']:.1f} -> {r['int8_mb']:.1f} MB  |  "
            f"top1 {r['acc_fp32']['top1']*100:.1f}% -> {r['acc_int8']['top1']*100:.1f}% "
            f"(drop {drop:+.1f} pp)  |  "
            f"{r['lat_fp32']['mean_ms']:.2f} -> {r['lat_int8']['mean_ms']:.2f} ms  "
            f"({r['speedup']:.2f}x)  |  "
            f"agree {r['agree']['agree']*100:.1f}%"
        )
        print(f"       wrote {r['int8_path']}")

    print(
        "\nLive demo with INT8 models:\n"
        "  python two_hand_realtime_demo_onnx_threaded.py --mirror "
        "--left-onnx left_gesture_model_int8.onnx "
        "--right-onnx right_gesture_model_int8.onnx\n"
    )

    # Machine-readable summary for reports / CI
    report_path = Path(__file__).resolve().parent / "quantization_last_run.json"
    serializable = []
    for r in results:
        serializable.append(
            {
                "label": r["label"],
                "fp32_path": r["fp32_path"],
                "int8_path": r["int8_path"],
                "fp32_mb": round(r["fp32_mb"], 3),
                "int8_mb": round(r["int8_mb"], 3),
                "top1_fp32": round(r["acc_fp32"]["top1"], 4),
                "top1_int8": round(r["acc_int8"]["top1"], 4),
                "agreement": round(r["agree"]["agree"], 4),
                "latency_fp32_mean_ms": round(r["lat_fp32"]["mean_ms"], 3),
                "latency_int8_mean_ms": round(r["lat_int8"]["mean_ms"], 3),
                "speedup": round(r["speedup"], 3),
            }
        )
    report_path.write_text(json.dumps({"results": serializable}, indent=2), encoding="utf-8")
    print(f"Wrote machine-readable summary: {report_path}")


if __name__ == "__main__":
    main()
