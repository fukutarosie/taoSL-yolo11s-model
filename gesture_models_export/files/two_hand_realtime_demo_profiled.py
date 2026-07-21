import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from inference import load_model, square_crop_box, transform


LEFT_ID_BY_NAME = {
    "left_stop": 0,
    "left_front": 1,
    "left_behind": 2,
    "left_left": 3,
    "left_right": 4,
    "left_frontdash": 5,
    "left_front_dash": 5,
    "left_behinddash": 6,
    "left_behind_dash": 6,
    "left_leftdash": 7,
    "left_left_dash": 7,
    "left_rightdash": 8,
    "left_right_dash": 8,
    "static": 0,
    "stop": 0,
}

RIGHT_ID_BY_NAME = {
    "fist": 0,
    "stop": 0,
    "iron": 1,
    "young": 2,
    "flow": 3,
    "burst": 4,
    "ground": 5,
    "confirm": 6,
    "like": 6,
    "thumbs_up": 6,
    "thumb_up": 6,
    "cancel": 7,
    "pause": 7,
    "cancel_pause": 7,
    "dislike": 7,
    "thumbs_down": 7,
    "thumb_down": 7,
}


# ---------------------------------------------------------------------------
# PROFILING HELPERS (new)
# ---------------------------------------------------------------------------
class StageTimer:
    """Accumulates elapsed time per named stage and prints a rolling summary."""

    def __init__(self, print_every=30):
        self.print_every = print_every
        self.totals = {}
        self.counts = {}
        self.frame_count = 0
        self._t0 = None
        self._stage = None

    def start(self, stage):
        # close out any open stage first
        self.stop()
        self._stage = stage
        self._t0 = time.perf_counter()

    def stop(self):
        if self._stage is None:
            return
        elapsed = time.perf_counter() - self._t0
        self.totals[self._stage] = self.totals.get(self._stage, 0.0) + elapsed
        self.counts[self._stage] = self.counts.get(self._stage, 0) + 1
        self._stage = None
        self._t0 = None

    def end_frame(self):
        self.frame_count += 1
        if self.frame_count % self.print_every == 0:
            self.report()
            self.totals = {}
            self.counts = {}

    def report(self):
        parts = []
        total_ms = 0.0
        for stage, total in self.totals.items():
            n = max(1, self.counts[stage])
            avg_ms = (total / n) * 1000.0
            total_ms += avg_ms
            parts.append(f"{stage}={avg_ms:6.2f}ms")
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0
        line = "  ".join(parts)
        print(f"\n[PROFILE] {line}  | total={total_ms:6.2f}ms (~{fps:5.1f} FPS)")


def select_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_label(label):
    label = str(label).strip()
    if "_" in label and label.split("_", 1)[0].isdigit():
        label = label.split("_", 1)[1]
    return label.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def label_to_id(label, side):
    lookup = LEFT_ID_BY_NAME if side == "Left" else RIGHT_ID_BY_NAME
    return lookup.get(normalize_label(label), -1)


def create_mediapipe_hands(min_confidence):
    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        raise SystemExit(
            "Your current mediapipe package does not expose mediapipe.solutions.\n"
            "Use the gesture env: conda activate gesture"
        )

    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,  # full model; frame-skip (--detect-every) recovers the speed instead
        min_detection_confidence=min_confidence,
        min_tracking_confidence=0.5,
    )


def detect_hands(frame, hands, crop_margin, swap_handedness):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return []

    height, width = frame.shape[:2]
    detections = []
    handedness = result.multi_handedness or []

    for idx, landmarks in enumerate(result.multi_hand_landmarks):
        xs = [landmark.x * width for landmark in landmarks.landmark]
        ys = [landmark.y * height for landmark in landmarks.landmark]
        raw_box = (
            max(0, int(min(xs))),
            max(0, int(min(ys))),
            min(width, int(max(xs))),
            min(height, int(max(ys))),
        )
        if raw_box[2] <= raw_box[0] or raw_box[3] <= raw_box[1]:
            continue

        side = "Left"
        side_conf = 0.0
        if idx < len(handedness) and handedness[idx].classification:
            cls = handedness[idx].classification[0]
            side = cls.label
            side_conf = cls.score
        if swap_handedness:
            side = "Right" if side == "Left" else "Left"

        crop_box = square_crop_box(raw_box, width, height, crop_margin)
        detections.append({"side": side, "side_conf": side_conf, "box": crop_box})

    return detections


def mirror_box(box, width):
    x1, y1, x2, y2 = box
    return width - x2, y1, width - x1, y2


class DINOClassifier:
    def __init__(self, model_path, device):
        self.device = device
        self.model, self.class_names = load_model(model_path)
        self.model.to(device)
        self.model.eval()

    def predict(self, crop_bgr):
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(crop_rgb).convert("RGB")
        x = transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)[0]
            top_idx = int(prob.argmax().item())

        return self.class_names[top_idx], float(prob[top_idx].item())


class YOLOClassifier:
    def __init__(self, model_path):
        self.model = YOLO(str(model_path))

    def predict(self, crop_bgr):
        result = self.model(crop_bgr, verbose=False)[0]
        if result.probs is not None:
            idx = int(result.probs.top1)
            conf = float(result.probs.top1conf.item())
            return result.names[idx], conf

        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf
            best_idx = int(confs.argmax().item())
            cls_idx = int(result.boxes.cls[best_idx].item())
            conf = float(confs[best_idx].item())
            return result.names[cls_idx], conf

        return "unknown", 0.0


def build_classifiers(args, device):
    if args.backend == "dinov2":
        if not args.left_dino or not args.right_dino:
            raise SystemExit("--backend dinov2 requires --left-dino and --right-dino")
        return {
            "Left": DINOClassifier(args.left_dino, device),
            "Right": DINOClassifier(args.right_dino, device),
        }

    if not args.left_yolo or not args.right_yolo:
        raise SystemExit("--backend yolo requires --left-yolo and --right-yolo")
    return {
        "Left": YOLOClassifier(args.left_yolo),
        "Right": YOLOClassifier(args.right_yolo),
    }


def main():
    parser = argparse.ArgumentParser(description="Realtime two-hand gesture demo that outputs [left_id, right_id].")
    parser.add_argument("--backend", choices=("yolo", "dinov2"), default="yolo")
    parser.add_argument("--left-yolo", default=None)
    parser.add_argument("--right-yolo", default="right8_yolo11s_cls_best.pt")
    parser.add_argument("--left-dino", default=None)
    parser.add_argument("--right-dino", default="/Users/wangqilin/Desktop/FYP/dinov2_dataset_hand_vits_mlp.pth")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--det-conf", type=float, default=0.35)
    parser.add_argument("--cls-conf", type=float, default=0.45)
    parser.add_argument("--crop-margin", type=float, default=0.18)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Show/process handedness in selfie view, while classifying the original unmirrored crop.",
    )
    parser.add_argument("--swap-handedness", action="store_true")
    parser.add_argument(
        "--profile-every",
        type=int,
        default=30,
        help="Print a timing summary every N frames (new, for profiling).",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=2,
        help="Run mediapipe hand detection every N frames, reusing the last known boxes in between (new).",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"[PROFILE] device = {device}")
    classifiers = build_classifiers(args, device)
    hands = create_mediapipe_hands(args.det_conf)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    last = {
        "Left": {"id": -1, "label": "none", "conf": 0.0},
        "Right": {"id": -1, "label": "none", "conf": 0.0},
    }
    history = {"Left": deque(maxlen=5), "Right": deque(maxlen=5)}
    frame_idx = 0
    timer = StageTimer(print_every=args.profile_every)
    last_detections = []  # reused on frames where detection is skipped

    print("Camera opened. Press q or Esc to quit.")
    while True:
        timer.start("capture")
        ok, frame = cap.read()
        if not ok:
            print("Camera returned no frame.")
            break

        raw_frame = frame
        timer.start("mirror_prep")
        if args.mirror:
            mediapipe_frame = cv2.flip(raw_frame, 1)
            display_frame = mediapipe_frame.copy()
        else:
            mediapipe_frame = raw_frame
            display_frame = raw_frame.copy()

        timer.start("mediapipe_detect")
        if frame_idx % max(1, args.detect_every) == 0:
            detections = detect_hands(mediapipe_frame, hands, args.crop_margin, args.swap_handedness)
            last_detections = detections
        else:
            detections = last_detections

        seen_sides = set()
        frame_height, frame_width = raw_frame.shape[:2]

        timer.start("classify")
        for det in detections:
            side = det["side"]
            if side not in classifiers:
                continue
            seen_sides.add(side)
            x1, y1, x2, y2 = det["box"]
            crop_box = mirror_box(det["box"], frame_width) if args.mirror else det["box"]
            cx1, cy1, cx2, cy2 = crop_box
            cx1 = max(0, min(frame_width, cx1))
            cx2 = max(0, min(frame_width, cx2))
            cy1 = max(0, min(frame_height, cy1))
            cy2 = max(0, min(frame_height, cy2))
            crop = raw_frame[cy1:cy2, cx1:cx2]

            if frame_idx % max(1, args.every) == 0 and crop.size:
                label, conf = classifiers[side].predict(crop)
                pred_id = label_to_id(label, side) if conf >= args.cls_conf else -1
                history[side].append(pred_id)
                if history[side]:
                    pred_id = max(set(history[side]), key=list(history[side]).count)
                last[side] = {"id": pred_id, "label": label, "conf": conf}

            color = (80, 220, 80) if side == "Left" else (80, 160, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            info = last[side]
            cv2.putText(
                display_frame,
                f"{side} id={info['id']} {info['label']} {info['conf']:.2f}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        for side in ("Left", "Right"):
            if side not in seen_sides:
                last[side] = {"id": -1, "label": "none", "conf": 0.0}
                history[side].clear()

        timer.start("render")
        output = [last["Left"]["id"], last["Right"]["id"]]
        cv2.putText(
            display_frame,
            f"gesture_array = {output}",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        print(f"\rgesture_array={output}", end="", flush=True)

        cv2.imshow("Two Hand Gesture Demo", display_frame)
        key = cv2.waitKey(1) & 0xFF

        timer.stop()  # close out "render"
        timer.end_frame()

        if key in (27, ord("q")):
            break
        frame_idx += 1

    print()
    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
