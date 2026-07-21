import argparse
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONF_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.20

BACKBONE_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


class DINOv2GestureClassifier(nn.Module):
    def __init__(self, num_classes, backbone="dinov2_vitb14"):
        super().__init__()
        if backbone not in BACKBONE_DIMS:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.encoder = torch.hub.load(
            "facebookresearch/dinov2",
            backbone
        )

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(BACKBONE_DIMS[backbone], 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            feat = self.encoder(x)
        return self.mlp(feat)


def create_hand_detector(detector, model_path, min_confidence):
    if detector == "none":
        return "full"
    if detector == "skin":
        return "skin"

    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        raise SystemExit(
            "This mediapipe install does not expose mediapipe.solutions.\n"
            "Use Python 3.10/3.11 and install: pip install 'mediapipe<0.11'"
        )

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=0.5,
    )
    return {"type": "mediapipe", "model": hands}


def detect_skin_box(image_rgb):
    height, width = image_rgb.shape[:2]
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

    hsv_mask = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([25, 255, 255]))
    ycrcb_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = width * height * 0.003
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        candidates.append((area, x, y, x + w, y + h))

    if not candidates:
        return None

    _, x1, y1, x2, y2 = max(candidates, key=lambda item: item[0])
    pad = int(max(x2 - x1, y2 - y1) * 0.25)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def detect_hand_box(image_rgb, hands_detector):
    if hands_detector is None:
        return None
    if hands_detector == "skin":
        return detect_skin_box(image_rgb)

    if not isinstance(hands_detector, dict) or hands_detector.get("type") != "mediapipe":
        return None

    height, width = image_rgb.shape[:2]
    result = hands_detector["model"].process(image_rgb)
    if not result.multi_hand_landmarks:
        return None

    boxes = []
    for hand_landmarks in result.multi_hand_landmarks:
        xs = [landmark.x * width for landmark in hand_landmarks.landmark]
        ys = [landmark.y * height for landmark in hand_landmarks.landmark]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))
        boxes.append(((x2 - x1) * (y2 - y1), x1, y1, x2, y2))

    if not boxes:
        return None

    _, x1, y1, x2, y2 = max(boxes, key=lambda item: item[0])

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def square_crop_box(box, width, height, margin=0.15):
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    side = int(round(max(box_w, box_h) * (1.0 + 2.0 * margin)))
    side = max(1, min(side, width, height))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    right = min(width, left + side)
    bottom = min(height, top + side)

    left = max(0, right - side)
    top = max(0, bottom - side)
    return int(left), int(top), int(right), int(bottom)


def crop_hand(image, hands_detector, crop_margin=0.15):
    if hands_detector == "full":
        return image

    image_rgb = np.array(image)
    box = detect_hand_box(image_rgb, hands_detector)
    if box is None:
        return None

    box = square_crop_box(box, image.width, image.height, crop_margin)
    return image.crop(box)


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint["classes"]
    backbone = checkpoint.get("backbone", "dinov2_vitb14")

    model = DINOv2GestureClassifier(num_classes=len(class_names), backbone=backbone).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, class_names


def predict(image_path, model, class_names, hands_detector, crop_margin):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")

    crop = crop_hand(image, hands_detector, crop_margin)
    if crop is None:
        return "other", 0.0

    x = transform(crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        top2_prob, top2_idx = torch.topk(prob, k=2, dim=1)

    conf = top2_prob[0, 0].item()
    margin = top2_prob[0, 0].item() - top2_prob[0, 1].item()
    pred_idx = top2_idx[0, 0].item()

    if conf < CONF_THRESHOLD or margin < MARGIN_THRESHOLD:
        return "other", conf

    return class_names[pred_idx], conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a hand gesture image.")
    parser.add_argument("image", nargs="?", default="test.jpg")
    parser.add_argument("--model", default="dinov2_gesture_mlp.pth")
    parser.add_argument("--detector", choices=("skin", "mediapipe", "none"), default="skin")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task")
    parser.add_argument("--min-hand-confidence", type=float, default=0.35)
    parser.add_argument("--crop-margin", type=float, default=0.15)
    args = parser.parse_args()

    model, class_names = load_model(args.model)

    hands_detector = create_hand_detector(args.detector, args.hand_model, args.min_hand_confidence)
    if hands_detector is None:
        print("Warning: no detector available")

    detector_context = hands_detector if hasattr(hands_detector, "__enter__") else nullcontext()
    with detector_context:
        label, conf = predict(args.image, model, class_names, hands_detector, args.crop_margin)

    print("Prediction:", label)
    print("Confidence:", conf)
