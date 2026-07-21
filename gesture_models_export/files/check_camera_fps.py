import time
import cv2

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera {CAMERA_INDEX}")

reported_fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Driver-reported FPS: {reported_fps}")
print(f"Current resolution: {int(width)}x{int(height)}")

# Warm up (first few frames are often slow/buffered)
for _ in range(10):
    cap.read()

# Measure actual delivered FPS over ~3 seconds
n_frames = 0
start = time.perf_counter()
duration = 3.0
while time.perf_counter() - start < duration:
    ok, frame = cap.read()
    if not ok:
        break
    n_frames += 1

elapsed = time.perf_counter() - start
measured_fps = n_frames / elapsed
print(f"Measured FPS over {elapsed:.2f}s: {measured_fps:.2f}  ({n_frames} frames)")

cap.release()
