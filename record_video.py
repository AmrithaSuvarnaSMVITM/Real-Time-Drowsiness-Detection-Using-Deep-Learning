"""
Record a guided test video from your laptop webcam (about 70 seconds).
Follow the text on screen. The text is NOT saved in the video.

  python record_video.py
Saves: videos/my_test.mp4 (next to this script)   (press q to stop early)
"""
import time
from pathlib import Path

import cv2

try:
    import winsound
except ImportError:
    winsound = None

# (seconds, instruction)
PLAN = [
    (5, "Get ready - look at the screen"),
    (15, "EYES OPEN - look normally"),
    (10, "BLINK naturally"),
    (8, "CLOSE your eyes and keep them closed"),
    (7, "EYES OPEN again"),
    (3, "Slow blink: close 1 second, open"),
    (8, "CLOSE your eyes and keep them closed"),
    (7, "EYES OPEN"),
    (7, "Slightly look down (like reading), eyes open"),
]

out_dir = Path(__file__).parent / "videos"
out_dir.mkdir(exist_ok=True)
path = out_dir / "my_test.mp4"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Could not open the camera. Close other apps using it (Zoom, Camera app).")
for _ in range(10):          # let the camera adjust exposure
    cap.read()
ok, frame = cap.read()
h, w = frame.shape[:2]
fps = 20.0                    # fixed: we pace the writing below so playback speed is correct
writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

start = time.time()
total = sum(s for s, _ in PLAN)
next_frame_t = start
phase_end, idx = [], 0
acc = 0
for s, _ in PLAN:
    acc += s
    phase_end.append(acc)

while True:
    t = time.time() - start
    if t >= total:
        break
    ok, frame = cap.read()
    if not ok:
        break
    while next_frame_t <= time.time():   # write frames at a steady 20 fps of real time
        writer.write(frame)
        next_frame_t += 1.0 / fps
    while t >= phase_end[idx]:
        idx += 1
        if winsound:
            winsound.Beep(800, 150)
    shown = frame.copy()
    cv2.putText(shown, PLAN[idx][1], (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(shown, f"{int(total - t)}s left", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Recording (q to stop)", shown)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

writer.release()
cap.release()
cv2.destroyAllWindows()
print("Saved:", path)
