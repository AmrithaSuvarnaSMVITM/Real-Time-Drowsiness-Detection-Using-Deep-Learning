"""
Real-time drowsiness detection from webcam.

  python realtime.py --weights D:/drowsy_out/mobilenet_v3_small.pt --model mobilenet_v3_small

Keys:  q = quit | o / c = save current eye crop as 'open' / 'closed' into my_data/
       (use these to record YOUR OWN webcam data, then pass --cross_root my_data to
        train_compare.py to measure the real-world domain gap, or fine-tune on it)
"""
import argparse
import collections
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from train_compare import build_model

try:
    import winsound
except ImportError:
    winsound = None

LEFT = [362, 385, 387, 263, 373, 380]   # p1..p6 order for EAR
RIGHT = [33, 160, 158, 133, 153, 144]


def ear(p):
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (
        2 * np.linalg.norm(p[0] - p[3]) + 1e-6)


def crop_eye(gray, p, size):
    cx, cy = p.mean(0)
    s = int(np.linalg.norm(p[0] - p[3]) * 0.9)
    h, w = gray.shape
    x0, y0, x1, y1 = max(int(cx - s), 0), max(int(cy - s), 0), min(int(cx + s), w), min(int(cy + s), h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return cv2.resize(gray[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def crop_face(gray, pts, size):
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    m = 0.1 * max(x1 - x0, y1 - y0)
    h, w = gray.shape
    x0, y0, x1, y1 = max(int(x0 - m), 0), max(int(y0 - m), 0), min(int(x1 + m), w), min(int(y1 + m), h)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return cv2.resize(gray[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--mode", choices=["eye", "face"], default="eye", help="what the model was trained on")
    ap.add_argument("--ear_thresh", type=float, default=0.20, help="EAR baseline threshold")
    ap.add_argument("--max_closed", type=float, default=1.5, help="seconds of continuous closure -> alarm")
    ap.add_argument("--perclos", type=float, default=0.5, help="closed fraction over window -> alarm")
    ap.add_argument("--window", type=float, default=4.0, help="PERCLOS window (s)")
    ap.add_argument("--source", default="0", help="0 = webcam, or path to a video file (e.g. test.mp4)")
    args = ap.parse_args()

    torch.set_num_threads(2)
    model = build_model(args.model, pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    if args.source.isdigit():
        idx = int(args.source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if winsound else cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(args.source)  # video file
    hist = collections.deque()
    closed_since, last_beep = None, 0.0
    save_dir = Path("my_data")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        status, alarm, crops = "no face", False, []
        if res.multi_face_landmarks:
            h, w = gray.shape
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[l.x * w, l.y * h] for l in lm])
            le, re_ = pts[LEFT], pts[RIGHT]
            ear_val = (ear(le) + ear(re_)) / 2
            if args.mode == "face":
                crops = [c for c in (crop_face(gray, pts, args.size),) if c is not None]
            else:
                crops = [c for c in (crop_eye(gray, le, args.size), crop_eye(gray, re_, args.size)) if c is not None]
            if crops:
                x = torch.from_numpy(np.stack(crops)[:, None]).float() / 255
                with torch.no_grad():
                    p_closed = torch.softmax(model(x), 1)[:, 1].mean().item()
                closed = p_closed >= 0.5
                now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if not args.source.isdigit() else time.time()
                hist.append((now, closed))
                while hist and now - hist[0][0] > args.window:
                    hist.popleft()
                perclos = sum(c for _, c in hist) / len(hist)
                if closed:
                    if closed_since is None:
                        closed_since = now
                else:
                    closed_since = None
                alarm = (closed_since is not None and now - closed_since >= args.max_closed) or \
                        (len(hist) > 15 and perclos >= args.perclos)
                status = f"CNN p(closed)={p_closed:.2f} | EAR={ear_val:.2f} ({'closed' if ear_val < args.ear_thresh else 'open'}) | PERCLOS={perclos:.2f}"
                cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            for p in (le, re_):
                cv2.polylines(frame, [p.astype(np.int32)], True, (0, 255, 0), 1)
        if alarm:
            cv2.putText(frame, "DROWSY! WAKE UP", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if winsound and time.time() - last_beep > 1.0:
                winsound.Beep(1000, 300)
                last_beep = time.time()
        cv2.imshow("Drowsiness detector (q to quit)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k in (ord("o"), ord("c")) and crops:
            d = save_dir / ("open" if k == ord("o") else "closed")
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{int(time.time() * 1000)}.png"), crops[0])
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
