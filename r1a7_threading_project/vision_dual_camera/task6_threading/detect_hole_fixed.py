#!/usr/bin/env python3
"""R1-A7 task6: fixed USB camera circular-hole detector.

Stage-1 only: detect the hole in image coordinates. This script DOES NOT control the robot.

Keys:
  q / ESC : quit
  s       : save current annotated frame
  r       : clear temporal history

Default output JSON: /tmp/r1a7_task6_hole.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def parse_roi(text: str | None):
    if not text:
        return None
    vals = [int(v.strip()) for v in text.split(',')]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    x, y, w, h = vals
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("ROI width/height must be > 0")
    return x, y, w, h


def open_camera(device: str, width: int, height: int, fps: int):
    # /dev/videoN works reliably with V4L2 on Linux.
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def circle_contrast_score(gray: np.ndarray, cx: int, cy: int, r: int):
    """Higher score means a darker circular interior surrounded by a brighter ring."""
    h, w = gray.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    inner = d2 <= (0.65 * r) ** 2
    ring = (d2 >= (1.05 * r) ** 2) & (d2 <= (1.55 * r) ** 2)
    if inner.sum() < 10 or ring.sum() < 10:
        return -1e9, 0.0
    inner_mean = float(gray[inner].mean())
    ring_mean = float(gray[ring].mean())
    contrast = ring_mean - inner_mean
    # Normalize to approximately 0..1 for display only.
    confidence = float(np.clip(contrast / 70.0, 0.0, 1.0))
    return contrast, confidence


def detect_hough(gray: np.ndarray, min_r: int, max_r: int):
    blur = cv2.GaussianBlur(gray, (9, 9), 1.8)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_r * 2),
        param1=120,
        param2=24,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []

    out = []
    for c in np.round(circles[0]).astype(int):
        cx, cy, r = map(int, c)
        if cx - 2 * r < 0 or cy - 2 * r < 0 or cx + 2 * r >= gray.shape[1] or cy + 2 * r >= gray.shape[0]:
            continue
        contrast, conf = circle_contrast_score(gray, cx, cy, r)
        out.append({
            "cx": float(cx), "cy": float(cy), "r": float(r),
            "score": float(contrast), "confidence": conf,
            "source": "hough",
        })
    return out


def detect_contour(gray: np.ndarray, min_r: int, max_r: int):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            continue
        peri = float(cv2.arcLength(cnt, True))
        if peri <= 1e-6:
            continue
        circularity = 4.0 * math.pi * area / (peri * peri)
        if circularity < 0.55:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if r < min_r or r > max_r:
            continue
        fill = area / (math.pi * r * r + 1e-9)
        if fill < 0.45:
            continue

        contrast, conf = circle_contrast_score(gray, int(round(cx)), int(round(cy)), int(round(r)))
        # Prefer round contours with dark center and bright surroundings.
        score = contrast + 20.0 * circularity + 5.0 * fill
        out.append({
            "cx": float(cx), "cy": float(cy), "r": float(r),
            "score": float(score),
            "confidence": float(np.clip(0.65 * conf + 0.35 * circularity, 0.0, 1.0)),
            "source": "contour",
        })
    return out


def pick_best(gray: np.ndarray, min_r: int, max_r: int):
    candidates = detect_hough(gray, min_r, max_r)
    candidates += detect_contour(gray, min_r, max_r)
    if not candidates:
        return None, []

    # Reject candidates that are not meaningfully darker than their surrounding ring.
    filtered = [c for c in candidates if c["score"] > 5.0]
    if not filtered:
        filtered = candidates
    best = max(filtered, key=lambda c: c["score"])
    return best, candidates


def robust_median(history: deque):
    arr = np.array([[d["cx"], d["cy"], d["r"]] for d in history], dtype=np.float64)
    med = np.median(arr, axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def atomic_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description="R1-A7 task6 fixed-camera hole detector (no robot control)")
    ap.add_argument("--device", default="/dev/video14")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--min-radius", type=int, default=6, help="minimum hole radius in pixels")
    ap.add_argument("--max-radius", type=int, default=80, help="maximum hole radius in pixels")
    ap.add_argument("--roi", type=parse_roi, default=None, help="optional ROI: x,y,w,h")
    ap.add_argument("--history", type=int, default=7, help="temporal median window")
    ap.add_argument("--print-period", type=float, default=0.25)
    ap.add_argument("--output-json", default="/tmp/r1a7_task6_hole.json")
    ap.add_argument("--save-dir", default="task6_hole_debug")
    args = ap.parse_args()

    cap = open_camera(args.device, args.width, args.height, args.fps)
    hist = deque(maxlen=max(1, args.history))
    out_path = Path(args.output_json)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(f"[INFO] camera={args.device} actual={actual_w}x{actual_h} fps={actual_fps:.1f}")
    print("[INFO] Stage-1 only: image detection. Robot will NOT move.")
    print("[KEY] q/ESC quit, s save frame, r reset history")

    last_print = 0.0
    frame_id = 0
    last_payload = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] camera read failed")
                time.sleep(0.05)
                continue
            frame_id += 1

            h, w = frame.shape[:2]
            if args.roi is None:
                x0, y0, rw, rh = 0, 0, w, h
            else:
                x0, y0, rw, rh = args.roi
                x0 = max(0, min(x0, w - 1))
                y0 = max(0, min(y0, h - 1))
                rw = max(1, min(rw, w - x0))
                rh = max(1, min(rh, h - y0))

            roi_bgr = frame[y0:y0 + rh, x0:x0 + rw]
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            best, candidates = pick_best(gray, args.min_radius, args.max_radius)

            annotated = frame.copy()
            if args.roi is not None:
                cv2.rectangle(annotated, (x0, y0), (x0 + rw, y0 + rh), (255, 180, 0), 2)

            # Show weak candidates in thin gray for tuning.
            for c in candidates[:25]:
                cx = int(round(c["cx"] + x0))
                cy = int(round(c["cy"] + y0))
                rr = int(round(c["r"]))
                cv2.circle(annotated, (cx, cy), rr, (120, 120, 120), 1)

            payload = {
                "timestamp": time.time(),
                "frame_id": frame_id,
                "camera": args.device,
                "image_width": w,
                "image_height": h,
                "detected": False,
            }

            if best is not None:
                detected = dict(best)
                detected["cx"] += x0
                detected["cy"] += y0
                hist.append(detected)
                u, v, r = robust_median(hist)

                cx, cy, rr = int(round(u)), int(round(v)), int(round(r))
                cv2.circle(annotated, (cx, cy), rr, (0, 255, 0), 2)
                cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                label = f"HOLE u={u:.1f} v={v:.1f} r={r:.1f}px conf={best['confidence']:.2f}"
                cv2.putText(annotated, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                payload.update({
                    "detected": True,
                    "u_px": u,
                    "v_px": v,
                    "radius_px": r,
                    "confidence": float(best["confidence"]),
                    "source": best["source"],
                    "history_len": len(hist),
                })
            else:
                cv2.putText(annotated, "HOLE NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # A small reference cross at the optical image center.
            cv2.drawMarker(annotated, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 16, 1)

            now = time.time()
            if now - last_print >= args.print_period:
                atomic_write_json(out_path, payload)
                if payload["detected"]:
                    print(
                        f"[HOLE] u={payload['u_px']:.2f} v={payload['v_px']:.2f} "
                        f"r={payload['radius_px']:.2f}px conf={payload['confidence']:.2f} "
                        f"src={payload['source']}"
                    )
                else:
                    print("[HOLE] NOT_FOUND")
                last_print = now
                last_payload = payload

            cv2.imshow("R1-A7 Task6 - Fixed Camera Hole Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('r'):
                hist.clear()
                print("[INFO] history reset")
            if key == ord('s'):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = save_dir / f"hole_{stamp}_{frame_id:06d}.png"
                cv2.imwrite(str(path), annotated)
                print(f"[SAVE] {path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if last_payload is not None:
            try:
                atomic_write_json(out_path, last_payload)
            except Exception:
                pass


if __name__ == "__main__":
    main()
