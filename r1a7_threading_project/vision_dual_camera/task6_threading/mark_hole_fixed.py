#!/usr/bin/env python3
"""R1-A7 task6: manually mark the target hole in the fixed USB camera image.

Stage-1 manual mode only. This script DOES NOT control the robot.

Mouse:
  Left click  : set / update hole center
  Right click : clear selected hole

Keys:
  q / ESC : quit
  c       : clear selected hole
  s       : save current annotated frame

The selected image coordinate is continuously written to:
  /tmp/r1a7_task6_hole.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

WINDOW_NAME = "R1-A7 Task6 - Click Hole Center"


def open_camera(device: str, width: int, height: int, fps: int):
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


def atomic_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(
        description="R1-A7 task6 fixed-camera manual hole marker (no robot control)"
    )
    ap.add_argument("--device", default="/dev/video14")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--marker-radius", type=int, default=16)
    ap.add_argument("--print-period", type=float, default=0.5)
    ap.add_argument("--output-json", default="/tmp/r1a7_task6_hole.json")
    ap.add_argument("--save-dir", default="task6_hole_debug")
    args = ap.parse_args()

    cap = open_camera(args.device, args.width, args.height, args.fps)
    out_path = Path(args.output_json)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

    selection = {"u": None, "v": None}

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection["u"] = int(x)
            selection["v"] = int(y)
            print(f"[SELECT] hole center u={x} v={y}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            selection["u"] = None
            selection["v"] = None
            print("[CLEAR] hole selection cleared")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print(f"[INFO] camera={args.device} actual={actual_w}x{actual_h} fps={actual_fps:.1f}")
    print("[INFO] Manual hole marking mode. Robot will NOT move.")
    print("[MOUSE] left-click=set hole center, right-click=clear")
    print("[KEY] q/ESC quit, c clear, s save frame")

    frame_id = 0
    last_print = 0.0
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
            annotated = frame.copy()

            # Image center is only a visual reference.
            cv2.drawMarker(
                annotated,
                (w // 2, h // 2),
                (255, 255, 0),
                cv2.MARKER_CROSS,
                18,
                1,
            )

            u = selection["u"]
            v = selection["v"]
            selected = u is not None and v is not None

            payload = {
                "timestamp": time.time(),
                "frame_id": frame_id,
                "camera": args.device,
                "image_width": w,
                "image_height": h,
                "detected": bool(selected),
                "mode": "manual_click",
            }

            if selected:
                # Clamp selection in case the capture geometry changes.
                u = max(0, min(int(u), w - 1))
                v = max(0, min(int(v), h - 1))
                selection["u"] = u
                selection["v"] = v

                cv2.circle(annotated, (u, v), args.marker_radius, (0, 255, 0), 2)
                cv2.drawMarker(
                    annotated,
                    (u, v),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    30,
                    2,
                )
                cv2.putText(
                    annotated,
                    f"SELECTED HOLE  u={u}  v={v}",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    annotated,
                    "Left click again to update | Right click/c to clear",
                    (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 255),
                    2,
                )

                payload.update({
                    "u_px": float(u),
                    "v_px": float(v),
                    "confidence": 1.0,
                    "source": "manual_click",
                })
            else:
                cv2.putText(
                    annotated,
                    "LEFT-CLICK THE TARGET HOLE CENTER",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    annotated,
                    "Robot control is disabled in this program",
                    (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 255),
                    2,
                )

            now = time.time()
            if now - last_print >= args.print_period:
                atomic_write_json(out_path, payload)
                if selected:
                    print(f"[HOLE] u={u:.2f} v={v:.2f} mode=manual_click")
                else:
                    print("[HOLE] NOT_SELECTED")
                last_print = now
                last_payload = payload

            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            if key == ord("c"):
                selection["u"] = None
                selection["v"] = None
                print("[CLEAR] hole selection cleared")
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = save_dir / f"manual_hole_{stamp}_{frame_id:06d}.png"
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
