import cv2
import numpy as np
import time
import zlib
import math
import json
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


INTERFACE = "enp6s0"
WINDOW = "R1-A7 Head Visual Baseline"

# 孔板 ROI：优先读取上一次手工标定结果
ROI_FILE = Path(
    "r1a7_threading_project/vision/r1a7_head_hole_roi.json"
)

board_roi = [455, 350, 360, 170]

if ROI_FILE.exists():
    try:
        saved = json.loads(ROI_FILE.read_text())
        if (
            isinstance(saved, list)
            and len(saved) == 4
        ):
            board_roi = [int(v) for v in saved]
            print("[ROI] loaded:", board_roi)
    except Exception as e:
        print("[ROI] load failed:", e)

target_hole = None
wire_tip = None
grasp_point = None
hole_candidates = []

latest_frame = None

# 鼠标左键当前选择模式：
# "hole" = 选择目标孔
# "tip"  = 选择细线自由端
click_mode = "hole"


def detect_holes(frame, roi):
    x, y, w, h = roi

    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)

    if w <= 0 or h <= 0:
        return []

    crop = frame[y:y+h, x:x+w]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 提升局部对比度
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        1.0
    )

    # 当前头部图中的孔很小，因此半径范围设得较小
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=10,
        param1=80,
        param2=10,
        minRadius=2,
        maxRadius=8
    )

    result = []

    if circles is not None:
        circles = np.round(
            circles[0]
        ).astype(int)

        for cx, cy, r in circles:
            result.append(
                (x + cx, y + cy, r)
            )

    return result


def nearest_hole(px, py, candidates, max_dist=25):
    best = None
    best_dist = 1e9

    for cx, cy, r in candidates:
        d = math.hypot(
            px - cx,
            py - cy
        )

        if d < best_dist:
            best_dist = d
            best = (cx, cy)

    if best_dist <= max_dist:
        return best

    return (px, py)


def mouse_callback(event, x, y, flags, param):
    global target_hole
    global wire_tip
    global grasp_point
    global click_mode

    # 只使用左键，避免 OpenCV/Qt 右键菜单冲突
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if click_mode == "hole":

        target_hole = nearest_hole(
            x,
            y,
            hole_candidates,
            max_dist=25
        )

        print(
            "[TARGET HOLE]",
            target_hole
        )

    elif click_mode == "tip":

        wire_tip = (x, y)

        print(
            "[WIRE TIP]",
            wire_tip
        )

    elif click_mode == "grasp":

        grasp_point = (x, y)

        print(
            "[GRASP POINT]",
            grasp_point
        )

def draw_overlay(frame):
    global hole_candidates
    global grasp_point

    out = frame.copy()

    # ------------------------------
    # ROI
    # ------------------------------
    x, y, w, h = board_roi

    cv2.rectangle(
        out,
        (x, y),
        (x + w, y + h),
        (255, 200, 0),
        2
    )

    cv2.putText(
        out,
        "hole ROI",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 200, 0),
        2
    )

    # ------------------------------
    # 自动孔候选
    # ------------------------------
    hole_candidates = detect_holes(
        frame,
        board_roi
    )

    for cx, cy, r in hole_candidates:
        cv2.circle(
            out,
            (cx, cy),
            r,
            (0, 255, 0),
            1
        )

        cv2.circle(
            out,
            (cx, cy),
            1,
            (0, 255, 0),
            -1
        )

    # ------------------------------
    # 目标孔
    # ------------------------------
    if target_hole is not None:
        hx, hy = target_hole

        cv2.circle(
            out,
            (hx, hy),
            10,
            (0, 0, 255),
            2
        )

        cv2.putText(
            out,
            "TARGET HOLE",
            (hx + 12, hy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    # ------------------------------
    # 线端
    # ------------------------------
    if wire_tip is not None:
        tx, ty = wire_tip

        cv2.drawMarker(
            out,
            (tx, ty),
            (255, 0, 255),
            cv2.MARKER_CROSS,
            18,
            2
        )

        cv2.putText(
            out,
            "WIRE TIP",
            (tx + 12, ty + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2
        )

    # ------------------------------
    # Grasp point
    # ------------------------------
    if grasp_point is not None:
        gx, gy = grasp_point

        cv2.drawMarker(
            out,
            (gx, gy),
            (255, 255, 0),
            cv2.MARKER_TILTED_CROSS,
            18,
            2
        )

        cv2.putText(
            out,
            "GRASP",
            (gx + 12, gy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )

    # 目前先显示像素自由长度。
    # 后续完成像素/mm标定后换算成实际毫米。
    if grasp_point is not None and wire_tip is not None:
        gx, gy = grasp_point
        tx, ty = wire_tip

        free_px = math.hypot(
            tx - gx,
            ty - gy
        )

        cv2.line(
            out,
            (gx, gy),
            (tx, ty),
            (255, 255, 0),
            2
        )

        mx = int((gx + tx) / 2)
        my = int((gy + ty) / 2)

        cv2.putText(
            out,
            f"L_free={free_px:.1f}px",
            (mx + 8, my),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

    # ------------------------------
    # Visual Servo Error
    # ------------------------------
    if target_hole is not None and wire_tip is not None:
        hx, hy = target_hole
        tx, ty = wire_tip

        eu = hx - tx
        ev = hy - ty

        dist = math.hypot(
            eu,
            ev
        )

        # 从线头指向目标孔
        cv2.arrowedLine(
            out,
            (tx, ty),
            (hx, hy),
            (0, 255, 255),
            2,
            tipLength=0.2
        )

        text = (
            f"eu={eu:+d}px  "
            f"ev={ev:+d}px  "
            f"|e|={dist:.1f}px"
        )

        cv2.rectangle(
            out,
            (10, 10),
            (520, 55),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            out,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

    # 状态
    cv2.putText(
        out,
        f"holes={len(hole_candidates)}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        out,
        f"MODE={click_mode.upper()}  H:hole T:tip G:grasp  Left-click:select",
        (20, out.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        out,
        "B:ROI H:hole T:tip G:grasp C:clear S:save ESC:quit",
        (20, out.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return out


print("[1] Initializing DDS")
ChannelFactoryInitialize(
    0,
    INTERFACE
)

print("[2] Initializing R1-A7 VideoClient")

client = VideoClient()
client.SetTimeout(3.0)
client.Init()

cv2.namedWindow(
    WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.setMouseCallback(
    WINDOW,
    mouse_callback
)

last_crc = None

new_frames = 0
duplicate_frames = 0

t0 = time.time()
last_print = t0

print()
print("====================================")
print("R1-A7 visual measurement baseline")
print("------------------------------------")
print("H + Left click : select target hole")
print("T + Left click : select insertion wire tip")
print("G + Left click : select tweezer grasp point")
print("B          : select board ROI")
print("C          : clear selections")
print("S          : save annotated image")
print("ESC        : quit")
print("====================================")
print()

while True:

    ret, data = client.GetImageSample()

    if ret != 0 or data is None or len(data) == 0:
        continue

    raw = bytes(data)

    crc = zlib.crc32(raw)

    # 跳过重复 JPEG
    if crc == last_crc:
        duplicate_frames += 1
        continue

    last_crc = crc
    new_frames += 1

    buf = np.frombuffer(
        raw,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        buf,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        continue

    latest_frame = frame

    vis = draw_overlay(frame)

    cv2.imshow(
        WINDOW,
        vis
    )

    now = time.time()

    if now - last_print >= 2.0:

        fps = new_frames / (
            now - t0
        )

        print(
            f"[VISION] "
            f"new_fps={fps:.2f} "
            f"holes={len(hole_candidates)} "
            f"duplicates={duplicate_frames}"
        )

        if (
            target_hole is not None
            and wire_tip is not None
        ):
            hx, hy = target_hole
            tx, ty = wire_tip

            print(
                f"         "
                f"hole=({hx},{hy}) "
                f"tip=({tx},{ty}) "
                f"eu={hx-tx:+d} "
                f"ev={hy-ty:+d}"
            )

        last_print = now

    key = cv2.waitKey(1) & 0xFF

    # ESC
    if key == 27:
        break

    # H/h: 选择目标孔模式
    elif key in (ord('h'), ord('H')):

        click_mode = "hole"

        print(
            "[MODE] TARGET HOLE"
        )

    # T/t: 选择线端模式
    elif key in (ord('t'), ord('T')):

        click_mode = "tip"

        print(
            "[MODE] WIRE TIP"
        )

    # G/g: 选择镊子抓取点
    elif key in (ord('g'), ord('G')):

        click_mode = "grasp"

        print(
            "[MODE] GRASP POINT"
        )

    # B/b: 重新选择孔板 ROI
    elif key in (ord('b'), ord('B')):

        roi = cv2.selectROI(
            "Select HOLE BOARD ROI",
            frame,
            fromCenter=False,
            showCrosshair=True
        )

        cv2.destroyWindow(
            "Select HOLE BOARD ROI"
        )

        rx, ry, rw, rh = [
            int(v)
            for v in roi
        ]

        if rw > 0 and rh > 0:

            board_roi[:] = [
                rx,
                ry,
                rw,
                rh
            ]

            ROI_FILE.write_text(
                json.dumps(board_roi)
            )

            print(
                "[NEW ROI]",
                board_roi
            )

            print(
                "[ROI SAVED]",
                ROI_FILE
            )

    # C/c
    elif key in (ord('c'), ord('C')):

        target_hole = None
        wire_tip = None
        grasp_point = None

        print(
            "[CLEAR]"
        )

    # S/s
    elif key in (ord('s'), ord('S')):

        path = (
            "/tmp/"
            "r1a7_visual_baseline_"
            f"{int(time.time())}.jpg"
        )

        cv2.imwrite(
            path,
            vis
        )

        print(
            "[SAVED]",
            path
        )

cv2.destroyAllWindows()

print()
print("Stopped safely.")
print(
    "unique/new frames =",
    new_frames
)
print(
    "duplicate frames =",
    duplicate_frames
)
