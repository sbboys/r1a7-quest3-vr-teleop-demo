import cv2
import numpy as np
import time
import json
import math
import zlib
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


INTERFACE = "enp6s0"
WINDOW = "R1-A7 Wire Endpoint Test"

ROI_FILE = Path(
    "r1a7_threading_project/vision/r1a7_head_wire_roi.json"
)

# 第一次没有标定时给一个大致初始值。
# 后面按 W 手动重选并自动保存。
wire_roi = [650, 320, 140, 190]

if ROI_FILE.exists():
    try:
        saved = json.loads(ROI_FILE.read_text())
        if isinstance(saved, list) and len(saved) == 4:
            wire_roi = [int(v) for v in saved]
            print("[WIRE ROI] loaded:", wire_roi)
    except Exception as e:
        print("[WIRE ROI] load failed:", e)


def skeletonize(binary):
    """
    OpenCV基础形态学骨架化。
    输入必须是 0/255 二值图。
    """
    img = binary.copy()

    skel = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (3, 3)
    )

    while True:
        eroded = cv2.erode(img, element)

        temp = cv2.dilate(
            eroded,
            element
        )

        temp = cv2.subtract(
            img,
            temp
        )

        skel = cv2.bitwise_or(
            skel,
            temp
        )

        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel


def remove_small_components(binary, min_area=20):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    out = np.zeros_like(binary)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            out[labels == i] = 255

    return out


def select_best_component(binary):
    """
    选择最像细长线体的连通域。
    评分偏好：
    - 较长
    - 较细
    - 面积不过小
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    best_label = None
    best_score = -1e9

    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 15:
            continue

        long_side = max(w, h)
        short_side = max(1, min(w, h))

        elongation = long_side / short_side

        # 太大块通常是孔板边缘/夹具，而不是细线
        if area > 2500:
            continue

        score = (
            3.0 * elongation
            + 0.08 * long_side
            - 0.002 * area
        )

        if score > best_score:
            best_score = score
            best_label = i

    if best_label is None:
        return np.zeros_like(binary)

    out = np.zeros_like(binary)
    out[labels == best_label] = 255

    return out


def find_skeleton_endpoints(skel):
    """
    skeleton 中：
    8邻域只有1个邻居的像素视为端点。
    """
    mask = (skel > 0).astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    neighbor_count = cv2.filter2D(
        mask,
        -1,
        kernel
    )

    # filter2D包含自身，所以：
    # 自身1 + 一个邻居 = 2
    endpoint_mask = (
        (mask == 1)
        & (neighbor_count == 2)
    )

    ys, xs = np.where(endpoint_mask)

    return list(zip(xs.tolist(), ys.tolist()))


def farthest_pair(points):
    """
    若端点数量>2，选择欧氏距离最远的一对。
    """
    if len(points) < 2:
        return None

    best = None
    best_d = -1

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]

            d = (
                (x1 - x2) ** 2
                + (y1 - y2) ** 2
            )

            if d > best_d:
                best_d = d
                best = (
                    points[i],
                    points[j]
                )

    return best


def detect_wire(frame, roi):
    x, y, w, h = roi

    H, W = frame.shape[:2]

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    crop = frame[
        y:y+h,
        x:x+w
    ]

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(6, 6)
    )

    gray_eq = clahe.apply(gray)

    # 细线目前是深色，因此用 black-hat 增强
    kernel_v = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 21)
    )

    blackhat = cv2.morphologyEx(
        gray_eq,
        cv2.MORPH_BLACKHAT,
        kernel_v
    )

    blackhat = cv2.GaussianBlur(
        blackhat,
        (3, 3),
        0
    )

    # Otsu自动阈值
    _, binary = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    # 连接细小断点
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 5)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )

    binary = remove_small_components(
        binary,
        min_area=15
    )

    component = select_best_component(
        binary
    )

    skel = skeletonize(
        component
    )

    endpoints = find_skeleton_endpoints(
        skel
    )

    pair = farthest_pair(
        endpoints
    )

    global_pair = None

    if pair is not None:
        p1, p2 = pair

        global_pair = (
            (x + p1[0], y + p1[1]),
            (x + p2[0], y + p2[1])
        )

    debug = {
        "crop": crop,
        "gray": gray_eq,
        "blackhat": blackhat,
        "binary": binary,
        "component": component,
        "skeleton": skel,
        "endpoints": endpoints
    }

    return global_pair, debug


def draw(frame, pair):
    out = frame.copy()

    x, y, w, h = wire_roi

    cv2.rectangle(
        out,
        (x, y),
        (x+w, y+h),
        (255, 255, 0),
        2
    )

    cv2.putText(
        out,
        "WIRE ROI",
        (x, max(20, y-8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )

    if pair is not None:
        p1, p2 = pair

        cv2.circle(
            out,
            p1,
            7,
            (0, 0, 255),
            2
        )

        cv2.circle(
            out,
            p2,
            7,
            (255, 0, 255),
            2
        )

        cv2.line(
            out,
            p1,
            p2,
            (0, 255, 255),
            1
        )

        cv2.putText(
            out,
            "END 1",
            (p1[0]+8, p1[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

        cv2.putText(
            out,
            "END 2",
            (p2[0]+8, p2[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2
        )

    cv2.putText(
        out,
        "W: coarse ROI   Z: 4x fine ROI",
        (20, out.shape[0]-45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        out,
        "S: save   ESC: quit",
        (20, out.shape[0]-18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return out


print("[1] Initialize DDS:", INTERFACE)

ChannelFactoryInitialize(
    0,
    INTERFACE
)

print("[2] Initialize VideoClient")

client = VideoClient()
client.SetTimeout(3.0)
client.Init()

cv2.namedWindow(
    WINDOW,
    cv2.WINDOW_NORMAL
)

last_crc = None

new_frames = 0
detect_ok = 0

t0 = time.time()
last_print = t0

latest_vis = None
latest_debug = None

print()
print("================================")
print("R1-A7 automatic wire endpoint test")
print("--------------------------------")
print("W : select COARSE wire ROI")
print("Z : select FINE wire ROI in 4x zoom")
print("S : save detection result")
print("ESC : quit")
print("================================")
print()

while True:

    ret, data = client.GetImageSample()

    if (
        ret != 0
        or data is None
        or len(data) == 0
    ):
        continue

    raw = bytes(data)

    crc = zlib.crc32(raw)

    # 跳过重复视频帧
    if crc == last_crc:
        continue

    last_crc = crc
    new_frames += 1

    frame = cv2.imdecode(
        np.frombuffer(
            raw,
            dtype=np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        continue

    pair, debug = detect_wire(
        frame,
        wire_roi
    )

    if pair is not None:
        detect_ok += 1

    vis = draw(
        frame,
        pair
    )

    latest_vis = vis
    latest_debug = debug

    cv2.imshow(
        WINDOW,
        vis
    )

    # 调试窗口
    cv2.imshow(
        "wire blackhat",
        debug["blackhat"]
    )

    cv2.imshow(
        "wire binary",
        debug["binary"]
    )

    cv2.imshow(
        "wire component",
        debug["component"]
    )

    cv2.imshow(
        "wire skeleton",
        debug["skeleton"]
    )

    now = time.time()

    if now - last_print >= 2.0:

        fps = (
            new_frames /
            (now - t0)
        )

        success_rate = (
            100.0 * detect_ok /
            max(1, new_frames)
        )

        print(
            f"[WIRE] new_fps={fps:.2f} "
            f"detect={success_rate:.1f}% "
            f"pair={pair}"
        )

        last_print = now

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    elif key in (
        ord('w'),
        ord('W')
    ):

        print("[ROI] Step 1/2: select COARSE wire area")

        # ------------------------------------------
        # Step 1: 在原始1280x720图上粗框
        # ------------------------------------------
        coarse = cv2.selectROI(
            "STEP 1 - COARSE WIRE ROI",
            frame,
            fromCenter=False,
            showCrosshair=True
        )

        cv2.destroyWindow(
            "STEP 1 - COARSE WIRE ROI"
        )

        cx, cy, cw, ch = [
            int(v)
            for v in coarse
        ]

        if cw <= 0 or ch <= 0:
            print("[ROI] coarse selection cancelled")
            continue

        print(
            "[ROI] coarse =",
            [cx, cy, cw, ch]
        )

        coarse_crop = frame[
            cy:cy+ch,
            cx:cx+cw
        ].copy()

        # ------------------------------------------
        # Step 2: 放大粗ROI，再精细选择
        # ------------------------------------------
        ZOOM = 5

        zoomed = cv2.resize(
            coarse_crop,
            None,
            fx=ZOOM,
            fy=ZOOM,
            interpolation=cv2.INTER_NEAREST
        )

        cv2.putText(
            zoomed,
            "STEP 2: select wire region, leave margin",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        print("[ROI] Step 2/2: select FINE ROI in 5x zoom")

        fine = cv2.selectROI(
            "STEP 2 - FINE WIRE ROI (5x)",
            zoomed,
            fromCenter=False,
            showCrosshair=True
        )

        cv2.destroyWindow(
            "STEP 2 - FINE WIRE ROI (5x)"
        )

        zx, zy, zw, zh = [
            int(v)
            for v in fine
        ]

        if zw <= 0 or zh <= 0:
            print("[ROI] fine selection cancelled")
            continue

        # ------------------------------------------
        # 放大图坐标 -> 原图坐标
        # ------------------------------------------
        fx = cx + int(round(zx / ZOOM))
        fy = cy + int(round(zy / ZOOM))

        fw = max(
            1,
            int(round(zw / ZOOM))
        )

        fh = max(
            1,
            int(round(zh / ZOOM))
        )

        wire_roi[:] = [
            fx,
            fy,
            fw,
            fh
        ]

        ROI_FILE.write_text(
            json.dumps(wire_roi)
        )

        print(
            "[WIRE ROI SAVED]",
            wire_roi
        )

        print(
            "[ROI] two-stage selection complete"
        )

    elif key in (
        ord('z'),
        ord('Z')
    ):

        # 当前 wire_roi 作为粗 ROI
        x, y, w, h = wire_roi

        H, W = frame.shape[:2]

        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        crop = frame[
            y:y+h,
            x:x+w
        ].copy()

        ZOOM = 4

        zoomed = cv2.resize(
            crop,
            None,
            fx=ZOOM,
            fy=ZOOM,
            interpolation=cv2.INTER_NEAREST
        )

        cv2.putText(
            zoomed,
            "Select wire area - NOT exact line width",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        fine = cv2.selectROI(
            "FINE WIRE ROI - 4x",
            zoomed,
            fromCenter=False,
            showCrosshair=True
        )

        cv2.destroyWindow(
            "FINE WIRE ROI - 4x"
        )

        zx, zy, zw, zh = [
            int(v)
            for v in fine
        ]

        if zw > 0 and zh > 0:

            # 放大图坐标 -> 原图坐标
            fx = x + int(round(zx / ZOOM))
            fy = y + int(round(zy / ZOOM))

            fw = max(
                1,
                int(round(zw / ZOOM))
            )

            fh = max(
                1,
                int(round(zh / ZOOM))
            )

            wire_roi[:] = [
                fx,
                fy,
                fw,
                fh
            ]

            ROI_FILE.write_text(
                json.dumps(wire_roi)
            )

            print(
                "[FINE WIRE ROI SAVED]",
                wire_roi
            )

    elif key in (
        ord('s'),
        ord('S')
    ):

        stamp = int(time.time())

        if latest_vis is not None:
            path = (
                f"/tmp/"
                f"r1a7_wire_detection_{stamp}.jpg"
            )

            cv2.imwrite(
                path,
                latest_vis
            )

            print(
                "[SAVED]",
                path
            )

        if latest_debug is not None:

            for name in [
                "blackhat",
                "binary",
                "component",
                "skeleton"
            ]:

                path = (
                    f"/tmp/"
                    f"r1a7_wire_{name}_{stamp}.png"
                )

                cv2.imwrite(
                    path,
                    latest_debug[name]
                )

                print(
                    "[SAVED]",
                    path
                )

cv2.destroyAllWindows()

print("Stopped safely.")
