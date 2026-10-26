#!/usr/bin/env python3

import argparse
import json
import time

import cv2
import numpy as np
from skimage.morphology import skeletonize

from ridge_detector import detect_wire
from wire_path_selector import select_wire_path, load_wire_init


def path_array(path):

    if path is None or len(path) == 0:
        return np.empty((0, 2), dtype=np.float32)

    return np.asarray(
        path,
        dtype=np.float32
    ).reshape(-1, 2)


def draw_path(
    image,
    points,
    color=(0, 255, 0),
    thickness=2
):

    if len(points) < 2:
        return

    pts = np.rint(points).astype(np.int32)

    cv2.polylines(
        image,
        [pts],
        False,
        color,
        thickness,
        cv2.LINE_AA
    )


class RealtimeWireTracker:

    def __init__(
        self,
        init_json="wire_init.json"
    ):

        # ------------------------------------------------
        # 这里只用于第一帧初始化
        # 不是固定坐标
        # ------------------------------------------------

        self.init_tip, self.init_grasp = load_wire_init(
            init_json
        )

        # 上一帧状态

        self.prev_gray = None

        self.prev_path = None

        self.prev_tip = None

        self.prev_grasp = None

        self.missed = 0


    # ====================================================
    # 1. 光流预测上一帧细线在当前帧的位置
    # ====================================================

    def predict_previous_path(
        self,
        gray
    ):

        if self.prev_gray is None:
            return None

        if self.prev_path is None:
            return None

        if len(self.prev_path) < 3:
            return None


        points = self.prev_path.astype(
            np.float32
        ).reshape(-1, 1, 2)


        next_points, status, _ = cv2.calcOpticalFlowPyrLK(

            self.prev_gray,

            gray,

            points,

            None,

            winSize=(21, 21),

            maxLevel=3,

            criteria=(
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                20,
                0.03
            )
        )


        if next_points is None:
            return None

        if status is None:
            return None


        good = status.reshape(-1) > 0


        if np.count_nonzero(good) < max(
            5,
            len(points) // 5
        ):

            return None


        return next_points.reshape(-1, 2)[good]


    # ====================================================
    # 2. 对当前帧 mask 做轻微连接
    # ====================================================

    def bridge_mask(
        self,
        mask
    ):

        kernel = cv2.getStructuringElement(

            cv2.MORPH_ELLIPSE,

            (5, 5)
        )


        return cv2.morphologyEx(

            mask,

            cv2.MORPH_CLOSE,

            kernel,

            iterations=1
        )


    # ====================================================
    # 3. 当前帧细线检测
    # ====================================================

    def detect_current_frame(
        self,
        frame
    ):

        ridge = detect_wire(
            frame
        )


        mask = (
            ridge["mask"] > 0
        ).astype(
            np.uint8
        ) * 255


        mask = self.bridge_mask(
            mask
        )


        skeleton = skeletonize(
            mask > 0
        ).astype(
            np.uint8
        ) * 255


        return (
            ridge,
            mask,
            skeleton
        )


    # ====================================================
    # 4. 用动态 tip / grasp 寻找当前细线路径
    # ====================================================

    def select_path(
        self,
        skeleton,
        tip,
        grasp
    ):

        if tip is None:
            return None

        if grasp is None:
            return None


        selected = select_wire_path(

            skeleton,

            tuple(
                np.rint(tip).astype(int)
            ),

            tuple(
                np.rint(grasp).astype(int)
            )
        )


        if selected is None:
            return None


        if len(selected) < 10:
            return None


        return selected


    # ====================================================
    # 5. 单帧更新
    # ====================================================

    def update(
        self,
        frame
    ):

        if frame is None:

            return {
                "valid": False,
                "tip": None,
                "grasp": None,
                "centerline": [],
                "confidence": 0.0,
                "missed": self.missed
            }


        gray = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2GRAY
        )


        # --------------------------------------------
        # 当前帧细线检测
        # --------------------------------------------

        ridge, mask, skeleton = \
            self.detect_current_frame(
                frame
            )


        # --------------------------------------------
        # 第一优先：
        # 根据上一帧细线进行光流预测
        # --------------------------------------------

        predicted = \
            self.predict_previous_path(
                gray
            )


        selected = None


        if predicted is not None:

            predicted_tip = predicted[0]

            predicted_grasp = predicted[-1]


            selected = self.select_path(

                skeleton,

                predicted_tip,

                predicted_grasp
            )


        # --------------------------------------------
        # 第二优先：
        # 在上一帧细线附近建立局部搜索区域
        # --------------------------------------------

        if selected is None:

            if self.prev_path is not None:

                corridor = np.zeros_like(
                    mask
                )


                previous = np.rint(
                    self.prev_path
                ).astype(
                    np.int32
                )


                cv2.polylines(

                    corridor,

                    [previous],

                    False,

                    255,

                    25,

                    cv2.LINE_AA
                )


                local_mask = cv2.bitwise_and(

                    mask,

                    corridor
                )


                local_skeleton = \
                    skeletonize(
                        local_mask > 0
                    ).astype(
                        np.uint8
                    ) * 255


                selected = self.select_path(

                    local_skeleton,

                    self.prev_path[0],

                    self.prev_path[-1]
                )


        # --------------------------------------------
        # 第三优先：
        # 第一次启动使用人工初始化点
        # --------------------------------------------

        if selected is None:

            selected = self.select_path(

                skeleton,

                self.init_tip,

                self.init_grasp
            )


        # =================================================
        # 得到有效细线
        # =================================================

        if selected is not None:

            path = path_array(
                selected
            )


            # -----------------------------------------
            # 保证 path[0] 是细线自由端
            # path[-1] 是抓取端
            # -----------------------------------------

            if self.prev_tip is not None:

                d0 = np.linalg.norm(

                    path[0] -
                    np.asarray(
                        self.prev_tip
                    )
                )


                d1 = np.linalg.norm(

                    path[-1] -
                    np.asarray(
                        self.prev_tip
                    )
                )


                if d1 < d0:

                    path = path[::-1]


            # -----------------------------------------
            # 更新时序状态
            # -----------------------------------------

            self.prev_path = path

            self.prev_tip = tuple(

                np.rint(
                    path[0]
                ).astype(int)
            )


            self.prev_grasp = tuple(

                np.rint(
                    path[-1]
                ).astype(int)
            )


            self.missed = 0


            # -----------------------------------------
            # 当前版本先用路径长度作为基础置信度
            # 后续再加入 ridge response
            # -----------------------------------------

            confidence = min(

                1.0,

                len(path) / 120.0
            )


            result = {

                "valid": True,

                "tip": list(
                    self.prev_tip
                ),

                "grasp": list(
                    self.prev_grasp
                ),

                "centerline":
                    np.rint(
                        path
                    ).astype(
                        int
                    ).tolist(),

                "confidence":
                    float(
                        confidence
                    ),

                "missed": 0
            }


        # =================================================
        # 当前帧失败
        # =================================================

        else:

            self.missed += 1


            result = {

                "valid": False,

                "tip":
                    list(
                        self.prev_tip
                    )
                    if self.prev_tip is not None
                    else None,

                "grasp":
                    list(
                        self.prev_grasp
                    )
                    if self.prev_grasp is not None
                    else None,

                "centerline":
                    np.rint(
                        self.prev_path
                    ).astype(
                        int
                    ).tolist()
                    if self.prev_path is not None
                    else [],

                "confidence": 0.0,

                "missed":
                    self.missed
            }


        # 保存当前灰度图作为下一帧参考

        self.prev_gray = gray


        return result


# ======================================================
# 可视化
# ======================================================

def draw_result(
    frame,
    result
):

    image = frame.copy()


    points = path_array(

        result["centerline"]
    )


    # 绿色：细线中心线

    if len(points) >= 2:

        draw_path(

            image,

            points,

            (0, 255, 0),

            2
        )


    # 红色：细线自由端

    if result["tip"] is not None:

        cv2.circle(

            image,

            tuple(
                result["tip"]
            ),

            7,

            (0, 0, 255),

            -1
        )


    # 蓝色：细线抓取端

    if result["grasp"] is not None:

        cv2.circle(

            image,

            tuple(
                result["grasp"]
            ),

            7,

            (255, 0, 0),

            -1
        )


    text = (

        f"wire={result['valid']} "

        f"conf={result['confidence']:.2f} "

        f"points={len(points)} "
    )


    cv2.putText(

        image,

        text,

        (10, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    return image


# ======================================================
# 单张图片测试
# ======================================================

def run_image(
    image_path
):

    image = cv2.imread(
        image_path
    )


    if image is None:

        raise FileNotFoundError(
            image_path
        )


    tracker = \
        RealtimeWireTracker()


    result = tracker.update(
        image
    )


    output = draw_result(

        image,

        result
    )


    cv2.imwrite(

        "wire_realtime_single.png",

        output
    )


    with open(

        "wire_realtime_result.json",

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            result,

            f,

            ensure_ascii=False,

            indent=2,
            default=lambda x: x.item() if hasattr(x, "item") else x.tolist() if hasattr(x, "tolist") else str(x),
        )


    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )


    print(
        "saved wire_realtime_single.png"
    )


# ======================================================
# OpenCV 摄像头实时测试
# ======================================================

def run_camera(
    index
):

    cap = cv2.VideoCapture(

        index,

        cv2.CAP_V4L2
    )


    if not cap.isOpened():

        cap.release()

        cap = cv2.VideoCapture(
            index
        )


    if not cap.isOpened():

        raise RuntimeError(

            f"cannot open camera {index}"
        )


    tracker = \
        RealtimeWireTracker()


    start_time = time.time()

    frame_count = 0


    print(
        "Realtime wire tracker started."
    )

    print(
        "Press Q or ESC to quit."
    )


    while True:

        ok, frame = cap.read()


        if not ok:

            print(
                "camera frame read failed"
            )

            break


        result = tracker.update(
            frame
        )


        frame_count += 1


        output = draw_result(

            frame,

            result
        )


        fps = (

            frame_count /
            max(
                1e-6,
                time.time() - start_time
            )
        )


        cv2.putText(

            output,

            f"FPS={fps:.1f}",

            (10, 60),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2,

            cv2.LINE_AA
        )


        cv2.imshow(

            "R1-A7 Right Wrist Wire Tracker",

            output
        )


        key = cv2.waitKey(1) & 0xFF


        if key in (
            27,
            ord("q")
        ):

            break


    cap.release()

    cv2.destroyAllWindows()


# ======================================================
# main
# ======================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--image",
        type=str
    )


    parser.add_argument(
        "--camera",
        type=int
    )


    args = parser.parse_args()


    if args.image:

        run_image(
            args.image
        )


    elif args.camera is not None:

        run_camera(
            args.camera
        )


    else:

        parser.error(
            "use --image PATH or --camera INDEX"
        )
