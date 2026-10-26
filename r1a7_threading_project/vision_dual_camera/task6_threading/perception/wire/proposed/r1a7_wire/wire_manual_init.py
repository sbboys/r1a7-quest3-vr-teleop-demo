#!/usr/bin/env python3

import cv2
import json


IMAGE="../../dataset/wire_test_external1.png"


points=[]


img=cv2.imread(
    IMAGE
)


if img is None:
    raise RuntimeError(
        "image load failed"
    )


display=img.copy()


def callback(event,x,y,flags,param):

    global display

    if event==cv2.EVENT_LBUTTONDOWN:

        points.append(
            [x,y]
        )

        cv2.circle(
            display,
            (x,y),
            8,
            (0,0,255),
            -1
        )

        cv2.imshow(
            "manual init",
            display
        )


cv2.namedWindow(
    "manual init"
)


cv2.setMouseCallback(
    "manual init",
    callback
)


print("======================")
print("Click wire tip first")
print("Click grasp point second")
print("Press ENTER")
print("======================")


cv2.imshow(
    "manual init",
    display
)


while True:

    key=cv2.waitKey(1)&0xff

    if key==13 and len(points)>=2:

        break


cv2.destroyAllWindows()


result={

    "tip":
        points[0],

    "grasp":
        points[1]

}


with open(
    "wire_init.json",
    "w"
) as f:

    json.dump(
        result,
        f,
        indent=2
    )


print(
    json.dumps(
        result,
        indent=2
    )
)

print(
    "saved wire_init.json"
)
