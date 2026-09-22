#!/usr/bin/env python3

import json
import time
from pathlib import Path


INPUT = Path(
    "vision_dual_camera/task6_threading/wrist_error.json"
)


OUTPUT = Path(
    "/tmp/r1a7_wrist_ibvs_velocity.json"
)


# ============================================================
# PID parameters
# ============================================================

KP_Y = 0.015
KP_Z = 0.015


MAX_SPEED = 5.0


def limit(v):

    if v > MAX_SPEED:
        return MAX_SPEED

    if v < -MAX_SPEED:
        return -MAX_SPEED

    return v



def main():

    print(
        "[WRIST PID] started",
        flush=True
    )


    while True:


        if not INPUT.exists():

            time.sleep(
                0.1
            )

            continue


        try:

            data=json.loads(
                INPUT.read_text()
            )


            du=float(
                data["du"]
            )

            dv=float(
                data["dv"]
            )


            # Wrist camera mapping:
            #
            # U error -> robot Y
            # V error -> robot Z
            #

            vy = limit(
                KP_Y * du
            )


            vz = limit(
                KP_Z * (-dv)
            )


            output={

                "enabled": True,

                "vy_mm_s": vy,

                "vz_mm_s": vz,

                "du": du,

                "dv": dv,

                "timestamp": time.time()

            }


            OUTPUT.write_text(
                json.dumps(
                    output,
                    indent=2
                )
            )


            print(
                "[WRIST PID]",
                output,
                flush=True
            )


        except Exception as e:

            print(
                "[WRIST PID ERROR]",
                e,
                flush=True
            )


        time.sleep(
            0.05
        )



if __name__ == "__main__":
    main()
