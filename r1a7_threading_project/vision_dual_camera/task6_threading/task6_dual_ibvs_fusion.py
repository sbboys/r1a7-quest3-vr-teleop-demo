#!/usr/bin/env python3

import json
import time
from pathlib import Path


EXT_FILE = Path(
"/tmp/r1a7_ibvs_velocity.json"
)

WRIST_FILE = Path(
"/tmp/r1a7_wrist_ibvs_velocity.json"
)

OUT_FILE = Path(
"/tmp/r1a7_dual_ibvs_velocity.json"
)


EXT_Z_WEIGHT = 0.7
WRIST_Z_WEIGHT = 0.3



def read_json(path):

    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text()
        )

    except:
        return {}



def main():

    print(
        "[DUAL IBVS FUSION START]",
        flush=True
    )


    while True:

        ext = read_json(
            EXT_FILE
        )

        wrist = read_json(
            WRIST_FILE
        )


        vx = float(
            ext.get(
                "vx_mm_s",
                0
            )
        )


        vy = float(
            wrist.get(
                "vy_mm_s",
                0
            )
        )


        vz_ext = float(
            ext.get(
                "vz_mm_s",
                0
            )
        )


        vz_wrist = float(
            wrist.get(
                "vz_mm_s",
                0
            )
        )


        vz = (
            EXT_Z_WEIGHT*vz_ext
            +
            WRIST_Z_WEIGHT*vz_wrist
        )


        out={

            "enabled":True,

            "vx_mm_s":vx,

            "vy_mm_s":vy,

            "vz_mm_s":vz,

            "external":ext,

            "wrist":wrist,

            "timestamp":time.time()

        }


        OUT_FILE.write_text(
            json.dumps(
                out,
                indent=2
            )
        )


        print(
            "[DUAL IBVS]",
            f"vx={vx:+.2f}",
            f"vy={vy:+.2f}",
            f"vz={vz:+.2f}",
            flush=True
        )


        time.sleep(
            0.05
        )


if __name__=="__main__":
    main()
