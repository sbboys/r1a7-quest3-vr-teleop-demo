#!/usr/bin/env python3

import numpy as np


class EndpointDetector:


    def __init__(
        self,
        tip_hint=(33,427),
        grasp_hint=(237,299)
    ):

        self.tip_hint=np.array(
            tip_hint,
            dtype=float
        )

        self.grasp_hint=np.array(
            grasp_hint,
            dtype=float
        )



    def find_two_endpoints(
        self,
        centerline
    ):

        pts=np.array(
            centerline,
            dtype=float
        )


        if len(pts)<2:
            return []


        # 使用路径首尾作为候选
        return [
            pts[0],
            pts[-1]
        ]



    def assign_roles(
        self,
        endpoints
    ):

        if len(endpoints)!=2:

            return {
                "tip":None,
                "grasp":None
            }


        d0=np.linalg.norm(
            endpoints[0]
            -
            self.tip_hint
        )

        d1=np.linalg.norm(
            endpoints[1]
            -
            self.tip_hint
        )


        # 更接近人工tip的是线端

        if d0<d1:

            tip=endpoints[0]
            grasp=endpoints[1]

        else:

            tip=endpoints[1]
            grasp=endpoints[0]


        return {

            "tip":
                tip.tolist(),

            "grasp":
                grasp.tolist()

        }



    def direction(
        self,
        tip,
        grasp
    ):

        tip=np.array(
            tip,
            dtype=float
        )

        grasp=np.array(
            grasp,
            dtype=float
        )


        v=grasp-tip


        n=np.linalg.norm(
            v
        )


        if n<1e-6:

            return [
                0,
                0
            ]


        return (
            v/n
        ).tolist()



    def process(
        self,
        centerline
    ):


        endpoints=(
            self.find_two_endpoints(
                centerline
            )
        )


        result=(
            self.assign_roles(
                endpoints
            )
        )


        if (
            result["tip"] is not None
            and
            result["grasp"] is not None
        ):

            result["direction"]=(
                self.direction(
                    result["tip"],
                    result["grasp"]
                )
            )


        return result



def detect_endpoint(
    centerline,
    tip_hint=(33,427),
    grasp_hint=(237,299)
):

    detector=EndpointDetector(
        tip_hint,
        grasp_hint
    )


    return detector.process(
        centerline
    )
