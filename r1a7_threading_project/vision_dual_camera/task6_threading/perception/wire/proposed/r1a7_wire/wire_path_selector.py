#!/usr/bin/env python3

import cv2
import numpy as np
import json
from scipy.spatial import cKDTree


class WirePathSelector:


    def __init__(
        self,
        tip_hint=(33,427),
        grasp_hint=(237,299)
    ):

        self.tip_hint=np.array(
            tip_hint
        )

        self.grasp_hint=np.array(
            grasp_hint
        )



    def get_points(self, skeleton):

        y,x=np.where(
            skeleton>0
        )

        points=np.stack(
            [
                x,
                y
            ],
            axis=1
        )

        return points



    def distance_score(
        self,
        points,
        target
    ):

        tree=cKDTree(
            points
        )

        d,_=tree.query(
            target
        )

        return d



    def select(
        self,
        skeleton
    ):


        points=self.get_points(
            skeleton
        )


        if len(points)==0:

            return []



        # 找距离人工点击点最近的骨架点

        tip_id=np.argmin(
            np.linalg.norm(
                points-
                self.tip_hint,
                axis=1
            )
        )


        grasp_id=np.argmin(
            np.linalg.norm(
                points-
                self.grasp_hint,
                axis=1
            )
        )


        tip_point=points[tip_id]

        grasp_point=points[grasp_id]



        # 构造简单连线约束:
        # 保留距离两端连线附近区域


        vector=(
            grasp_point-
            tip_point
        )

        length=np.linalg.norm(
            vector
        )


        if length<1:

            return []



        direction=(
            vector/
            length
        )


        selected=[]


        for p in points:


            proj=np.dot(
                p-tip_point,
                direction
            )


            if (
                proj>=-30
                and
                proj<=length+100
            ):


                closest=(
                    tip_point+
                    proj*direction
                )


                error=np.linalg.norm(
                    p-closest
                )


                # 距离路径 <30 px

                if error<30:

                    selected.append(
                        p.tolist()
                    )


        return selected



def select_wire_path(
    skeleton,
    tip_hint=(33,427),
    grasp_hint=(237,299)
):

    selector=WirePathSelector(
        tip_hint,
        grasp_hint
    )

    return selector.select(
        skeleton
    )


def load_wire_init(
    filename="wire_init.json"
):

    with open(
        filename,
        "r"
    ) as f:

        data=json.load(f)


    return (
        tuple(data["tip"]),
        tuple(data["grasp"])
    )
