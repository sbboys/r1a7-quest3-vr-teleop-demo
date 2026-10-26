#!/usr/bin/env python3

import cv2
import numpy as np
from skimage.morphology import skeletonize


class SkeletonTracker:


    def __init__(self):
        pass


    def skeleton(self, mask):

        binary = (
            mask > 0
        )


        skel = skeletonize(
            binary
        )


        return (
            skel.astype(
                np.uint8
            )
            *
            255
        )


    def extract_points(self, skel):

        ys,xs=np.where(
            skel>0
        )


        points=[]

        for x,y in zip(xs,ys):

            points.append(
                [
                    int(x),
                    int(y)
                ]
            )


        return points



    def endpoint(self, skel):

        points=[]


        ys,xs=np.where(
            skel>0
        )


        pixel=set(
            zip(xs,ys)
        )


        for x,y in pixel:


            neighbors=0


            for dx in [-1,0,1]:

                for dy in [-1,0,1]:

                    if dx==0 and dy==0:
                        continue


                    if (
                        x+dx,
                        y+dy
                    ) in pixel:

                        neighbors+=1


            # skeleton endpoint
            if neighbors==1:

                points.append(
                    [
                        x,
                        y
                    ]
                )


        return points



    def process(self, mask):

        skel=self.skeleton(
            mask
        )


        centerline=self.extract_points(
            skel
        )


        endpoints=self.endpoint(
            skel
        )


        return {

            "centerline":
                centerline,

            "endpoints":
                endpoints,

            "skeleton":
                skel
        }



def track_wire(mask):

    tracker=SkeletonTracker()

    return tracker.process(
        mask
    )
