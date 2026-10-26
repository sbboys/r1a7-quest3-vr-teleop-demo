#!/usr/bin/env python3

import cv2
import json
import numpy as np

from skimage.filters import frangi


class RidgeWireDetector:


    def __init__(self):

        self.clip_limit = 3.0

        self.sigmas = range(1,5)



    def load_init(
        self,
        filename="wire_init.json"
    ):

        with open(
            filename,
            "r"
        ) as f:

            data=json.load(f)


        return (
            np.array(data["tip"]),
            np.array(data["grasp"])
        )



    def create_roi(
        self,
        image,
        tip,
        grasp
    ):

        h,w=image.shape[:2]


        x1,y1=tip
        x2,y2=grasp


        xmin=max(
            0,
            min(x1,x2)-80
        )

        xmax=min(
            w,
            max(x1,x2)+120
        )


        ymin=max(
            0,
            min(y1,y2)-80
        )

        ymax=min(
            h,
            max(y1,y2)+80
        )


        return (
            image[
                ymin:ymax,
                xmin:xmax
            ],
            (
                xmin,
                ymin
            )
        )



    def preprocess(
        self,
        image
    ):

        gray=cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        clahe=cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(8,8)
        )


        return clahe.apply(
            gray
        )



    def remove_small_components(
        self,
        mask,
        min_area=15
    ):

        num,labels,stats,_=(
            cv2.connectedComponentsWithStats(
                mask,
                8
            )
        )


        output=np.zeros_like(
            mask
        )


        for i in range(1,num):

            area=stats[i,
                cv2.CC_STAT_AREA
            ]


            if area>=min_area:

                output[
                    labels==i
                ]=255


        return output



    def detect(
        self,
        image
    ):

        tip,grasp=self.load_init()


        roi,offset=self.create_roi(
            image,
            tip,
            grasp
        )


        enhanced=self.preprocess(
            roi
        )


        ridge=frangi(
            enhanced,
            sigmas=self.sigmas,
            black_ridges=False
        )


        ridge_img=cv2.normalize(
            ridge,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(
            np.uint8
        )


        # adaptive threshold
        threshold=np.percentile(
            ridge_img,
            92
        )


        _,mask=cv2.threshold(
            ridge_img,
            threshold,
            255,
            cv2.THRESH_BINARY
        )


        mask=self.remove_small_components(
            mask,
            15
        )


        kernel=cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5,5)
        )


        mask=cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )


        # restore full image size

        full_mask=np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )


        x0,y0=offset


        full_mask[
            y0:y0+mask.shape[0],
            x0:x0+mask.shape[1]
        ]=mask



        confidence=(
            float(
                np.mean(
                    ridge_img[mask>0]
                )/255
            )
            if np.any(mask)
            else 0.0
        )


        return {

            "wire_valid":
                bool(
                    np.sum(full_mask)>0
                ),

            "mask":
                full_mask,

            "confidence":
                confidence,

            "roi_offset":
                offset
        }



def detect_wire(image):

    detector=RidgeWireDetector()

    return detector.detect(
        image
    )
