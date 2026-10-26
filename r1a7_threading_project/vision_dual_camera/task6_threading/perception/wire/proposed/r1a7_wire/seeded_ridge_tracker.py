#!/usr/bin/env python3

import cv2
import json
import heapq
import numpy as np

from skimage.filters import frangi


class SeededRidgeTracker:


    def __init__(
        self,
        init_file="wire_init.json"
    ):

        with open(
            init_file,
            "r"
        ) as f:

            data=json.load(f)


        self.tip=np.array(
            data["tip"],
            dtype=float
        )

        self.grasp=np.array(
            data["grasp"],
            dtype=float
        )



    def ridge_response(
        self,
        image
    ):

        gray=cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        clahe=cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8,8)
        )


        enhanced=clahe.apply(
            gray
        )


        ridge=frangi(
            enhanced,
            sigmas=range(1,5),
            black_ridges=False
        )


        ridge=cv2.normalize(
            ridge,
            None,
            0,
            1,
            cv2.NORM_MINMAX
        )


        return ridge



    def build_cost(
        self,
        ridge
    ):

        # ridge越强，代价越低

        cost=1.0-ridge


        return cost+0.01



    def nearest_pixel(
        self,
        point,
        shape
    ):

        x=int(
            np.clip(
                point[0],
                0,
                shape[1]-1
            )
        )

        y=int(
            np.clip(
                point[1],
                0,
                shape[0]-1
            )
        )


        return (
            x,
            y
        )



    def search(
        self,
        cost
    ):

        h,w=cost.shape


        start=self.nearest_pixel(
            self.tip,
            cost.shape
        )


        goal=self.nearest_pixel(
            self.grasp,
            cost.shape
        )


        pq=[]


        heapq.heappush(
            pq,
            (
                0,
                start
            )
        )


        dist={
            start:0
        }


        parent={}


        while pq:


            current_cost,current=heapq.heappop(
                pq
            )


            if current==goal:

                break


            x,y=current


            for dx,dy in [
                (-1,0),
                (1,0),
                (0,-1),
                (0,1),
                (-1,-1),
                (-1,1),
                (1,-1),
                (1,1)
            ]:


                nx=x+dx
                ny=y+dy


                if (
                    nx<0 or
                    ny<0 or
                    nx>=w or
                    ny>=h
                ):
                    continue


                step=np.sqrt(
                    dx*dx+dy*dy
                )


                new_cost=(
                    current_cost
                    +
                    cost[ny,nx]
                    *
                    step
                )


                node=(nx,ny)


                if (
                    node not in dist
                    or
                    new_cost<dist[node]
                ):

                    dist[node]=new_cost

                    parent[node]=current

                    heapq.heappush(
                        pq,
                        (
                            new_cost,
                            node
                        )
                    )


        if goal not in parent:

            return []


        path=[]

        cur=goal


        while cur!=start:

            path.append(
                [
                    cur[0],
                    cur[1]
                ]
            )

            cur=parent[cur]


        path.append(
            [
                start[0],
                start[1]
            ]
        )


        path.reverse()


        return path



    def track(
        self,
        image
    ):

        ridge=self.ridge_response(
            image
        )


        cost=self.build_cost(
            ridge
        )


        path=self.search(
            cost
        )


        return {

            "centerline":
                path,

            "ridge":
                ridge

        }



def track_seeded_ridge(
    image
):

    tracker=SeededRidgeTracker()

    return tracker.track(
        image
    )
