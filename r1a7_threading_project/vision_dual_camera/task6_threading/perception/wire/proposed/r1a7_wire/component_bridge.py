#!/usr/bin/env python3

import cv2
import numpy as np


class ComponentBridge:


    def __init__(
        self,
        max_distance=15
    ):

        self.max_distance=max_distance



    def get_endpoints(
        self,
        component
    ):

        points=np.column_stack(
            np.where(
                component>0
            )
        )

        # y,x -> x,y

        points=points[:,::-1]


        if len(points)==0:
            return []


        endpoints=[]


        point_set=set(
            map(
                tuple,
                points
            )
        )


        for p in points:

            x,y=p

            n=0

            for dx in [-1,0,1]:

                for dy in [-1,0,1]:

                    if dx==0 and dy==0:
                        continue


                    if (
                        x+dx,
                        y+dy
                    ) in point_set:

                        n+=1


            if n==1:

                endpoints.append(
                    p
                )


        return endpoints



    def connect_line(
        self,
        image,
        p1,
        p2
    ):

        cv2.line(
            image,
            tuple(
                map(
                    int,
                    p1
                )
            ),
            tuple(
                map(
                    int,
                    p2
                )
            ),
            255,
            1
        )



    def bridge(
        self,
        skeleton
    ):


        result=skeleton.copy()


        num,labels,stats,_=(
            cv2.connectedComponentsWithStats(
                skeleton,
                8
            )
        )


        components=[]


        for i in range(1,num):

            comp=np.zeros_like(
                skeleton
            )


            comp[
                labels==i
            ]=255


            components.append(
                comp
            )


        # component数量少于2无需处理

        if len(components)<2:

            return result



        # 获取所有component endpoint

        endpoint_list=[]


        for idx,comp in enumerate(components):

            eps=self.get_endpoints(
                comp
            )


            for e in eps:

                endpoint_list.append(
                    (
                        idx,
                        e
                    )
                )


        # 最近端点连接

        used=set()


        for i in range(
            len(endpoint_list)
        ):

            c1,p1=endpoint_list[i]


            best=None
            best_d=9999


            for j in range(
                len(endpoint_list)
            ):

                if i==j:
                    continue


                c2,p2=endpoint_list[j]


                if c1==c2:
                    continue


                d=np.linalg.norm(
                    p1-p2
                )


                if (
                    d<best_d
                    and
                    d<self.max_distance
                ):

                    best_d=d

                    best=(
                        c2,
                        p2
                    )


            if best is not None:

                c2,p2=best


                key=tuple(
                    sorted(
                        [
                            c1,
                            c2
                        ]
                    )
                )


                if key not in used:

                    self.connect_line(
                        result,
                        p1,
                        p2
                    )

                    used.add(
                        key
                    )


        return result



def bridge_components(
    skeleton
):

    bridge=ComponentBridge()

    return bridge.bridge(
        skeleton
    )
