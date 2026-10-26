#!/usr/bin/env python3

import numpy as np
from scipy.spatial import cKDTree
import heapq


class WireGraphTracker:


    def __init__(
        self,
        tip_hint,
        grasp_hint
    ):

        self.tip=np.array(
            tip_hint,
            dtype=float
        )

        self.grasp=np.array(
            grasp_hint,
            dtype=float
        )



    def skeleton_points(
        self,
        skeleton
    ):

        y,x=np.where(
            skeleton>0
        )

        pts=np.stack(
            [
                x,
                y
            ],
            axis=1
        )

        return pts



    def build_graph(
        self,
        points
    ):

        tree=cKDTree(
            points
        )


        graph={}


        for i,p in enumerate(points):

            graph[i]=[]


            ids=tree.query_ball_point(
                p,
                2.0
            )


            for j in ids:

                if i==j:
                    continue


                d=np.linalg.norm(
                    points[i]-points[j]
                )

                graph[i].append(
                    (
                        j,
                        d
                    )
                )


        return graph



    def nearest_point(
        self,
        points,
        target
    ):

        d=np.linalg.norm(
            points-target,
            axis=1
        )

        return int(
            np.argmin(d)
        )



    def shortest_path(
        self,
        graph,
        start,
        goal
    ):

        queue=[]

        heapq.heappush(
            queue,
            (0,start)
        )


        dist={
            start:0
        }


        parent={}


        while queue:

            cost,node=heapq.heappop(
                queue
            )


            if node==goal:
                break


            for nxt,w in graph[node]:

                new_cost=(
                    cost+w
                )


                if (
                    nxt not in dist
                    or
                    new_cost<dist[nxt]
                ):

                    dist[nxt]=new_cost

                    parent[nxt]=node

                    heapq.heappush(
                        queue,
                        (
                            new_cost,
                            nxt
                        )
                    )


        if goal not in parent:

            return []


        path=[]

        cur=goal


        while cur!=start:

            path.append(
                cur
            )

            cur=parent[cur]


        path.append(
            start
        )


        path.reverse()


        return path



    def track(
        self,
        skeleton
    ):


        points=self.skeleton_points(
            skeleton
        )


        if len(points)==0:

            return []


        graph=self.build_graph(
            points
        )


        start=self.nearest_point(
            points,
            self.tip
        )


        goal=self.nearest_point(
            points,
            self.grasp
        )


        ids=self.shortest_path(
            graph,
            start,
            goal
        )


        centerline=[]


        for i in ids:

            centerline.append(
                points[i].tolist()
            )


        return centerline



def track_wire_graph(
    skeleton,
    tip_hint,
    grasp_hint
):

    tracker=WireGraphTracker(
        tip_hint,
        grasp_hint
    )


    return tracker.track(
        skeleton
    )
