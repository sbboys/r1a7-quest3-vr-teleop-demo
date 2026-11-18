#!/usr/bin/env python3

from lower_branch_free_tracker_v1 import (
    LowerBranchFreeTracker,
)


def check(
    name,
    cond,
):
    if not cond:
        raise AssertionError(name)

    print("[PASS]", name)


# ============================================================
# TEST 1
# Normal lower branch.
# ============================================================

t = LowerBranchFreeTracker(
    prediction_gate_px=30.0,
)

t.initialize(
    grasp_uv=(100, 100),
    free_uv=(100, 160),
)

r = t.choose_candidate(
    grasp_uv=(100, 100),
    endpoint_candidates=[
        (100, 161),
        (100, 60),      # opposite branch
    ],
)

check(
    "normal lower endpoint accepted",
    r["identity_valid"],
)

check(
    "upper endpoint excluded",
    r["branch_projection_px"] > 0,
)


# ============================================================
# TEST 2
# Entire grasped wire translates.
# ============================================================

r = t.choose_candidate(
    grasp_uv=(108, 105),
    endpoint_candidates=[
        (108, 166),
        (108, 66),
    ],
)

check(
    "whole wire translation follows grasp motion",
    r["identity_valid"],
)


# ============================================================
# TEST 3
# Wrong upper-half point is close in Euclidean space.
# Must still be rejected.
# ============================================================

t2 = LowerBranchFreeTracker(
    prediction_gate_px=100.0,
)

t2.initialize(
    grasp_uv=(200, 200),
    free_uv=(205, 250),
)

r = t2.choose_candidate(
    grasp_uv=(200, 200),
    endpoint_candidates=[
        (201, 180),
    ],
)

check(
    "upper-half identity switch rejected",
    not r["identity_valid"],
)


# ============================================================
# TEST 4
# Implausible motion on correct branch.
# ============================================================

t3 = LowerBranchFreeTracker(
    prediction_gate_px=15.0,
)

t3.initialize(
    grasp_uv=(50, 50),
    free_uv=(50, 100),
)

r = t3.choose_candidate(
    grasp_uv=(50, 50),
    endpoint_candidates=[
        (90, 150),
    ],
)

check(
    "large motion residual rejected",
    not r["identity_valid"],
)


# ============================================================
# TEST 5
# Temporary loss -> PREDICT_ONLY.
# ============================================================

t4 = LowerBranchFreeTracker(
    lost_after=3,
)

t4.initialize(
    grasp_uv=(0, 0),
    free_uv=(0, 50),
)

r = t4.choose_candidate(
    grasp_uv=(0, 0),
    endpoint_candidates=[],
)

check(
    "first miss becomes PREDICT_ONLY",
    r["state"] == "PREDICT_ONLY",
)

check(
    "prediction never becomes control measurement",
    r["free_uv_control"] is None,
)


# ============================================================
# TEST 6
# Persistent loss -> LOST.
# ============================================================

t4.choose_candidate(
    grasp_uv=(0, 0),
    endpoint_candidates=[],
)

r = t4.choose_candidate(
    grasp_uv=(0, 0),
    endpoint_candidates=[],
)

check(
    "persistent loss becomes LOST",
    r["state"] == "LOST",
)


# ============================================================
# TEST 7
# Two cameras maintain independent states.
# ============================================================

ext = LowerBranchFreeTracker()
left = LowerBranchFreeTracker()

ext.initialize(
    grasp_uv=(100, 100),
    free_uv=(100, 150),
)

left.initialize(
    grasp_uv=(300, 100),
    free_uv=(280, 170),
)

re = ext.choose_candidate(
    grasp_uv=(101, 101),
    endpoint_candidates=[
        (101, 151),
    ],
)

rl = left.choose_candidate(
    grasp_uv=(301, 101),
    endpoint_candidates=[
        (281, 171),
    ],
)

check(
    "external independent state",
    re["identity_valid"],
)

check(
    "left independent state",
    rl["identity_valid"],
)

check(
    "camera states are not shared",
    ext.free_uv != left.free_uv,
)


print()
print("ALL LOWER-BRANCH FREE TRACKER TESTS PASSED")
