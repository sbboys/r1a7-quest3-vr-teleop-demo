from src.motion.trajectory_generator import quintic_interpolate


def test_quintic_interpolate_hits_endpoints():
    path = quintic_interpolate([0.0, 1.0], [1.0, 3.0], 1.0, 0.1)
    assert path[0] == [0.0, 1.0]
    assert path[-1] == [1.0, 3.0]
    assert len(path) == 11
