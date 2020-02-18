import unittest
import numpy as np
import river_generator as rg
from pprint import pprint

tc = unittest.TestCase()


def test_river_create_env():
    nx = 4
    ny = 3
    env = rg.create_env(nx, ny, 0.8)
    pprint(env)

    keys = env.keys()

    # assert bridge keys
    for i in range((ny - 1) * nx + 1, nx * ny):
        assert str(i) in keys

    # assert bank keys
    for i in range(1, ny - 1):
        assert str((i * nx) + 1) in keys
        assert str((i * 1) + nx) in keys
    assert str(nx * ny) in keys

    # assert waterfall keys
    begin = 1 + (ny - 1) * nx
    end = nx * ny - 1
    for i in range(begin, end + 1):
        tc.assertDictEqual(env[str(i)], {
            'goal': False,
            'Adj': [{
                'name': str(i),
                'A': {
                    'N': 1,
                    'S': 1,
                    'E': 1,
                    'W': 1
                }
            }]
        })

    # assert river keys
    for i in range(1, ny - 1):
        for j in range(nx - 1):
            assert str(i * nx + j + 1) in keys
