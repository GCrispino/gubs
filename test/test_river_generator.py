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
    for i in range(1, nx + 1):
        assert str(i) in keys
        assert env[str(i)]['heuristic'] == ny + (nx - i) - 1

    # assert bank keys
    for i in range(1, ny - 1):
        print('bank: ', (i * nx) + 1, (i * nx) + nx)
        assert str((i * nx) + 1) in keys
        assert env[str((i * nx) + 1)]['heuristic'] == nx + (ny - (i // ny) - 1)
        assert str((i * nx) + nx) in keys
        assert env[str((i * nx) + nx)]['heuristic'] == ny - (i // ny) - 2
    assert str(nx * ny) in keys

    # assert waterfall keys
    begin = 1 + (ny - 1) * nx
    end = nx * ny - 1
    for i in range(begin, end + 1):
        tc.assertDictEqual(env[str(i)], {
            'goal': False,
            'heuristic': (end - i),
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
        for j in range(1, nx - 1):
            k = str(i * nx + j + 1)
            assert k in keys
            assert env[k]['heuristic'] == ny - i + nx - j
