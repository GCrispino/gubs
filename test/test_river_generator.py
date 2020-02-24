import unittest
import numpy as np
import river_generator as rg

tc = unittest.TestCase()


def test_river_create_env():
    nx = 4
    ny = 3
    p = 0.8
    env = rg.create_env(nx, ny, p)
    expected = {
        '1': {
            'goal': False,
            'heuristic': 5,
            'Adj': [
                {
                    'name': '1',
                    'A': {'N': 1, 'W': 1}
                },
                {
                    'name': '5',
                    'A': {'S': 1}
                },
                {
                    'name': '2',
                    'A': {'E': 1}
                }
            ]
        },
        '2': {
            'goal': False,
            'heuristic': 4,
            'Adj': [
                {
                    'name': '2',
                    'A': {'N': 1}
                },
                {
                    'name': '6',
                    'A': {'S': 1}
                },
                {
                    'name': '3',
                    'A': {'E': 1}
                },
                {
                    'name': '1',
                    'A': {'W': 1}
                }
            ]
        },
        '3': {
            'goal': False,
            'heuristic': 3,
            'Adj': [
                {
                    'name': '3',
                    'A': {'N': 1}
                },
                {
                    'name': '7',
                    'A': {'S': 1}
                },
                {
                    'name': '4',
                    'A': {'E': 1}
                },
                {
                    'name': '2',
                    'A': {'W': 1}
                }
            ]
        },
        '4': {
            'goal': False,
            'heuristic': 2,
            'Adj': [
                {
                    'name': '4',
                    'A': {'N': 1, 'E': 1}
                },
                {
                    'name': '8',
                    'A': {'S': 1}
                },
                {
                    'name': '3',
                    'A': {'W': 1}
                }
            ]
        },
        '5': {
            'goal': False,
            'heuristic': 4,
            'Adj': [
                {
                    'name': '5',
                    'A': {'W': 1}
                },
                {
                    'name': '6',
                    'A': {'E': 1}
                },
                {
                    'name': '1',
                    'A': {'N': 1}
                },
                {
                    'name': '9',
                    'A': {'S': 1}
                }
            ]
        },
        '6': {
            'goal': False,
            'heuristic': 3,
            'Adj': [
                {
                    'name': '10',
                    'A': {'N': p, 'S': 1, 'E': p, 'W': p}
                },
                {
                    'name': '2',
                    'A': {'N': 1 - p}
                },
                {
                    'name': '7',
                    'A': {'E': 1 - p}
                },
                {
                    'name': '5',
                    'A': {'W': 1 - p}
                }
            ]
        },
        '7': {
            'goal': False,
            'heuristic': 2,
            'Adj': [
                {
                    'name': '11',
                    'A': {'N': p, 'S': 1, 'E': p, 'W': p}
                },
                {
                    'name': '3',
                    'A': {'N': 1 - p}
                },
                {
                    'name': '8',
                    'A': {'E': 1 - p}
                },
                {
                    'name': '6',
                    'A': {'W': 1 - p}
                }
            ]
        },
        '8': {
            'goal': False,
            'heuristic': 1,
            'Adj': [
                {
                    'name': '8',
                    'A': {'E': 1}
                },
                {
                    'name': '7',
                    'A': {'W': 1}
                },
                {
                    'name': '4',
                    'A': {'N': 1}
                },
                {
                    'name': '12',
                    'A': {'S': 1}
                }
            ]
        },
        '9': {
            'goal': False,
            'heuristic': 3,
            'Adj': [
                {
                    'name': '9',
                    'A': {'N': 1, 'S': 1, 'E': 1, 'W': 1}
                }
            ]
        },
        '10': {
            'goal': False,
            'heuristic': 2,
            'Adj': [
                {
                    'name': '10',
                    'A': {'N': 1, 'S': 1, 'E': 1, 'W': 1}
                }
            ]
        },
        '11': {
            'goal': False,
            'heuristic': 1,
            'Adj': [
                {
                    'name': '11',
                    'A': {'N': 1, 'S': 1, 'E': 1, 'W': 1}
                }
            ]
        },
        '12': {
            'goal': True,
            'heuristic': 0,
            'Adj': [
                {
                    'name': '12',
                    'A': {'N': 1, 'S': 1, 'E': 1, 'W': 1}
                }
            ]
        }
    }

    tc.maxDiff = None
    tc.assertDictEqual(env, expected)
