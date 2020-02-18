import argparse
import json
import os
import numpy as np
from utils import output


def create_state_obj(adjs, goal=False):
    obj = {
        'goal': goal,
        'Adj': adjs
    }
    return obj


def get_bank_adj(i, nx, ny):
    return [
        {
            'name': i,
            'A': {
                **({'E': 1} if int(i) % nx == 0 else {'W': 1})
            }
        },
        *([{
            'name': str(int(i) + 1),
            'A': {'E': 1}
        }] if int(i) % nx != 0 else []),
        *([{
            'name': str(int(i) - 1),
            'A': {'W': 1}
        }] if int(i) % nx == 0 else []),
        {
            'name': str(int(i) - nx),
            'A': {'N': 1}
        },
        {
            'name': str(int(i) + nx),
            'A': {'S': 1}
        }

    ]


def add_bridge_states(env, nx, ny):
    for i in range(1, nx + 1):
        env[str(i)] = {
            'goal': False,
            'Adj': [
                {
                    'name': str(i),
                    'A': {
                        'N': 1,
                        **({'W': 1} if i == 1 else {}),
                        ** ({'E': 1} if i == nx else {})
                    }
                },
                {
                    'name': str(i + nx),
                    'A': {'S': 1}
                },
                *([{
                    'name': str(i + 1),
                    'A': {'E': 1}
                }] if i != nx else []),
                *([{
                    'name': str(i - 1),
                    'A': {'W': 1}
                }] if i != 1 else []),
            ]
        }
    return env


def add_bank_states(env, nx, ny):
    for i in range(1, ny - 1):
        cell_i_1 = str(i * nx + 1)
        env[cell_i_1] = create_state_obj(get_bank_adj(cell_i_1, nx, ny))
        cell_i_2 = str((i + 1) * nx)
        env[cell_i_2] = create_state_obj(get_bank_adj(cell_i_2, nx, ny))
    goal_state_i = str(nx * ny)
    env[goal_state_i] = create_state_obj([
        {
            'name': goal_state_i,
            'A': {
                'N': 1,
                'S': 1,
                'E': 1,
                'W': 1,
            }
        }
    ], goal=True)

    return env


def add_river_states(env, nx, ny, p):
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            index = i * nx + j + 1
            env[str(index)] = {
                'goal': False,
                'Adj': [
                    {
                        # goes down the river
                        'name': str(index + nx),
                        'A': {
                            'N': p,
                            'S': 1,
                            'E': p,
                            'W': p
                        }
                    },
                    {
                        'name': str(index - nx),
                        'A': {'N': 1 - p}
                    },
                    {
                        'name': str(index + 1),
                        'A': {'E': 1 - p}
                    },
                    {
                        'name': str(index - 1),
                        'A': {'W': 1 - p}
                    }
                ]
            }
    return env


def add_waterfall_states(env, nx, ny):
    # começa no 1 + ny * nx
    begin = 1 + (ny - 1) * nx
    end = nx * ny - 1
    for i in range(begin, end + 1):
        env[str(i)] = {
            'goal': False,
            'Adj': [{
                'name': str(i),
                'A': {
                    'N': 1,
                    'S': 1,
                    'E': 1,
                    'W': 1,
                }
            }]
        }
    return env


def create_env(nx, ny, p):
    env = {}
    add_bridge_states(env, nx, ny)
    add_bank_states(env, nx, ny)
    add_river_states(env, nx, ny, p)
    add_waterfall_states(env, nx, ny)
    return env


DEFAULT_P = 0.4
DEFAULT_NX = 7
DEFAULT_NY = 10
DEFAULT_DEST_DIR = '.'

parser = argparse.ArgumentParser(
    description='River problem generator'
)
parser.add_argument('-p', dest='p', default=DEFAULT_P)
parser.add_argument('--nx', dest='nx', default=DEFAULT_NX)
parser.add_argument('--ny', dest='ny', default=DEFAULT_NY)
parser.add_argument('--dest_dir', dest='dest_dir', default=DEFAULT_DEST_DIR)

args = parser.parse_args()
nx = args.nx
ny = args.ny
p = args.p

env = create_env(nx, ny, p)

output('river%d-%d-%d.json' % (nx, ny, p * 100), env, args.dest_dir)
