import numpy as np
from utils import read_json, parse_args
from gubs import gubs

c_max = 10

args = parse_args()
mdp = read_json(args.file_input)

k_g = 0
l = 0.1


def u(c): return np.exp(-l * c)


V = gubs(c_max, u, k_g, mdp)
print('V: ', V)
