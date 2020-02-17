import numpy as np
import utils
from gubs import gubs


args = utils.parse_args()
c_max = args.c_max
l = args.lamb
kg = args.kg
mdp = utils.read_json(args.file_input)


def u(c): return np.exp(-l * c)


V, pi = gubs(c_max, u, kg, mdp)
print('V: ', V)
print('pi: ', pi)

if args.output:
    output_file_path = utils.output({'V': V.tolist(), 'pi': pi.tolist()})
    if output_file_path:
        print("Algorithm result written to ", output_file_path)
