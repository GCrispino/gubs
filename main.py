from datetime import datetime
import numpy as np
import utils
from gubs import gubs


def try_int(key):
    try:
        return int(key)
    except:
        return key


args = utils.parse_args()
c_max = args.c_max
l = args.lamb
kg = args.kg
mdp = utils.read_json(args.file_input)


def u(c): return np.exp(-l * c)


S = sorted(mdp.keys(), key=try_int)
V_i = {S[i]: i for i in range(len(S))}
V, pi = gubs(c_max, u, kg, mdp, V_i, S)
print('V: ', V)
print('pi: ', pi)
V_ = np.array(V)

if args.output:
    output_filename = str(datetime.time(datetime.now())) + '.json'
    output_file_path = utils.output(
        output_filename, {'V': V.tolist(), 'pi': pi.tolist()})
    if output_file_path:
        print("Algorithm result written to ", output_file_path)
