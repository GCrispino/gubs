import unittest
import numpy as np
import mdp
import gubs
from utils import read_json

tc = unittest.TestCase()
env = read_json('env1-reduced.json')


def test_Q():
    C_max = 5
    l = 0.1
    def u(c): return np.exp(-l * c)
    k_g = 0
    V = gubs.initialize(C_max, u, k_g, env)
    print("V: ", V)
    S = list(env.keys())
    V_i = {S[i]: i for i in range(len(S))}

    # run Q() for C=C_max
    q1N = mdp.Q('1', C_max, 'N', u, V, V_i, env)
    q1S = mdp.Q('1', C_max, 'S', u, V, V_i, env)
    q1E = mdp.Q('1', C_max, 'E', u, V, V_i, env)

    q2N = mdp.Q('2', C_max, 'N', u, V, V_i, env)
    q2S = mdp.Q('2', C_max, 'S', u, V, V_i, env)
    q2E = mdp.Q('2', C_max, 'E', u, V, V_i, env)

    q3N = mdp.Q('3', C_max, 'N', u, V, V_i, env)
    q3S = mdp.Q('3', C_max, 'S', u, V, V_i, env)
    q3E = mdp.Q('3', C_max, 'E', u, V, V_i, env)

    tc.assertListEqual([q1N, q1S, q1E], [V[V_i['1'], 5]] * 3)
    np.testing.assert_almost_equal(np.array([q2N, q2S, q2E]), np.array(
        [V[V_i['1'], 5], V[V_i['1'], 5], -0.33212484166562023]))
    tc.assertListEqual([q3N, q3S, q3E], [0.0, 0.0, 0.0])
