import unittest
import numpy as np
import mdp
import gubs
from utils import read_json

tc = unittest.TestCase()


def test_initialize():
    C_max = 5
    l = 0.1
    def u(c): return np.exp(-l * c)
    k_g = 0
    env = read_json('env1-reduced.json')
    V, pi = gubs.initialize(C_max, u, k_g, env)
    np.testing.assert_array_equal(
        V[2], np.zeros(C_max + 2))
    np.testing.assert_almost_equal(
        V[0, -2:], np.array([-0.6065306597126334, -0.5488116360940264]))
    np.testing.assert_almost_equal(
        V[1, -2:], np.array([-0.6065306597126334, -0.5488116360940264]))


def test_initialize_2():
    C_max = 5
    l = 0.1
    def u(c): return np.exp(-l * c)
    k_g = 0
    env = read_json('env1-reduced.json')
    V, pi = gubs.initialize(C_max, u, k_g, env, c=2)
    np.testing.assert_array_equal(
        V[2], np.zeros(C_max + 3))
    np.testing.assert_almost_equal(
        V[0, -3:], np.array([-0.6065306597126334, -0.5488116360940264, -0.4965853]))
    np.testing.assert_almost_equal(
        V[1, -3:], np.array([-0.6065306597126334, -0.5488116360940264, -0.4965853]))
