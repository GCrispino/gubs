import numpy as np
import itertools


def initialize(C_max, u, k_g, mdp, c=1):
    S = list(mdp.keys())
    V_i = {S[i]: i for i in range(len(S))}
    G = [V_i[i] for i, s in mdp.items() if s['goal']]
    not_goal = [V_i[i] for i, s in mdp.items() if not s['goal']]
    n_states = len(G) + len(not_goal)

    C = np.arange(c + 1) + C_max
    print(C, C.shape)
    V = np.full((n_states, C_max + c + 1), -1.0)
    print(V.shape, V)
    V[G] = k_g
    not_goal_and_C = np.array([[i, j]
                               for i, j in itertools.product(not_goal, C)]).T
    not_goal_i, C_i = not_goal_and_C
    V[not_goal_i, C_i] = ((u(np.full(C.shape, np.inf)) -
                           u(C)) * np.ones((2, len(C)))).flatten()

    return V
