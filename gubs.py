import numpy as np
import itertools
import mdp


def initialize(C_max, u, k_g, mdp_obj, c=1):
    S = list(mdp_obj.keys())
    V_i = {S[i]: i for i in range(len(S))}
    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    not_goal = [V_i[i] for i, s in mdp_obj.items() if not s['goal']]
    n_states = len(G) + len(not_goal)

    C = np.arange(c + 1) + C_max
    V = np.full((n_states, C_max + c + 1), -1.0)
    V[G] = k_g
    not_goal_and_C = np.array([[i, j]
                               for i, j in itertools.product(not_goal, C)]).T
    not_goal_i, C_i = not_goal_and_C
    V[not_goal_i, C_i] = ((u(np.full(C.shape, np.inf)) -
                           u(C)) * np.ones((len(not_goal), len(C)))).flatten()

    return V


def gubs(C_max, u, k_g, mdp_obj, c=1):
    S = list(mdp_obj.keys())
    A = mdp.get_actions(mdp_obj)
    V_i = {S[i]: i for i in range(len(S))}
    not_goal = [V_i[i] for i, s in mdp_obj.items() if not s['goal']]

    V = initialize(C_max, u, k_g, mdp_obj, c)
    for C in reversed(range(C_max)):
        for s in S:
            V[V_i[s], C] = max(
                [mdp.Q(s, C, a, u, V, V_i, mdp_obj, c) for a in A])

    return V
