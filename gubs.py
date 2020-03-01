import numpy as np
import itertools
import mdp


def initialize(C_max, u, k_g, mdp_obj, V_i, S, c=1):
    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    not_goal = [V_i[i] for i, s in mdp_obj.items() if not s['goal']]
    n_states = len(G) + len(not_goal)

    C = np.arange(c + 1) + C_max
    V = np.full((n_states, C_max + c + 1), -1.0)
    pi = np.full((n_states, C_max + c + 1), None)
    V[G] = k_g
    not_goal_and_C = np.array([[i, j]
                               for i, j in itertools.product(not_goal, C)]).T
    not_goal_i, C_i = not_goal_and_C
    V[not_goal_i, C_i] = ((u(np.full(C.shape, np.inf)) -
                           u(C)) * np.ones((len(not_goal), len(C)))).flatten()

    return V, pi


def gubs(C_max, u, k_g, mdp_obj, V_i, S, c=1):
    A = mdp.get_actions(mdp_obj)

    V, pi = initialize(C_max, u, k_g, mdp_obj, V_i, S, c)
    for C in reversed(range(C_max)):
        for s in S:
            actions_results = np.array(
                [mdp.Q(s, C, a, u, V, V_i, mdp_obj, c) for a in A])
            i_max = np.argmax(actions_results)
            pi[V_i[s], C] = A[i_max]
            V[V_i[s], C] = actions_results[i_max]

    return V, pi
