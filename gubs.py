import numpy as np
from numpy.linalg import norm
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


def risk_sensitive(lamb, mdp_obj, V_i, S, c=1, epsilon=1e-3):
    def u(c): return np.exp(lamb * c)

    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    not_goal = [i for i, s in mdp_obj.items() if not s['goal']]
    n_states = len(S)

    # initialize
    V = np.zeros(n_states, dtype=float)
    pi = np.full(n_states, None)
    P = np.zeros(n_states, dtype=float)
    V[G] = -np.sign(lamb)
    P[G] = 1
    A = np.array(mdp.get_actions(mdp_obj))

    i = 0

    P_not_max_prob = np.copy(P)
    while True:
        V_ = np.copy(V)
        P_ = np.copy(P)
        for s in not_goal:
            actions_results_p = np.array([
                np.sum([
                    P[V_i[s_['name']]] * s_['A'][a] for s_ in mdp.find_reachable(s, a, mdp_obj)
                ]) for a in A
            ])

            # set maxprob
            max_prob = np.max(actions_results_p)
            P_[V_i[s]] = max_prob
            A_max_prob = A[actions_results_p == max_prob]
            A_not_max_prob = A[actions_results_p != max_prob]
            not_max_prob_actions_results = np.array([
                np.sum([
                    P[V_i[s_['name']]] * s_['A'][a] for s_ in mdp.find_reachable(s, a, mdp_obj)
                ]) for a in A_not_max_prob
            ])

            # record maxprob obtained by actions that are in A_not_max_prob
            P_not_max_prob[V_i[s]] = P[V_i[s]] if len(not_max_prob_actions_results) == 0 else np.max(
                not_max_prob_actions_results)

            actions_results = np.array([
                np.sum([
                    V[V_i[s_['name']]] * s_['A'][a] for s_ in mdp.find_reachable(s, a, mdp_obj)
                ]) for a in A_max_prob
            ])

            i_a = np.argmax(actions_results)
            V_[V_i[s]] = u(c) * actions_results[i_a]
            pi[V_i[s]] = A_max_prob[i_a]

        v_norm = norm(V_ - V, np.inf)
        p_norm = norm(P_ - P, np.inf)

        P_diff = P_ - P_not_max_prob
        arg_min_p_diff = np.argmin(P_diff)
        min_p_diff = P_diff[arg_min_p_diff]

        if v_norm + p_norm < epsilon and min_p_diff >= 0:
            break
        V = V_
        P = P_
        i += 1

    print(f'{i} iterations')
    return V, P, pi
