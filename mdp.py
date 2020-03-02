import numpy as np


def flatten(l):
    return [x for l_i in l for x in l_i]


def get_actions(mdp):
    adjs = map(lambda s: s['Adj'], mdp.values())
    actions = map(lambda s: list(s['A'].keys()), flatten(adjs))
    return list(set(flatten(actions)))


def find_reachable(s, a, mdp):
    """ Find states that are reachable from state 's' after executing action 'a' """
    all_reachable_from_s = mdp[s]['Adj']
    return list(filter(
        lambda obj_s_: a in obj_s_['A'],
        all_reachable_from_s
    ))


def Q(s, C, a, u, V, V_i, mdp, c=1):
    reachable = find_reachable(s, a, mdp)
    c_ = 0 if mdp[s]['goal'] else c
    s_a_cost = u(C + c_) - u(C)
    return s_a_cost + sum([
        V[V_i[s_['name']], C + c_] * s_['A'][a] for s_ in reachable])


def get_probabilities(V_i, V, pi, S, mdp, epsilon=1e-3):
    P = np.zeros(len(S))
    G = [V_i[i] for i, s in mdp.items() if s['goal']]
    P[G] = 1
    i = 0
    while True:
        P_ = np.array(P)
        for s in S:
            if mdp[s]['goal']:
                continue
            a = pi[V_i[s], 0]
            reachable = find_reachable(s, a, mdp)
            P_[V_i[s]] = np.sum([s_['A'][a] * P[V_i[s_['name']]]
                                 for s_ in reachable])

        if np.linalg.norm(P_ - P, np.inf) < epsilon:
            break
        P = P_
        i += 1

    return P
