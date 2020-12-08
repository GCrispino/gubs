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
    print(' ', C, s, a, s_a_cost, sum([
        V[V_i[s_['name']], C + c_] * s_['A'][a] for s_ in reachable]), s_a_cost + sum([
            V[V_i[s_['name']], C + c_] * s_['A'][a] for s_ in reachable]))
    return s_a_cost + sum([
        V[V_i[s_['name']], C + c_] * s_['A'][a] for s_ in reachable])


def get_probabilities_finite(V_i, pi, S, C_max, H, mdp, c=1):
    P = np.zeros((len(S), C_max + c + 1, H + 1))
    G = [V_i[i] for i, s in mdp.items() if s['goal']]
    P[G] = 1
    print(S)

    for h in reversed(range(H)):
        for C in reversed(range(C_max + 1)):
            for s in S:
                i_s = V_i[s]
                try:
                    a = pi[i_s, C, h]
                except IndexError:
                    a = pi[i_s, C, -1]
                reachable = find_reachable(s, a, mdp)
        #        if s == '20':
        #            print(i_s, s, a, reachable, P[V_i['20'], C + c_, h + 1])
                c_ = 0 if mdp[s]['goal'] else c
                P[V_i[s], C, h] = np.sum([s_['A'][a] * P[V_i[s_['name']], C + c_, h + 1]
                                          for s_ in reachable])
        #print(h, P[19, 0, h])
    return P


def get_avg_cost_finite(V_i, pi, S, C_max, H, mdp, c=1):
    V_cost = np.zeros((len(S), C_max + c + 1, H + 1))

    for h in reversed(range(H)):
        for C in reversed(range(C_max + 1)):
            for s in S:
                i_s = V_i[s]
                try:
                    a = pi[i_s, C, h]
                except IndexError:
                    a = pi[i_s, C, -1]
                reachable = find_reachable(s, a, mdp)
                c_ = 0 if mdp[s]['goal'] else c
                V_cost[V_i[s], C, h] = c_ + np.sum([s_['A'][a] * V_cost[V_i[s_['name']], C + c_, h + 1]
                                                    for s_ in reachable])
    return V_cost


def get_probabilities(V_i, pi, S, mdp, epsilon=1e-3):
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
