import numpy as np
from numpy.linalg import norm
import itertools
import mdp


def initialize(C_max, u, k_g, mdp_obj, V_i, S, c=1):
    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    not_goal = [V_i[i] for i, s in mdp_obj.items() if not s['goal']]
    n_states = len(G) + len(not_goal)

    C = np.arange(c + 1) + C_max
    V = np.full((n_states, C_max + c + 1), 0.0)
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
    for C in reversed(range(C_max + 1)):
        for s in S:
            actions_results = np.array(
                [mdp.Q(s, C, a, u, V, V_i, mdp_obj, c) for a in A])
            i_max = np.argmax(actions_results)
            pi[V_i[s], C] = A[i_max]
            V[V_i[s], C] = actions_results[i_max]

    return V, pi


def finite_gubs(S, A, C_max, V_i, H, mdp_obj, k_g, u, c=1):
    n_states = len(S)
    n_actions = len(A)
    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    not_goal = [V_i[i] for i, s in mdp_obj.items() if not s['goal']]
    Q = np.zeros((n_states, C_max + c + 1, n_actions, H + 1))
    pi = np.full((n_states, C_max + c + 1, H + 1), None)
    # for i_a, _ in enumerate(A):
    #    Q[not_goal, :, i_a, H] = u(np.arange(C_max + c + 1)) - 1
    Q[G] = k_g

    for h in reversed(range(H)):
        for C in reversed(range(C_max + 1)):
            for s in S:
                i_s = V_i[s]
                for i_a, a in enumerate(A):
                    reachable = mdp.find_reachable(s, a, mdp_obj)
                    c_ = 0 if mdp_obj[s]['goal'] else c
                    s_a_cost = u(C + c_) - u(C)
                    Q[i_s, C, i_a, h] = s_a_cost + sum([
                        np.max(Q[V_i[s_['name']], C + c_, :, h + 1]) * s_['A'][a] for s_ in reachable])
                pi[i_s, C, h] = A[np.argmax(Q[i_s, C, :, h])]
    return Q, pi


def u(lamb, c): return np.exp(lamb * c)


def risk_sensitive(lamb, mdp_obj, V_i, S, c=1, epsilon=1e-3, n_iter=None):
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

            # TODO -> Ajeitar esse cálculo abaixo, ta dando 0 sempre em alguns casos.
            #           - Talvez não esteja pegando as ações que não são maxprob corretamente
            # record maxprob obtained by actions that are in A_not_max_prob
            P_not_max_prob[V_i[s]] = P[V_i[s]] if len(not_max_prob_actions_results) == 0 else np.max(
                not_max_prob_actions_results)

            actions_results = np.array([
                np.sum([
                    u(c) * V[V_i[s_['name']]] * s_['A'][a] for s_ in mdp.find_reachable(s, a, mdp_obj)
                ]) for a in A_max_prob
            ])

            i_a = np.argmax(actions_results)
            if s == '6':
                print('EITA')
                print(' ', actions_results_p)
                print(' ', A_max_prob, i_a, A_max_prob[i_a])
                print(' ', actions_results, actions_results[i_a])
            if s == '43':
                print('EITA2')
                print(' ', actions_results_p)
            if s == '42':
                print('EITA3')
                print(' ', actions_results_p)
            V_[V_i[s]] = actions_results[i_a]
            pi[V_i[s]] = A_max_prob[i_a]

        v_norm = norm(V_ - V, np.inf)
        p_norm = norm(P_ - P, np.inf)

        P_diff = P_ - P_not_max_prob
        arg_min_p_diff = np.argmin(P_diff)
        min_p_diff = P_diff[arg_min_p_diff]

        if n_iter and i == n_iter:
            break
        #print('delta1:', v_norm, p_norm, v_norm + p_norm)
        #print('prob:', P, P_)
        #print('delta2:', P_diff, min_p_diff)
        if v_norm + p_norm < epsilon and min_p_diff >= 0:
            break
        V = V_
        P = P_
        i += 1

    print(f'{i} iterations')
    return V, P, pi


def get_X(V, V_i, lamb, S, A, mdp_obj, c=1):

    list_X = [
        (
            (s, a),
            (V[V_i[s]] - np.sum(
                np.fromiter(
                    (s_['A'][a] * u(lamb, c) * V[V_i[s_['name']]]
                     for s_ in mdp.find_reachable(s, a, mdp_obj)), dtype=float))
             )
        )
        for (s, a) in itertools.product(S, A)
    ]

    X = np.array(list_X)

    return X[X.T[1] < 0]


def get_cmax(V, V_i, P, S, A, lamb, k_g, mdp_obj, c=1):
    X = get_X(V, V_i, lamb, S, A, mdp_obj)
    # print("X:", X)
    W = np.zeros(len(X))

    for i, ((s, a), x) in enumerate(X):
        # print('oi, ', s, a, x, P[V_i[s]], np.fromiter((s_['A'][a] * P[V_i[s_['name']]]
        #                                               for s_ in mdp.find_reachable(s, a, mdp_obj)), dtype=float), np.sum(np.fromiter((s_['A'][a] * P[V_i[s_['name']]]
        #                                                                                                                               for s_ in mdp.find_reachable(s, a, mdp_obj)), dtype=float)))
        denominator = k_g * (np.sum(np.fromiter((s_['A'][a] * P[V_i[s_['name']]]
                                                 for s_ in mdp.find_reachable(s, a, mdp_obj)), dtype=float)) - P[V_i[s]])
        if denominator == 0:
            W[i] = -np.inf
        else:
            # print('calc: ', -(1 / lamb), denominator, x /
            #      denominator, np.log(x / denominator))
            W[i] = -(1 / lamb) * np.log(
                x / denominator
            )

    print("W[s]:", [((s, a), W[i]) for i, ((s, a), x) in enumerate(X)])
    try:
        C_max = np.max(W[np.invert(np.isnan(W))])
    except:
        return 0
    if C_max < 0 or C_max == np.inf:
        return 0

    return int(np.ceil(C_max))


def exact_gubs(V_risk, P_risk, pi_risk, C_max, lamb, k_g, mdp_obj, V_i, S, A, c=1):
    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    n_states = len(S)
    n_actions = len(A)

    V = np.zeros((n_states, C_max + 1))
    V_risk_C = np.zeros((n_states, C_max + 2))
    P = np.zeros((n_states, C_max + 2))
    pi = np.full((n_states, C_max + 2), None)

    #print(n_states, C_max)
    #print('V_risk:', V_risk)
    # print(V_risk_C.shape)
    # print(G)
    #V_risk_C[G, :] = k_g
    V_risk_C[G, :] = V_risk[G]
    P[G, :] = 1
    # print('V_risk_C antes antes:', V_risk_C)
    V_risk_C[:, C_max + 1] = V_risk.T
    # print('V_risk_C antes:', V_risk_C)
    P[:, C_max + 1] = P_risk.T
    pi[:, C_max + 1] = pi_risk.T

    n_updates = 0
    # print(V_risk)
    for C in reversed(range(C_max + 1)):
        #print(f'C = {C}')
        Q = np.zeros(n_actions)
        P_a = np.zeros(n_actions)
        for s in S:
            i_s = V_i[s]
            n_updates += 1
            for i_a, a in enumerate(A):
                c__ = 0 if mdp_obj[s]['goal'] else c
                c_ = C + c__
                reachable = mdp.find_reachable(s, a, mdp_obj)

                # Get value
                gen_q = [s_['A'][a] * V_risk_C[V_i[s_['name']], c_]
                         for s_ in reachable]
                #print(' gen_q:', gen_q, lamb, c__)
                Q[i_a] = u(lamb, c__) * \
                    np.sum(np.fromiter(gen_q, dtype=np.float))

                # Get probability
                gen_p = (s_['A'][a] * P[V_i[s_['name']], c_]
                         for s_ in reachable)
                P_a[i_a] = np.sum(
                    np.fromiter(gen_p, dtype=np.float)
                )
                # if s == '3':
                #print(C, s, P_a[i_a], Q[i_a], c_)

                #print(s, C)
            i_a_opt = np.argmax(u(lamb, C) * Q + k_g * P_a)
            a_opt = A[i_a_opt]
            #print('Q:', Q)
            #print('argmax:', u(lamb, C) * Q + k_g * P_a, i_a_opt, a_opt, A)
            pi[i_s, C] = a_opt

            P[i_s, C] = P_a[i_a_opt]
            V_risk_C[i_s, C] = Q[i_a_opt]
            #print('P:', P[i_s, C])
            #print('V_risk:', V_risk)
            #print('V_risk_C:', V_risk_C)
            ##print("Q: ", Q)
            # print(
            #    f'{i_s}, {C}, {V_risk_C[i_s, C]}, {V_risk_C[i_s]}, {P[i_s, C]}')
            V[i_s, C] = V_risk_C[i_s, C] + k_g * P[i_s, C]
            # print('  result:', V_risk_C[i_s, C], k_g * P[i_s, C],
            #      V_risk_C[i_s, C] + k_g * P[i_s, C], V[i_s, C])

    print("Updates:", n_updates)
    return V, P, pi
