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
        V[V_i[s_['name']], C + c] * s_['A'][a] for s_ in reachable])
