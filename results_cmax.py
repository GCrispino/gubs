import numpy as np
import matplotlib.pyplot as plt
from utils import read_json
from gubs import risk_sensitive, get_cmax, get_X
import mdp

nx = 5
ny = 8
p = 40
file_input = f'./river{nx}-{ny}-{p}-2.json'
#file_input = './env1-reduced.json'
mdp_obj = read_json(file_input)
S = sorted(mdp_obj.keys(), key=int)
A = mdp.get_actions(mdp_obj)
V_i = {S[i]: i for i in range(len(S))}
lamb = -0.1
# lambs = np.arange(-0.4, 0.0, 0.05)
lambs = [lamb]
k_g = 1
#k_gs = np.arange(0, 11)
k_gs = [k_g]

Cs_lamb = []
for lamb in lambs:
    print(lamb)
    V, P, pi = risk_sensitive(lamb, mdp_obj, V_i, S,
                              epsilon=1e-18, n_iter=(nx * ny + 10) * 4)
    print(pi.reshape(ny, nx))
    print(V.reshape(ny, nx))
    print(P.reshape(ny, nx))
    X = get_X(V, V_i, lamb, S, A, mdp_obj)
    C_max = get_cmax(V, V_i, P, S, A, lamb, k_g, mdp_obj)
    Cs_lamb.append(C_max)
# print()
# print(X)
# print()
print('C_max:', C_max)
#Cs_kgs = []
# for k_g in k_gs:
#    print(k_g)
#    V, P, pi = risk_sensitive(lamb, mdp_obj, V_i, S, epsilon=1e-15)
#    print(pi.reshape(ny, nx))
#    X = get_X(V, V_i, lamb, S, A, mdp_obj)
#    C_max = get_cmax(V, V_i, P, S, A, lamb, k_g, mdp_obj)
#    print('C_max:', C_max)
#    Cs_kgs.append(C_max)
# print()
#print(V.reshape(ny, nx))
# print()
#print(pi.reshape(ny, nx))
# print()
#print(P.reshape(ny, nx))
# print()
#print(Cs_lamb, Cs_kgs)
#fig, ax = plt.subplots(2, 1)
#ax[0].plot(lambs, Cs_lamb)
#ax[1].plot(k_gs, Cs_kgs)
# fig.show()
# plt.show()
