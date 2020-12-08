import mdp
import gubs
from utils import read_json
import matplotlib.pyplot as plt
import numpy as np

nx = 7
ny = 10
p = 40
test = False
file_input = f'./env/river{nx}-{ny}-{p}-0{"-test" if test else ""}.json'
# file_input = './env/navigation5-3.json'
# file_input = './env/teste.json'
mdp_obj = read_json(file_input)
S = sorted(mdp_obj.keys(), key=int)
A = mdp.get_actions(mdp_obj)
V_i = {S[i]: i for i in range(len(S))}
lamb = -0.2
k_g = 0.5

# n_iter = (nx * ny + 10) * 4
n_iter = None
V, P, pi = gubs.risk_sensitive(lamb, mdp_obj, V_i, S,
                               epsilon=1e-10, n_iter=n_iter)
# np.set_printoptions(16)
print('V_risk: ', V)
print('pi_risk: ', pi)
print('P_risk: ', P)
print()
print(pi[[0, 1, 2, 3, 4]])
# print(V[[140, 0, 1, 2, 3, 4, 5]])
print(V[[10, 0, 1, 2, 3, 4, 5, 6]])
# exit()
X = gubs.get_X(V, V_i, lamb, S, A, mdp_obj)
C_max = gubs.get_cmax(V, V_i, P, S, A, lamb, k_g, mdp_obj)
V_gubs, P_gubs, pi_gubs = gubs.exact_gubs(
    V, P, pi, C_max, lamb, k_g, mdp_obj, V_i, S, A)

print('X:', X)
print('C_max:', C_max)
print('pi_gubs:', pi_gubs.shape, pi_gubs)
print('V_gubs:', V_gubs.shape, V_gubs)
try:
    print(V.reshape(ny, nx))
except ValueError:
    pass
print(V_gubs, V_gubs.shape)
print(P_gubs, P_gubs.shape)
print()
# print(V_gubs[3, 0], V_gubs[4, 1], V_gubs[0, 1])
# print(pi_gubs[3, 0], pi_gubs[0, 1])
# print(P_gubs[3, 0], P_gubs[0, 1])
# print(V_gubs[8, 0], V_gubs[4, 1], V_gubs[5, 2])
# print(V_gubs[8, 0] - k_g * P_gubs[8, 0])
# print(V_gubs[9, 1] - k_g * P_gubs[9, 1])
# print(V_gubs[10, 2] - k_g * P_gubs[10, 2])
# print(V_gubs[5, 2] - k_g * P_gubs[5, 2], V_gubs[1, 2] - k_g * P_gubs[1, 2])
# print(V_gubs[0, 2] - k_g * P_gubs[0, 2])
# print(V_gubs[9, 1] - k_g * P_gubs[9, 1])
# print(V_gubs[9, 5] - k_g * P_gubs[9, 5])
# print(V_gubs[4, 1] - k_g * P_gubs[4, 1])
# print(V_gubs[6, 3] - k_g * P_gubs[6, 3])
# print()
# print(P_gubs[8, 0], P_gubs[4, 1], P_gubs[5, 2])
# print(pi_gubs[8, 0], pi_gubs[4, 1], pi_gubs[5, 2])
# print(pi_gubs[7, 4], pi_gubs[7, 5])
# print()
# s_ = [(12, 1), (12, 2), (13, 2), (12, 3), (0, 4), (4, 4), (1, 4), (13, 3), (2, 4), (8, 5), (4, 5), (12, 5), (9, 4), (12, 4),
#      (5, 5), (0, 5), (13, 4), (1, 5), (6, 5), (9, 6), (1, 6), (6, 6), (4, 6), (13, 5), (10, 5), (14, 3), (7, 4), (2, 5)]
# for x in s_:
#    print((x[0] + 1, x[1]))
#    print(' ', V_gubs[x] - P_gubs[x] * k_g, P_gubs[x], pi_gubs[x])
# print(V_gubs[12, 0], V_gubs[8, 1], V_gubs[4, 2], V_gubs[0, 3])
# print(P_gubs[12, 0], P_gubs[8, 1], P_gubs[4, 2], P_gubs[0, 3])
# print(pi_gubs[12, 0], pi_gubs[8, 1], pi_gubs[4, 2], pi_gubs[0, 3])
# print(V_gubs[30, 0], V_gubs[25, 1], V_gubs[20, 2],
#      V_gubs[15, 3], V_gubs[10, 4], V_gubs[5, 5])
# print(pi_gubs[30, 0], pi_gubs[25, 1], pi_gubs[20, 2],
#      pi_gubs[15, 3], pi_gubs[10, 4], pi_gubs[5, 5], pi_gubs[0, 6])
# print(P_gubs[30, 0], P_gubs[25, 1], P_gubs[20, 2],
#      P_gubs[15, 3], P_gubs[10, 4], P_gubs[5, 5])
# print(P_gubs[30, 0], P_gubs[25, 1], P_gubs[20, 2],
#      P_gubs[15, 3], P_gubs[16, 4], P_gubs[17, 5], P_gubs[18, 6])
#print(V_gubs[25, 1] - k_g * P_gubs[25, 1])
#print(V_gubs[20, 1] - k_g * P_gubs[20, 1])
print(V_gubs[56, 0], V_gubs[49, 1], V_gubs[42, 2],
      V_gubs[35, 3], V_gubs[28, 4], V_gubs[21, 5])
print(pi_gubs[56, 0], pi_gubs[49, 1], pi_gubs[42, 2],
      pi_gubs[35, 3], pi_gubs[28, 4], pi_gubs[21, 5], pi_gubs[0, 6])
print(P_gubs[56, 0], P_gubs[49, 1], P_gubs[42, 2],
      P_gubs[35, 3], P_gubs[28, 4], P_gubs[21, 5])
print(P_gubs[56, 0], P_gubs[49, 1], P_gubs[42, 2],
      P_gubs[35, 3], P_gubs[28, 4], P_gubs[17, 5], P_gubs[18, 6])
# print(V_gubs[490, 0], V_gubs[49, 1], V_gubs[42, 2],
#      V_gubs[35, 3], V_gubs[28, 4], V_gubs[21, 5])
# print(pi_gubs[490, 0], pi_gubs[49, 1], pi_gubs[42, 2],
#      pi_gubs[35, 3], pi_gubs[28, 4], pi_gubs[21, 5], pi_gubs[0, 6])
# print(P_gubs[490, 0], P_gubs[49, 1], P_gubs[42, 2],
#      P_gubs[35, 3], P_gubs[28, 4], P_gubs[21, 5])
# print(V_gubs[240, 0], V_gubs[49, 1], V_gubs[42, 2],
#      V_gubs[35, 3], V_gubs[28, 4], V_gubs[21, 5])
# print(pi_gubs[240, 0], pi_gubs[49, 1], pi_gubs[42, 2],
#      pi_gubs[35, 3], pi_gubs[28, 4], pi_gubs[21, 5], pi_gubs[0, 6])
# print(P_gubs[240, 0], P_gubs[49, 1], P_gubs[42, 2],
#      P_gubs[35, 3], P_gubs[28, 4], P_gubs[21, 5])
# print(V_gubs[47, 7], P_gubs[47, 7], pi_gubs[47, 7])
# print(P_gubs[22, 5])
# print()
# print(V_gubs[10, 4], P_gubs[10, 4], pi_gubs[10, 4])
# print('oie')
# print(P.reshape(ny, nx))
# print(V.reshape(ny, nx))
# print(V_gubs.T[0].reshape(ny, nx))
# print('============================')
# print(P_gubs.T[0].reshape(ny, nx))
# print(V_gubs[:, 0].reshape(ny, nx))
