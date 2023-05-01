import numpy as np
import pandas as pd
import random

seed = 101
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

sample_user = 2000
sample_bot = 100
betaZ = 1


N = sample_user + sample_bot

def generate_network(z, bl):
    # input: latent trait, bot label
    edge_idx = []
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            p = 0.2 if bl[i] == bl[j] else 0.02
            friend = np.random.binomial(1, p)
            if friend == 1:
                edge_idx.append([i,j])
    return edge_idx

def cal_influence(edge_idx, s0, bl):
    # edge_idx: (num_edge, 2)
    friend_dict = {}
    for i in range(N):
        u, v = edge_idx[i,:][0], edge_idx[i,:][1]
        friend_dict.setdefault(u, []).append(v)

    inf_b, inf_u = np.zeros(N), np.zeros(N)
    for i in range(N):
        friend = friend_dict[i]
        friend_stance = s0[friend]
        friend_b = np.nonzero(bl[friend])
        friend_u = np.nonzero(1-bl[friend])

        inf_b[i] = np.sum(friend_stance[friend_b])/len(friend_b)
        inf_u[i] = np.sum(friend_stance[friend_u])/len(friend_u)
    return inf_b, inf_u


bot_label = np.array([0]*sample_user+[1]*sample_bot)
Z = np.random.choice([-1,0,1], sample_user+sample_bot)
s0 = betaZ * Z



match_score = []
for zi in Z:
    match_score
