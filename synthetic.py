import numpy as np
import pandas as pd
import random

seed = 101
random.seed(seed)
np.random.seed(seed)

sample_user = 3000
sample_bot = 100
betaZ = 0.5
betaT = 0.3
betaB = 0.2
EPSILON = 0.1


N = sample_user + sample_bot

def generate_network(z, bl):
    # input: latent trait, bot label
    edge_idx = []
    for i in range(N):
        for j in range(i + 1, N):
            if bl[i] or bl[j] == 1:
                p = 0.005
            else:
                p = 0.05 if z[i] == z[j] else 0.001
            friend = np.random.binomial(1, p)
            if friend == 1:
                edge_idx.append([i, j])
                edge_idx.append([j, i])
    return np.array(edge_idx)


def cal_outcome(Z, edge_idx, propagator_id, bl, eps):
    # edge_idx: (num_edge, 2)
    friend_dict = {}
    for i in range(len(edge_idx)):
        u, v = edge_idx[i][0], edge_idx[i][1]
        friend_dict.setdefault(u, []).append(v)

    y = np.zeros(N)
    Di, Bi = np.zeros(N), np.zeros(N)# 0-1 vector (sample_user + sample_bot)
    T = np.zeros(N) # treat-control label
    for i in range(N):
        if i not in friend_dict:
            y[i] = betaZ * Z[i] + eps[i]
            continue
        friend = friend_dict[i] # i's friend id
        prop_u = set(friend)&set(propagator_id)
        prop_b = np.nonzero(bl[friend])[0]
        di = 1 if len(prop_u)>0 else 0
        bi = 1 if len(prop_b)>0 else 0
        if bl[i]==0 and di==1 and bi==0:
            T[i] = 1    # treated
        elif bl[i]==0 and di==0 and bi==1:
            T[i] = -1   # control

        Di[i], Bi[i] = di, bi
        y[i] = betaZ * Z[i] + betaT * di + betaB * bi + eps[i]
    return y, Di, Bi, T

for dt in ['train', 'test']:
    bot_label = np.array([0]*sample_user+[1]*sample_bot)
    Zu = np.random.choice([0,1], sample_user)
    Zb = np.ones(sample_bot)
    Z = np.concatenate([Zu, Zb])
    propagator = np.zeros(sample_user+sample_bot)

    eps = np.random.normal(0, EPSILON, size=N)
    propagator_id = random.sample(set(np.nonzero(Zu)[0]),sample_bot)  # 推广产品的用户id (假设和bot数量相同)

    edge_index = generate_network(Z, bot_label)
    outcome, Di, Bi, T = cal_outcome(Z, edge_index, propagator_id, bot_label, eps)
    propagator[propagator_id] = 1

    out_data = pd.DataFrame({'bot_label': bot_label,
                             'propagator': propagator,
                             'Di': Di,
                             'Bi': Bi,
                             'treated': T,
                             'purchase': outcome})

    np.save('Dataset/synthetic/'+dt+'_edge.npy', edge_index)
    np.save('Dataset/synthetic/'+dt+'_bot_label.npy', bot_label)
    np.save('Dataset/synthetic/'+dt+'_T_label.npy', T)
    np.save('Dataset/synthetic/'+dt+'_y.npy', outcome)
    np.save('Dataset/synthetic/'+dt + '_prop_label.npy', propagator)
    out_data.to_csv('Dataset/synthetic/'+dt+'_bot1.csv', index=False)