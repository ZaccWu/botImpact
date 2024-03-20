import numpy as np
import pandas as pd
import random
import os
import torch
from torch_geometric.utils import degree
from torch_geometric.utils import to_networkx
import networkx as nx

seed = 101
random.seed(seed)
np.random.seed(seed)

sample_user = 3000
sample_bot = 100
betaZ = 1
betaT = 2
betaB = 1
EPSILON = 0.1
N = sample_user + sample_bot

def generate_network(z, bl, type):
    '''
    input: latent trait, bot label
    '''
    edge_idx = []
    if type == 'random':
        for i in range(N):
            for j in range(i + 1, N):
                if bl[i] or bl[j] == 1:
                    p = 0.01
                else:
                    p = 0.02 if z[i] == z[j] else 0.005
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])

    elif type == 'randomu':
        for i in range(sample_user):
            for j in range(i + 1, sample_user):
                p = 0.02 if z[i] == z[j] else 0.005
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])
        for u in range(sample_user):
            for b in range(sample_user, N):
                p = 0.01
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([u, b])
                    edge_idx.append([b, u])

    elif type == 'highdu':
        for i in range(sample_user):
            for j in range(i + 1, sample_user):
                p = 0.02 if z[i] == z[j] else 0.005
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])
        ndeg = degree(torch.LongTensor(edge_idx).T[0,:])
        max_deg = torch.max(ndeg).item()
        for u in range(sample_user):
            for b in range(sample_user, N):
                p = 0.02*ndeg[u]/max_deg
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([u, b])
                    edge_idx.append([b, u])

    elif type == 'lowdu':
        for i in range(sample_user):
            for j in range(i + 1, sample_user):
                p = 0.02 if z[i] == z[j] else 0.005
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])
        ndeg = degree(torch.LongTensor(edge_idx).T[0,:])
        for u in range(sample_user):
            for b in range(sample_user, N):
                p = 0.05*1/ndeg[u]
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([u, b])
                    edge_idx.append([b, u])

    elif type == 'highbc':
        edge_idx_nx = []
        for i in range(sample_user):
            for j in range(i + 1, sample_user):
                p = 0.02 if z[i] == z[j] else 0.005
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])
                    edge_idx_nx.append((i, j))
                    edge_idx_nx.append((j, i))
        G = nx.Graph(edge_idx_nx)
        betweenness_centrality = nx.betweenness_centrality(G)
        max_bc = max(betweenness_centrality.values())
        for u in range(sample_user):
            for b in range(sample_user, N):
                p = 0.02*betweenness_centrality[u]/max_bc
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([u, b])
                    edge_idx.append([b, u])

    elif type == 'highcc':
        edge_idx_nx = []
        for i in range(sample_user):
            for j in range(i + 1, sample_user):
                p = 0.02 if z[i] == z[j] else 0.005
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])
                    edge_idx_nx.append((i, j))
                    edge_idx_nx.append((j, i))
        G = nx.Graph(edge_idx_nx)
        betweenness_centrality = nx.closeness_centrality(G)
        max_bc = max(betweenness_centrality.values())
        for u in range(sample_user):
            for b in range(sample_user, N):
                p = 0.02*betweenness_centrality[u]/max_bc
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([u, b])
                    edge_idx.append([b, u])

    elif type == 'semi-homo-1':
        for i in range(N):
            for j in range(i + 1, N):
                if bl[i] or bl[j] == 1:
                    p = 0.01
                else:
                    p = 0.018 if z[i] == z[j] else 0.006
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])

    elif type == 'semi-homo-2':
        for i in range(N):
            for j in range(i + 1, N):
                if bl[i] or bl[j] == 1:
                    p = 0.01
                else:
                    p = 0.016 if z[i] == z[j] else 0.007
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])

    elif type == 'semi-homo-3':
        for i in range(N):
            for j in range(i + 1, N):
                if bl[i] or bl[j] == 1:
                    p = 0.01
                else:
                    p = 0.014 if z[i] == z[j] else 0.008
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])

    elif type == 'semi-homo-4':
        for i in range(N):
            for j in range(i + 1, N):
                if bl[i] or bl[j] == 1:
                    p = 0.01
                else:
                    p = 0.012 if z[i] == z[j] else 0.009
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
            T[i] = 1    # control
        elif bl[i]==0 and di==0 and bi==1:
            T[i] = -1   # treat(have bot)
        Di[i], Bi[i] = di, bi
        y[i] = betaZ * Z[i] + betaT * di + betaB * bi + eps[i]
    return y, Di, Bi, T


# type = 'random'   # bots randomly connect
# type = 'randomu'  # bots randomly connect users
# type = 'highdu'   # bots connect high-degree users
# type = 'lowdu'    # bots connect low-degree users
# type = 'highbc'   # bots connect high-betweeness-centrality users
# type = 'highcc'   # bots connect high-closeness-centrality users

# other robustness check
type = 'semi-homo-4'

for dt in range(100):
    bot_label = np.array([0]*sample_user+[1]*sample_bot)
    Zu = np.random.choice([0,1], sample_user)
    Zb = np.ones(sample_bot)
    Z = np.concatenate([Zu, Zb])
    propagator = np.zeros(sample_user+sample_bot)

    eps = np.random.normal(0, EPSILON, size=N)
    propagator_id = random.sample(set(np.nonzero(Zu)[0]),sample_bot)  # 推广产品的用户id (假设和bot数量相同)
    propagator[propagator_id] = 1

    edge_index = generate_network(Z, bot_label, type)
    outcome, Di, Bi, T = cal_outcome(Z, edge_index, propagator_id, bot_label, eps)
    out_data = pd.DataFrame({'bot_label': bot_label,
                             'propagator': propagator,
                             'Di': Di,
                             'Bi': Bi,
                             'treated': T,
                             'purchase': outcome})

    if not os.path.isdir('Dataset/synthetic/'+type+'/'):
        os.makedirs('Dataset/synthetic/'+type+'/')
    np.save('Dataset/synthetic/'+type+'/'+str(dt)+'_edge.npy', edge_index)
    np.save('Dataset/synthetic/'+type+'/'+str(dt)+'_bot_label.npy', bot_label)
    np.save('Dataset/synthetic/'+type+'/'+str(dt)+'_T_label.npy', T)
    np.save('Dataset/synthetic/'+type+'/'+str(dt)+'_y.npy', outcome)
    np.save('Dataset/synthetic/'+type+'/'+str(dt) + '_prop_label.npy', propagator)
    out_data.to_csv('Dataset/synthetic/'+type+'/'+str(dt)+'_bot.csv', index=False)
