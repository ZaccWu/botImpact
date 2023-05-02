import numpy as np
import pandas as pd
import torch
import random
from torch_geometric.data import Data
from torch_geometric.utils import degree

seed = 101
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


edge_index = np.load('Dataset/synthetic/edge.npy')
bot_label = np.load('Dataset/synthetic/bot_label.npy')
T = np.load('Dataset/synthetic/T_label.npy')
outcome = np.load('Dataset/synthetic/y.npy')

N = len(outcome)    # num of nodes
x = degree(torch.LongTensor(edge_index[:,0]))   # user node degree as feature
node_idx = [i for i in range(N)]
random.shuffle(node_idx)

train_mask = torch.zeros(N, dtype=torch.bool)
train_mask[node_idx[:int(N*0.8)]] = 1  # 前80%训练
test_mask = ~train_mask

data = Data(x=x, edge_index=edge_index.T, y=outcome, train_mask=train_mask, test_mask=test_mask)
