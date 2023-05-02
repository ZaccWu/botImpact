import numpy as np
import pandas as pd
import torch
import random
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report

seed = 101
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

edge_index = torch.LongTensor(np.load('Dataset/synthetic/edge.npy'))
bot_label = np.load('Dataset/synthetic/bot_label.npy')
T = np.load('Dataset/synthetic/T_label.npy')
outcome = np.load('Dataset/synthetic/y.npy')

N = len(outcome)    # num of nodes
x = degree(edge_index[:,0])   # user node degree as feature
node_idx = [i for i in range(N)]
random.shuffle(node_idx)

train_mask = torch.zeros(N, dtype=torch.bool)
train_mask[node_idx[:int(N*0.8)]] = 1  # 前80%训练
test_mask = ~train_mask
target_var = torch.LongTensor(np.concatenate([bot_label[:,np.newaxis], outcome[:,np.newaxis], T[:,np.newaxis]], axis=-1))
botData = Data(x=x.unsqueeze(-1), edge_index=edge_index.t().contiguous(), y=target_var, train_mask=train_mask, test_mask=test_mask)

model = GCN(1, 16, 2) # feature=1, hidden=16, out=2
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(botData.x, botData.edge_index)[botData.train_mask]
        loss = criterion(out, botData.y[:,0][botData.train_mask])
        loss.backward()
        optimizer.step()
        print(loss.item())

@torch.no_grad()
def test():
    model.eval()
    _, pred = model(botData.x, botData.edge_index).max(dim=-1)
    pred_label = pred[botData.test_mask]
    target = botData.y[:,0][botData.test_mask]
    report = classification_report(target, pred_label)
    print(report)

train()
test()
