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
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        #return F.log_softmax(x, dim=1)
        return x

def anomaly_loss(target, pred_score, m=1):
    # target: 0-1, pred_score: float
    Rs = torch.mean(pred_score)
    delta = torch.std(pred_score)
    dev_score = (pred_score - Rs)/delta
    cont_score = torch.max(torch.zeros(pred_score.shape), m-dev_score)
    loss = dev_score[(1-target).nonzero()].sum()+cont_score[target.nonzero()].sum()
    return loss

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

model = GCN(1, 16, 1) # feature=1, hidden=16, out=1
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



def train():
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(botData.x, botData.edge_index)
        target = botData.y[:,0][botData.train_mask]
        loss = anomaly_loss(target, out[botData.train_mask])
        loss.backward()
        optimizer.step()
        print(loss.item())

        # model evaluation
        threshold = torch.quantile(out, 0.96, dim=None, keepdim=False, interpolation='higher')
        pred = out.clone()
        pred[torch.where(out<threshold)]=0
        pred[torch.where(out>=threshold)]=1
        pred_label = pred[botData.train_mask]
        report = classification_report(target, pred_label.detach().numpy(), target_names=['class0','class1'], output_dict=True)
        print(report['class1']['precision'], report['class1']['recall'], report['macro avg']['f1-score'])

@torch.no_grad()
def test():
    model.eval()
    out = model(botData.x, botData.edge_index)
    threshold = torch.quantile(out, 0.96, dim=None, keepdim=False, interpolation='higher')
    pred = out.clone()
    pred[torch.where(out < threshold)] = 0
    pred[torch.where(out >= threshold)] = 1
    pred_label = pred[botData.test_mask]
    target = botData.y[:,0][botData.test_mask]
    report = classification_report(target, pred_label)
    print(report)

train()
test()
