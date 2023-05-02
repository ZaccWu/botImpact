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


class GCN_bot(torch.nn.Module):
    def __init__(self, in_dim, h_dim1, h_dim2, out_dim1=1, out_dim2=2):
        super(GCN_bot, self).__init__()
        self.convB1 = GCNConv(in_dim, h_dim1)
        self.convB2 = GCNConv(h_dim1, out_dim1)

        self.convY1 = GCNConv(in_dim, h_dim2)
        self.convY2 = GCNConv(h_dim2, out_dim2)

    def forward(self, x, edge_index):
        xB1 = F.relu(self.convB1(x, edge_index))
        xB1 = F.dropout(xB1, p=0.5, training=self.training)
        xB2 = self.convB2(xB1, edge_index)

        xY1 = F.relu(self.convY1(x, edge_index))
        xY1 = F.dropout(xY1, p=0.5, training=self.training)
        xY2 = self.convB2(xY1, edge_index)

        return xB2, F.log_softmax(xY2, dim=1)

def anomaly_loss(target, pred_score, m=1):
    # target: 0-1, pred_score: float
    Rs = torch.mean(pred_score)
    delta = torch.std(pred_score)
    dev_score = (pred_score - Rs)/delta
    cont_score = torch.max(torch.zeros(pred_score.shape), m-dev_score)
    loss = dev_score[(1-target).nonzero()].sum()+cont_score[target.nonzero()].sum()
    return loss

def outcome_loss(target, output):
    criterion = torch.nn.NLLLoss()
    loss = criterion(output, target)
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

model1 = GCN_bot(1, 16, 1) # feature=1, hidden=16, out=1
optimizer = torch.optim.Adam(model1.parameters(), lr=0.01)

def train():
    model1.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out_b, out_y = model1(botData.x, botData.edge_index)
        target_b, target_y = botData.y[:,0][botData.train_mask], botData.y[:,1][botData.train_mask]
        loss_b = anomaly_loss(target_b, out_b[botData.train_mask])
        loss_y = outcome_loss(target_y, out_y[botData.train_mask])

        loss = loss_b + loss_y
        loss.backward()
        optimizer.step()
        print(loss.item())

        # model evaluation
        threshold = torch.quantile(out_b, 0.96, dim=None, keepdim=False, interpolation='higher')
        pred_b = out_b.clone()
        pred_b[torch.where(out_b<threshold)]=0
        pred_b[torch.where(out_b>=threshold)]=1
        pred_label_b = pred_b[botData.train_mask]
        report = classification_report(target_b, pred_label_b.detach().numpy(), target_names=['class0','class1'], output_dict=True)
        print(report['class1']['precision'], report['class1']['recall'], report['macro avg']['f1-score'])

@torch.no_grad()
def test():
    model1.eval()
    out_b, out_y = model1(botData.x, botData.edge_index)

    # bot detection
    threshold = torch.quantile(out_b, 0.96, dim=None, keepdim=False, interpolation='higher')
    pred_b = out_b.clone()
    pred_b[torch.where(out_b < threshold)] = 0
    pred_b[torch.where(out_b >= threshold)] = 1
    pred_label_b = pred_b[botData.test_mask]
    target_b = botData.y[:,0][botData.test_mask]
    report = classification_report(target_b, pred_label_b)
    print(report)

    # purchase prediction
    

train()
test()
