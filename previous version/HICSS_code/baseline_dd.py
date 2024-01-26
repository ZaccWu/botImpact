import numpy as np
import pandas as pd
import torch
import random
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, GATConv
from geomloss import SamplesLoss
import warnings
warnings.filterwarnings("ignore")

seed = 101

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class impactDetect(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=1):
        super(impactDetect, self).__init__()
        self.convZ1 = GCNConv(in_dim, h_dim)
        self.convZ2 = GCNConv(h_dim, h_dim)
        self.yNet1 = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())
        self.yNet0 = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())

    def forward(self, x, edge_index, treat_idx, control_idx):
        # generate node embedding
        xZ1 = F.relu(self.convZ1(x, edge_index))
        xZ1 = F.dropout(xZ1, p=0.5, training=self.training)
        xZ2 = self.convZ2(xZ1, edge_index)    # xZ2: (num_nodes, h_dim)
        # predict outcome
        y1, yc0 = self.yNet1(xZ2[treat_idx]), self.yNet0(xZ2[treat_idx])       # yc0：如果有bot的node周围没bot会怎样
        y0, yc1 = self.yNet0(xZ2[control_idx]), self.yNet1(xZ2[control_idx])   # yc1: 如果没bot的node周围有bot会怎么样
        return y1.squeeze(-1), yc0.squeeze(-1), y0.squeeze(-1), yc1.squeeze(-1), xZ2

def evaluate_metric(pred_0, pred_1, pred_c1, pred_c0):
    tau_pred = torch.cat([pred_c1, pred_1], dim=0) - torch.cat([pred_0, pred_c0], dim=0)
    print("pred treat:", torch.mean(pred_1), torch.mean(pred_c1))
    print("pred control:", torch.mean(pred_0), torch.mean(pred_c0))
    tau_true = torch.ones(tau_pred.shape) * -1
    ePEHE = torch.sqrt(torch.mean(torch.square(tau_pred-tau_true)))
    eATE = torch.abs(torch.mean(tau_pred) - torch.mean(tau_true))
    return eATE, ePEHE

def load_data(dt):
    # load train data
    edge_index_train = torch.LongTensor(np.load('Dataset/synthetic/'+type+'/'+dt+'_edge.npy'))    # (num_edge, 2)
    bot_label_train = np.load('Dataset/synthetic/'+type+'/'+dt+'_bot_label.npy')
    T_train = np.load('Dataset/synthetic/'+type+'/'+dt+'_T_label.npy')
    outcome_train = np.load('Dataset/synthetic/'+type+'/'+dt+'_y.npy')
    prop_label = np.load('Dataset/synthetic/'+type+'/'+dt+'_prop_label.npy')
    N = len(outcome_train)  # num of nodes
    x = degree(edge_index_train[:, 0])  # user node degree as feature
    target_var = torch.tensor(
        np.concatenate([bot_label_train[:, np.newaxis], outcome_train[:, np.newaxis], T_train[:, np.newaxis]], axis=-1))  # (num_nodes, 3)
    botData = Data(x=x.unsqueeze(-1), edge_index=edge_index_train.t().contiguous(), y=target_var)
    return botData, N, prop_label

def main():
    botData_train, N_train, prop_label_train = load_data('train')
    botData_test, N_test, prop_label_test = load_data('train')

    model3 = impactDetect(in_dim=1, h_dim=16, out_dim=1)
    optimizer = torch.optim.Adam([{'params': model3.parameters(), 'lr': 0.001}])

    treat_idx_train, control_idx_train = torch.where(botData_train.y[:, 2]==-1)[0], torch.where(botData_train.y[:, 2]==1)[0]
    treat_idx_test, control_idx_test = torch.where(botData_test.y[:, 2]==-1)[0], torch.where(botData_test.y[:, 2]==1)[0]

    # train
    for epoch in range(300):
        model3.train()
        optimizer.zero_grad()

        # assess bot
        # out_y1, out_yc0: (num_treat_train), out_y0, out_yc1: (num_control_train), *_prob: (num_nodes)
        out_y1, out_yc0, out_y0, out_yc1, z = model3(botData_train.x, botData_train.edge_index, treat_idx_train, control_idx_train)
        out_y = torch.cat([out_y1, out_y0], dim=-1)
        target_y = torch.cat([botData_train.y[:, 1][treat_idx_train], botData_train.y[:, 1][control_idx_train]])
        loss_y = F.mse_loss(out_y.float(), target_y.float())
        samples_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.005, backend="tensorized")
        imbalance_loss = samples_loss(z[treat_idx_train], z[control_idx_train])

        print("{:.4f} {:.4f}".format(loss_y.item(), imbalance_loss.item()))
        loss = loss_y*par['ly'] + imbalance_loss*par['limb']
        loss.backward()
        optimizer.step()

        if epoch%10 == 0:
            model3.eval()
            out_y1, out_yc0, out_y0, out_yc1, z = model3(botData_test.x, botData_test.edge_index, treat_idx_test, control_idx_test)
            # treatment effect prediction result
            eATE_test, ePEHE_test = evaluate_metric(out_y0, out_y1, out_yc1, out_yc0)
            print("Epoch: " + str(epoch))
            print('eATE: {:.4f}'.format(eATE_test.detach().numpy()),
                  'ePEHE: {:.4f}'.format(ePEHE_test.detach().numpy()))



if __name__ == "__main__":
    type = 'highcc'
    gpu = 0
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    par = {'ly': 1,
           'limb': 1}
    print(par)
    main()
