import numpy as np
import pandas as pd
import torch
import random
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import GATConv

import warnings
warnings.filterwarnings("ignore")

seed = 101
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Generator(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=1):
        super(Generator, self).__init__()
        self.convZ1 = GATConv(in_dim, h_dim)
        self.convZ2 = GATConv(h_dim, h_dim)
        self.yNet1 = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())
        self.yNet0 = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())
    def forward(self, x, edge_index, fake_x, fake_edge_index, treat_idx, control_idx):
        # generate node embedding
        xZ1, xfZ1 = F.relu(self.convZ1(x, edge_index)), F.relu(self.convZ1(fake_x, fake_edge_index))
        xZ1, xfZ1 = F.dropout(xZ1, p=0.5, training=self.training), F.dropout(xfZ1, p=0.5, training=self.training)
        xZ2, xfZ2 = self.convZ2(xZ1, edge_index), self.convZ2(xfZ1, fake_edge_index)    # xZ2/xfZ2: (num_nodes, h_dim)

        # predict outcome
        y1, yc0 = self.yNet1(xZ2[treat_idx]), self.yNet0(xfZ2[treat_idx])       # yc0：如果有bot的node周围没bot会怎样
        y0, yc1 = self.yNet0(xZ2[control_idx]), self.yNet1(xfZ2[control_idx])   # yc1: 如果没bot的node周围有bot会怎么样

        return y1.squeeze(-1), yc0.squeeze(-1), y0.squeeze(-1), yc1.squeeze(-1), xZ2, xfZ2

class Discriminator(torch.nn.Module):
    def __init__(self, h_dim):
        super(Discriminator, self).__init__()
        self.balanceNet = torch.nn.Sequential(torch.nn.Linear(h_dim, 2))
        self.dw = torch.nn.Parameter(torch.Tensor(h_dim, h_dim))    # discriminator weight
        torch.nn.init.xavier_normal_(self.dw)

    def forward(self, xZ2, xfZ2):
        S, Sf = torch.mean(xZ2, dim=0).unsqueeze(-1), torch.mean(xfZ2, dim=0).unsqueeze(-1)  # S/Sf: (h_dim, 1)
        S_, Sf_ = torch.mm(torch.mm(xZ2, self.dw), S), torch.mm(torch.mm(xfZ2, self.dw),Sf)
        D, Df = torch.sigmoid(torch.clamp(S_, min=-5, max=5)).squeeze(-1), \
                torch.sigmoid(torch.clamp(Sf_, min=-5, max=5)).squeeze(-1)   # D/Df: (num_nodes)
        # judge the node is from factual & counterfactual graph
        fprob, fprob_f = self.balanceNet(xZ2), self.balanceNet(xfZ2)
        return fprob.squeeze(-1), fprob_f.squeeze(-1), D, Df

def shuffle_edge(edge_pool, edge_num):
    selected_edge = random.sample(edge_pool, int(edge_num/2))
    selected_edge = torch.LongTensor(list(map(list, selected_edge)))
    selected_edge = torch.cat([selected_edge,torch.flip(selected_edge,[1])],dim=0)
    return selected_edge.T

def match_node(fake_fact_graph, bot_label, prop_label, treat_idx, control_idx):
    # 判断原始图中treat_idx对应节点在fake_fact_graph是否为control，返回有对应control的节点
    friend_dict = {}
    for i in range(len(fake_fact_graph[0])):
        u, v = fake_fact_graph[0][i].item(), fake_fact_graph[1][i].item()
        friend_dict.setdefault(u, []).append(v)
    treat_idx_ok, control_idx_ok = [], []
    for id in treat_idx.tolist():
        if id not in friend_dict:
            continue
        friend = friend_dict[id]  # id's friend
        if bot_label[friend].sum()>0 and prop_label[friend].sum()==0:
            treat_idx_ok.append(id)

    for id in control_idx.tolist():
        if id not in friend_dict:
            continue
        friend = friend_dict[id]  # id's friend
        if prop_label[friend].sum()>0 and bot_label[friend].sum()==0:
            control_idx_ok.append(id)

    return torch.LongTensor(treat_idx_ok), torch.LongTensor(control_idx_ok)

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

    model_g = Generator(in_dim=1, h_dim=16, out_dim=1)
    model_d = Discriminator(h_dim=16)
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=0.001)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.001)

    # for counterfactual edge generation
    edge_pool_train = set(map(tuple, np.array([[i, j] for i in range(N_train) for j in range(i, N_train)])))
    treat_idx_train, control_idx_train = torch.where(botData_train.y[:, 2]==-1)[0], torch.where(botData_train.y[:, 2]==1)[0]

    edge_pool_test = set(map(tuple, np.array([[i, j] for i in range(N_test) for j in range(i, N_test)])))
    treat_idx_test, control_idx_test = torch.where(botData_test.y[:, 2]==-1)[0], torch.where(botData_test.y[:, 2]==1)[0]

    # train
    for epoch in range(300):
        model_g.train()
        model_d.train()
        fake_fact_graph = shuffle_edge(edge_pool_train, edge_num=len(botData_train.edge_index[0])) # for effect estimation
        botData_fake_fact = Data(x=botData_train.x, edge_index=fake_fact_graph.contiguous(), y=botData_train.y)
        #print("original treat/control: ", treat_idx_train.shape, control_idx_train.shape)
        treat_idx_ok, control_idx_ok = match_node(fake_fact_graph, botData_train.y[:, 0], prop_label_train, treat_idx_train, control_idx_train)
        #print("matched treat/control: ", treat_idx_ok.shape, control_idx_ok.shape)

        # Generator training
        # out_y1, out_yc0: (num_treat_train), out_y0, out_yc1: (num_control_train), *_prob: (num_nodes)
        optimizer_g.zero_grad()
        out_y1, out_yc0, out_y0, out_yc1, z, zf = model_g(botData_train.x, botData_train.edge_index,
                                              botData_fake_fact.x, botData_fake_fact.edge_index, treat_idx_ok, control_idx_ok)
        # out_y, target_y: (num_treat+control_train)
        out_y = torch.cat([out_y1, out_y0], dim=-1)
        target_y = torch.cat([botData_train.y[:, 1][treat_idx_ok], botData_train.y[:, 1][control_idx_ok]])
        loss_y = F.mse_loss(out_y.float(), target_y.float())
        loss_g = loss_y*par['ly']
        loss_g.backward()
        optimizer_g.step()

        # Discriminator training
        optimizer_d.zero_grad()
        fact_prob, fact_prob_f, D, Df = model_d(z.detach(), zf.detach())
        # target_judgefact, out_judgefact: (2*num_train)
        target_judgefact = torch.cat([torch.ones(len(fact_prob)),torch.zeros(len(fact_prob_f))], dim=-1)
        out_judgefact = torch.cat([fact_prob, fact_prob_f], dim=0)
        #print("pred treat compare:", torch.mean(treat_prob[treat_idx_ok]).item(), torch.mean(treat_prob[control_idx_ok]).item())
        # MI loss
        target_mi = torch.cat([torch.ones(len(D)), torch.zeros(len(Df))], dim=-1)
        out_mi = torch.cat([D, Df], dim=0)
        loss_jf = torch.nn.CrossEntropyLoss()(out_judgefact, target_judgefact.long())
        loss_m = torch.nn.BCELoss()(out_mi, target_mi.float())
        loss_d = - loss_m*par['lm'] + loss_jf*par['ljf']
        loss_d.backward()
        optimizer_d.step()

        print("{:.4f} {:.4f} {:.4f}".format(loss_y.item(), loss_jf.item(), loss_m.item()))
        if epoch%10 == 0:
            model_g.eval()
            model_d.eval()
            # 验证/测试时无需新构造fake graph
            # fake_fact_graph = shuffle_edge(edge_pool_test, edge_num=len(botData_test.edge_index[0]))  # for effect estimation
            # botData_fake_fact = Data(x=botData_test.x, edge_index=fake_fact_graph.contiguous(), y=botData_test.y)
            # treat_idx_ok, control_idx_ok = match_node(fake_fact_graph, botData_test.y[:, 0], prop_label_test,
            #                                           treat_idx_test, control_idx_test)
            out_y1, out_yc0, out_y0, out_yc1, z, zf = model_g(botData_test.x, botData_test.edge_index,
                                                                        botData_fake_fact.x, botData_fake_fact.edge_index, treat_idx_ok, control_idx_ok)

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
           'ljf': 1,
           'lm': 1}
    print(par)
    main()
