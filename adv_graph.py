import numpy as np
import pandas as pd
import torch
import sys
import random

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import classification_report
from torch_geometric.transforms import RandomLinkSplit
import warnings
warnings.filterwarnings("ignore")
import argparse

EPS = 1e-15
parser = argparse.ArgumentParser('BotImpact')


# data parameters
parser.add_argument('--type', type=str, help='data used', default='random')
# model parameters
parser.add_argument('--mask_homo', type=float, help='mask edge percentage', default=0.6)
# training parameters
parser.add_argument('--seed', type=int, help='random seed', default=101)
parser.add_argument('--gpu', type=int, help='gpu', default=0)
parser.add_argument('--ly', type=float, help='reg for outcome pred', default=10)
parser.add_argument('--ljt', type=float, help='reg for treat pred', default=1)
parser.add_argument('--ljf', type=float, help='reg for cf discrim', default=10)


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


class MaskEncoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=16):
        super(MaskEncoder, self).__init__()
        self.convM1 = GATConv(in_dim, h_dim)
        self.convM2 = GATConv(h_dim, out_dim)

    def forward(self, x, edge_index):
        # edge_index: e(2, num_edge)
        xM1 = F.leaky_relu(self.convM1(x, edge_index))
        xM1 = F.dropout(xM1, p=0.5, training=self.training)
        xM2 = self.convM2(xM1, edge_index)  # xM2: (num_nodes, h_dim)
        value = (xM2[edge_index[0]] * xM2[edge_index[1]]).sum(dim=1) # (num_edges)
        # select homophily edges
        _, topk_homo = torch.topk(value, int(len(value)*args.mask_homo), largest=True)
        _, topk_hetero = torch.topk(value, int(len(value)*(1-args.mask_homo)), largest=False)
        return edge_index[:,topk_homo], edge_index[:,topk_hetero], xM2

class BotImpact(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=1, heads=1):
        super(BotImpact, self).__init__()
        self.convZ1 = GATConv(in_dim, h_dim, heads)
        self.convZ2 = GATConv(h_dim*heads, h_dim, heads)
        self.yNet1 = torch.nn.Sequential(torch.nn.Linear(h_dim*heads, out_dim), torch.nn.LeakyReLU())
        self.yNet0 = torch.nn.Sequential(torch.nn.Linear(h_dim*heads, out_dim), torch.nn.LeakyReLU())
        #self.balanceNet = torch.nn.Sequential(torch.nn.Linear(h_dim, 2), torch.nn.LeakyReLU())
        #self.propenNet = torch.nn.Sequential(torch.nn.Linear(h_dim, 2), torch.nn.LeakyReLU())

        self.propenNet = torch.nn.Sequential(torch.nn.Linear(h_dim*heads, h_dim), torch.nn.LeakyReLU(), torch.nn.Linear(h_dim, 2), torch.nn.LeakyReLU())

    def forward(self, x, edge_index, fake_x, fake_edge_index, treat_idx, control_idx):
        # generate node embedding
        xZ1, xfZ1 = F.relu(self.convZ1(x, edge_index)), F.relu(self.convZ1(fake_x, fake_edge_index))
        xZ1, xfZ1 = F.dropout(xZ1, p=0.5, training=self.training), F.dropout(xfZ1, p=0.5, training=self.training)
        xZ2, xfZ2 = self.convZ2(xZ1, edge_index), self.convZ2(xfZ1, fake_edge_index)    # xZ2/xfZ2: (num_nodes, h_dim)

        # predict outcome
        y1, yc0 = self.yNet1(xZ2[treat_idx]), self.yNet0(xfZ2[treat_idx])
        y0, yc1 = self.yNet0(xZ2[control_idx]), self.yNet1(xfZ2[control_idx])

        # judge the node is treated or controled
        tprob, tprob_f = self.propenNet(xZ2), self.propenNet(xfZ2)
        return y1.squeeze(-1), yc0.squeeze(-1), y0.squeeze(-1), yc1.squeeze(-1), xZ2, xfZ2, tprob.squeeze(-1), tprob_f.squeeze(-1)


class Discriminator(torch.nn.Module):
    def __init__(self, h_dim, heads=1):
        super(Discriminator, self).__init__()
        self.balanceNet = torch.nn.Sequential(torch.nn.Linear(h_dim * heads, 2))

    def forward(self, xZ2, xfZ2):
        # judge the node is from factual & counterfactual graph
        fprob, fprob_f = self.balanceNet(xZ2), self.balanceNet(xfZ2)
        return fprob.squeeze(-1), fprob_f.squeeze(-1)

class InnerProductDecoder(torch.nn.Module):
	def forward(self, z, edge_index, sigmoid=True):
		value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
		return torch.sigmoid(value) if sigmoid else value

	def forward_all(self, z, sigmoid=True):
		adj = torch.matmul(z, z.t())
		return torch.sigmoid(adj) if sigmoid else adj

def recon_loss(z, edge_tar, neg_edge_tar):
	decoder = InnerProductDecoder()
	pos_edge_index, neg_edge_index = edge_tar, neg_edge_tar
	pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
	neg_loss = -torch.log(1 - decoder(z, neg_edge_index, sigmoid=True) +
						  EPS).mean()
	return pos_loss + neg_loss

def generate_counterfactual_edge(edge_pool, var_edge_index, inv_edge_index):
    # iput edge_index: (2, num_edge)
    inv_edge_pool = set(map(tuple, np.array(inv_edge_index.T)))
    var_edge_pool = set(map(tuple, np.array(var_edge_index.T)))
    selected_edge = random.sample(edge_pool - inv_edge_pool - var_edge_pool, int(len(inv_edge_pool)/2))
    selected_edge = torch.LongTensor(list(map(list, selected_edge)))
    selected_edge = torch.cat([selected_edge,torch.flip(selected_edge,[1])],dim=0)
    new_edge_index = torch.cat([inv_edge_index, selected_edge.T], dim=1)
    return new_edge_index

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
    edge_index_train = torch.LongTensor(np.load('Dataset/synthetic/'+args.type+'/'+dt+'_edge.npy'))    # (num_edge, 2)
    bot_label_train = np.load('Dataset/synthetic/'+args.type+'/'+dt+'_bot_label.npy')
    T_train = np.load('Dataset/synthetic/'+args.type+'/'+dt+'_T_label.npy')
    outcome_train = np.load('Dataset/synthetic/'+args.type+'/'+dt+'_y.npy')
    prop_label = np.load('Dataset/synthetic/'+args.type+'/'+dt+'_prop_label.npy')
    N = len(outcome_train)  # num of nodes
    x = degree(edge_index_train[:, 0])  # user node degree as feature
    #x = torch.eye(N)

    # target: bot&human, opinion, treat&control
    target_var = torch.tensor(
        np.concatenate([bot_label_train[:, np.newaxis], outcome_train[:, np.newaxis], T_train[:, np.newaxis]], axis=-1))  # (num_nodes, 3)
    botData = Data(x=x.unsqueeze(-1), edge_index=edge_index_train.t().contiguous(), y=target_var)
    return botData, N, prop_label

def main():
    botData_train, N_train, prop_label_train = load_data('train')

    train_edge, _, _ = RandomLinkSplit(is_undirected=False, num_val=0, num_test=0, neg_sampling_ratio=1.0)(botData_train)

    model_f = MaskEncoder(in_dim=1, h_dim=32, out_dim=32)
    model_g = BotImpact(in_dim=1, h_dim=32, out_dim=1)
    model_d = Discriminator(h_dim=32)

    optimizer_fg = torch.optim.Adam([{'params': model_f.parameters(), 'lr': 0.001},
                                  {'params': model_g.parameters(), 'lr': 0.001}])
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.001)

    # for counterfactual edge generation
    edge_pool_train = set(map(tuple, np.array([[i, j] for i in range(N_train) for j in range(i, N_train)])))
    treat_idx_train, control_idx_train = torch.where(botData_train.y[:, 2]==-1)[0], torch.where(botData_train.y[:, 2]==1)[0]

    # train
    for epoch in range(300):
        model_f.train()
        model_g.train()
        model_d.train()

        # mask encoder
        optimizer_fg.zero_grad()
        homo_edge_index, hetero_edge_index, Z_f = model_f(botData_train.x, botData_train.edge_index)

        # counterfactual graph generation
        cfgraph_edge = generate_counterfactual_edge(edge_pool_train, var_edge_index=homo_edge_index,
                                                       inv_edge_index=hetero_edge_index) # for effect estimation
        botData_cf = Data(x=botData_train.x, edge_index=cfgraph_edge.contiguous(), y=botData_train.y)
        #print("treat/control: ", treat_idx_train.shape, control_idx_train.shape)
        # assess bot
        # out_y1, out_yc0: (num_treat_train), out_y0, out_yc1: (num_control_train), *_prob: (num_nodes)
        out_y1, out_yc0, out_y0, out_yc1, Zf, Zcf, treat_prob, treat_prob_f = model_g(botData_train.x, botData_train.edge_index,
                                              botData_cf.x, botData_cf.edge_index, treat_idx_train, control_idx_train)
        out_y = torch.cat([out_y1, out_y0], dim=-1)
        target_y = torch.cat([botData_train.y[:, 1][treat_idx_train], botData_train.y[:, 1][control_idx_train]])

        target_judgetreat = torch.cat([torch.ones(len(treat_prob[treat_idx_train])),torch.zeros(len(treat_prob[control_idx_train]))])
        out_judgetreat = torch.cat([treat_prob[treat_idx_train],treat_prob[control_idx_train]], dim=0)
        # loss function
        loss_y = F.mse_loss(out_y.float(), target_y.float())
        loss_judgetreat = torch.nn.CrossEntropyLoss()(out_judgetreat, target_judgetreat.long())
        loss_g = loss_y*args.ly + loss_judgetreat*args.ljt
        loss_g.backward()
        optimizer_fg.step()


        # counterfactual discriminator
        optimizer_d.zero_grad()
        fact_prob, fact_prob_f = model_d(Zf.detach(), Zcf.detach())
        # target_judgefact, out_judgefact: (2*num_train)
        target_judgefact = torch.cat([torch.ones(len(fact_prob)),torch.zeros(len(fact_prob_f))], dim=-1)
        out_judgefact = torch.cat([fact_prob, fact_prob_f], dim=0)
        #print("pred treat compare:", torch.mean(treat_prob[treat_idx_ok]).item(), torch.mean(treat_prob[control_idx_ok]).item())
        loss_judgefact = torch.nn.CrossEntropyLoss()(out_judgefact, target_judgefact.long())
        loss_d = - loss_judgefact * args.ljf
        loss_d.backward()   # loss_d越小, discriminator越不容易区分两者
        optimizer_d.step()

        print("{:.4f} {:.4f}".format(loss_g.item(),loss_d.item()))



        if epoch%10 == 0:
            model_g.eval()
            # whether the node have a counterfactual counterpart
            treat_idx_ok, control_idx_ok = match_node(cfgraph_edge, botData_train.y[:, 0], prop_label_train,
                                                      treat_idx_train, control_idx_train)
            out_y1, out_yc0, out_y0, out_yc1, _, _, _, _ = model_g(botData_train.x, botData_train.edge_index,
                                                                        botData_cf.x, botData_cf.edge_index, treat_idx_ok, control_idx_ok)
            print("treat/control: ", treat_idx_ok.shape, control_idx_ok.shape)
            eATE_test, ePEHE_test = evaluate_metric(out_y0, out_y1, out_yc1, out_yc0)
            print("Epoch: " + str(epoch))
            print('eATE: {:.4f}'.format(eATE_test.detach().numpy()),
                  'ePEHE: {:.4f}'.format(ePEHE_test.detach().numpy()))


if __name__ == "__main__":

    main()
