import numpy as np
import pandas as pd
import torch
import sys
import random
import os
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import GATConv

from util import pairwise_similarity, similarity_check, match_node

import warnings
warnings.filterwarnings("ignore")
import argparse

EPS = 1e-15
parser = argparse.ArgumentParser('BotImpact')

# data parameters
parser.add_argument('--type', type=str, help='data used', default='t1_pos')
parser.add_argument('--effect_true', type=float, help='ground-truth effect', default=0) # synthetic: -1, empirical: 0
# model parameters
parser.add_argument('--mask_homo', type=float, help='mask edge percentage', default=0.6) # 0.6
# training parameters
parser.add_argument('--epoch', type=int, help='num epochs', default=800) # syn: 300, emp: 500
parser.add_argument('--gpu', type=int, help='gpu', default=0)
parser.add_argument('--ly', type=float, help='reg for outcome pred', default=100) # emp: 100
parser.add_argument('--ljt', type=float, help='reg for treat pred', default=0.01) # emp: 0.01
parser.add_argument('--ljg', type=float, help='reg for cf generate', default=100) # emp: 10
parser.add_argument('--ljd', type=float, help='reg for cf discrim', default=1) # emp: 1
# saving embedding
parser.add_argument('--save', type=bool, help='whether save emb', default=False)
parser.add_argument('--save_epoch', type=int, help='saving emb epoch', default=150)


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_data(dt='train'):
    if args.type in ['random', 'randomu', 'highbc', 'highcc', 'lowdu', 'highdu']:
        # load train data
        edge_index = torch.LongTensor(np.load('Dataset/synthetic/'+args.type+'/'+dt+'_edge.npy'))    # (num_edge, 2)
        bot_label = np.load('Dataset/synthetic/'+args.type+'/'+dt+'_bot_label.npy')
        treat_indicator = np.load('Dataset/synthetic/'+args.type+'/'+dt+'_T_label.npy')
        outcome = np.load('Dataset/synthetic/'+args.type+'/'+dt+'_y.npy')
        prop_label = np.load('Dataset/synthetic/'+args.type+'/'+dt+'_prop_label.npy')
    elif args.type in ['t1_pos', 't2_pos', 't3_pos', 't1_neg', 't2_neg', 't3_neg']:
        # load train data
        edge_index = torch.LongTensor(np.load('Dataset/twi22/'+args.type[:2]+'/'+args.type+'_edge.npy'))    # (num_edge, 2)
        bot_label = np.load('Dataset/twi22/'+args.type[:2]+'/'+args.type+'_bot_label.npy')
        treat_indicator = np.load('Dataset/twi22/'+args.type[:2]+'/'+args.type+'_T_label.npy')
        outcome = np.load('Dataset/twi22/'+args.type[:2]+'/'+args.type+'_y.npy')
        prop_label = np.load('Dataset/twi22/'+args.type[:2]+'/'+args.type+'_prop_label.npy')
    # cal basic
    N = len(outcome)  # num of nodes
    #x = torch.FloatTensor(degree(edge_index[:, 0]))  # user node degree as feature
    x = torch.FloatTensor(bot_label)
    # target: bot&human, opinion, treat&control
    target_var = torch.tensor(
        np.concatenate([bot_label[:, np.newaxis], outcome[:, np.newaxis], treat_indicator[:, np.newaxis]], axis=-1))  # (num_nodes, 3)
    botData = Data(x=x.unsqueeze(-1), edge_index=edge_index.t().contiguous(), y=target_var).to(device)
    return botData, N, prop_label


class MaskEncoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=16):
        super(MaskEncoder, self).__init__()
        self.convM1 = GATConv(in_dim, h_dim)
        self.convM2 = GATConv(h_dim, out_dim)

    def forward(self, x, edge_index):
        # edge_index: e(2, num_edge)
        xM1 = F.leaky_relu(self.convM1(x, edge_index))
        #xM1 = F.dropout(xM1, p=0.5, training=self.training)
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
        self.yNetS = torch.nn.Sequential(torch.nn.Linear(h_dim * heads, h_dim), torch.nn.LeakyReLU())
        
        self.yNet1 = torch.nn.Sequential(torch.nn.Linear(h_dim, h_dim), torch.nn.LeakyReLU(), torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())
        self.yNet0 = torch.nn.Sequential(torch.nn.Linear(h_dim, h_dim), torch.nn.LeakyReLU(), torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())
        self.propenNet = torch.nn.Sequential(torch.nn.Linear(h_dim*heads, 2), torch.nn.LeakyReLU())

    def forward(self, x, edge_index, fake_x, fake_edge_index, treat_idx, control_idx):
        # generate node embedding
        xZ1, xfZ1 = F.relu(self.convZ1(x, edge_index)), F.relu(self.convZ1(fake_x, fake_edge_index))
        xZ1, xfZ1 = F.dropout(xZ1, p=0.5, training=self.training), F.dropout(xfZ1, p=0.5, training=self.training)
        xZ2, xfZ2 = self.convZ2(xZ1, edge_index), self.convZ2(xfZ1, fake_edge_index)    # xZ2/xfZ2: (num_nodes, h_dim)
        # predict outcome
        y1, yc0 = self.yNet1(self.yNetS(xZ2[treat_idx])), self.yNet0(self.yNetS(xfZ2[treat_idx])) # xfZ的treat_idx是counterfactual中的control
        y0, yc1 = self.yNet0(self.yNetS(xZ2[control_idx])), self.yNet1(self.yNetS(xfZ2[control_idx]))
        # predict treatment
        tprob = self.propenNet(xZ2)
        return y1.squeeze(-1), yc0.squeeze(-1), y0.squeeze(-1), yc1.squeeze(-1), xZ2, xfZ2, tprob.squeeze(-1)


class Discriminator(torch.nn.Module):
    def __init__(self, h_dim, heads=1):
        super(Discriminator, self).__init__()
        self.balanceNet = torch.nn.Sequential(torch.nn.Linear(h_dim * heads, h_dim), torch.nn.LeakyReLU(),
                                              torch.nn.Linear(h_dim, 1), torch.nn.Sigmoid())
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
    neg_loss = -torch.log(1 - decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
    return pos_loss + neg_loss

def generate_counterfactual_edge2(N, N_var_edge, inv_edge_index):
    id_seq = [i for i in range(N)]
    follower = random.choices(id_seq, k=N_var_edge)
    following = random.choices(id_seq, k=N_var_edge)
    selected_edge = torch.cat([torch.LongTensor(follower).unsqueeze(-1),torch.LongTensor(following).unsqueeze(-1)],dim=-1)
    new_edge_index = torch.cat([inv_edge_index, selected_edge.T.to(device)], dim=1)
    return new_edge_index

def evaluate_metric(pred_0, pred_1, pred_c1, pred_c0):
    tau_pred = torch.cat([pred_c1, pred_1], dim=0) - torch.cat([pred_0, pred_c0], dim=0)
    ave_treat = torch.mean(torch.cat([pred_1, pred_c1], dim=0)).item()
    ave_control = torch.mean(torch.cat([pred_0, pred_c0], dim=0)).item()
    print("treat ave: {:.4f}".format(ave_treat))
    print("control ave: {:.4f}".format(ave_control))
    print("--------------------------------")
    tau_true = torch.ones(tau_pred.shape).to(device) * args.effect_true
    ePEHE = torch.sqrt(torch.mean(torch.square(tau_pred-tau_true)))
    eATE = torch.abs(torch.mean(tau_pred) - torch.mean(tau_true))
    Treat_eff = torch.mean(tau_pred)
    return eATE, ePEHE, Treat_eff, ave_treat, ave_control

def main():
    botData_f, N_train, prop_label_train = load_data('train')
    print("Finish loading data.")
    model_f = MaskEncoder(in_dim=1, h_dim=32, out_dim=32).to(device)
    model_g = BotImpact(in_dim=1, h_dim=32, out_dim=1).to(device)
    model_d = Discriminator(h_dim=32).to(device)
    optimizer_fg = torch.optim.Adam([{'params': model_f.parameters(), 'lr': 0.001},
                                  {'params': model_g.parameters(), 'lr': 0.001}])
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.001)

    # for counterfactual edge generation
    treat_idx, control_idx = torch.nonzero(botData_f.y[:, 2] == -1).squeeze(), torch.nonzero(botData_f.y[:, 2] == 1).squeeze()

    # split train-val (outcome mask) 也可以根据test y MSE选模型
    treat_rd_idx, control_rd_idx = torch.randperm(treat_idx.size(0)), torch.randperm(control_idx.size(0))
    Ntreat_train, Ncontrol_train = int(treat_idx.size(0) * 0.5), int(control_idx.size(0) * 0.5)
    treat_idx_train, control_idx_train = treat_idx[treat_rd_idx[:Ntreat_train]], control_idx[control_rd_idx[:Ncontrol_train]]
    treat_idx_test, control_idx_test = treat_idx[treat_rd_idx[Ntreat_train:]], control_idx[control_rd_idx[Ncontrol_train:]]
    print("Preparing for training...")

    # for training record
    r_y, r_jt, r_cffool, r_jf = 0, 0, 0, 0

    # train
    for epoch in range(args.epoch):
        model_f.train()
        model_g.train()
        model_d.train()

        # Generator
        optimizer_fg.zero_grad()
        homo_edge_index, hetero_edge_index, Z_f = model_f(botData_f.x, botData_f.edge_index) # mask encoder
        N_homo_edge, N_hetero_edge = homo_edge_index.shape[-1], hetero_edge_index.shape[-1]

        # counterfactual graph generation
        cfgraph_edge = generate_counterfactual_edge2(N_train, N_homo_edge, hetero_edge_index) # for effect estimation
        botData_cf = Data(x=botData_f.x, edge_index=cfgraph_edge.contiguous(), y=botData_f.y).to(device)
        #print("treat/control: ", treat_idx_train.shape, control_idx_train.shape)

        # Model output
        # out_y1, out_yc0: (num_treat_train), out_y0, out_yc1: (num_control_train), *_prob: (num_nodes)
        out_y1, out_yc0, out_y0, out_yc1, Zf, Zcf, treat_prob = model_g(botData_f.x, botData_f.edge_index,
                                              botData_cf.x, botData_cf.edge_index, treat_idx_train, control_idx_train)

        # Task 1: Predict the outcome and treatment
        out_y = torch.cat([out_y1, out_y0], dim=-1)
        target_y = torch.cat([botData_f.y[:, 1][treat_idx_train], botData_f.y[:, 1][control_idx_train]])
        loss_y = F.mse_loss(out_y.float(), target_y.float())
        # Task 2: fool the discriminator
        _, fact_prob_cf = model_d(Zf.detach(), Zcf.detach())
        loss_cffool = torch.nn.BCELoss()(fact_prob_cf, torch.ones_like(fact_prob_cf))
        # Reg 1: Max the discrepancy between treat/control
        loss_jt = - pairwise_similarity(Zf[treat_idx], Zf[control_idx]).mean()
        # Generator Loss
        loss_g = loss_y*args.ly + loss_jt*args.ljt + loss_cffool * args.ljg
        loss_g.backward()
        optimizer_fg.step()

        # Discriminator
        optimizer_d.zero_grad()
        fact_prob, fact_prob_cf = model_d(Zf.detach(), Zcf.detach())
        loss_jf = torch.nn.BCELoss()(torch.cat([fact_prob, fact_prob_cf], dim=0),
                                            torch.cat([torch.ones_like(fact_prob), torch.zeros_like(fact_prob_cf)], dim=0))
        loss_d = loss_jf * args.ljd
        loss_d.backward()
        optimizer_d.step()

        # check the loss function
        print("ly, ljt, lcff, ljf: {:.4f} {:.4f} {:.4f} {:.4f}".format(loss_y.item(), loss_jt.item(),
                                                                       loss_cffool.item(), loss_jf.item()))
        r_cffool, r_jf = r_cffool+loss_cffool.item(), r_jf+loss_jf.item()

        # Evaluation
        if epoch%10 == 0:
            model_g.eval()
            # out_y1, out_yc0: (num_treat_train), out_y0, out_yc1: (num_control_train), *_prob: (num_nodes)
            out_y1, _, out_y0, _, Zf, Zcf, _ = model_g(botData_f.x, botData_f.edge_index,
                                                                            botData_cf.x, botData_cf.edge_index,
                                                                            treat_idx_test, control_idx_test)
            # predict the outcome and treatment
            out_y = torch.cat([out_y1, out_y0], dim=-1)
            target_y = torch.cat([botData_f.y[:, 1][treat_idx_test], botData_f.y[:, 1][control_idx_test]])
            outcome_MSE = F.mse_loss(out_y.float(), target_y.float())

            # whether the node have a counterfactual counterpart
            treat_idx_ok, control_idx_ok = match_node(cfgraph_edge, botData_f.y[:, 0], prop_label_train,
                                                      treat_idx, control_idx)
            out_y1, out_yc0, out_y0, out_yc1, Zf, Zcf, _ = model_g(botData_f.x, botData_f.edge_index,
                                                                        botData_cf.x, botData_cf.edge_index, treat_idx_ok, control_idx_ok)

            print("treat/control: ", treat_idx_ok.shape, control_idx_ok.shape)
            eATE_test, ePEHE_test, treat_eff, ave_treat, ave_control = evaluate_metric(out_y0, out_y1, out_yc1, out_yc0)
            print("Epoch: " + str(epoch), "Data: ", args.type)
            #similarity_check(Zf[treat_idx_ok], Zcf[control_idx_ok])
            #similarity_check(Zf[control_idx_ok], Zcf[treat_idx_ok])
            print('eATE: {:.4f}'.format(eATE_test.detach().cpu().numpy()),
                  'ePEHE: {:.4f}'.format(ePEHE_test.detach().cpu().numpy()),
                  'Effect: {:.4f}'.format(treat_eff.detach().cpu().numpy()))
            print("================================")

            res['Epoch'].append(epoch)
            res['aveT'].append(ave_treat)
            res['aveC'].append(ave_control)
            res['eATE'].append(eATE_test.detach().cpu().numpy())
            res['ePEHE'].append(ePEHE_test.detach().cpu().numpy())
            res['lcff'].append(np.mean(r_cffool))
            res['rjf'].append(np.mean(r_jf))
            res['ly'].append(outcome_MSE.detach().cpu().numpy())

            r_cffool, r_jf = 0, 0

            # save embedding
            if epoch == args.save_epoch:
                Z_emb, treat_lb, y_tar = Zf.detach().cpu().numpy(), botData_f.y[:, 2].detach().cpu().numpy(), botData_f.y[:, 1].detach().cpu().numpy()
                if args.save:
                    savedt_path = 'result/save_emb/'+args.type+'/'
                    if not os.path.isdir(savedt_path):
                        os.makedirs(savedt_path)
                    # emb_H + treat + y
                    np.save(savedt_path + 'AdvG'+str(seed)+'.npy', np.concatenate([Z_emb,treat_lb[:,np.newaxis],y_tar[:,np.newaxis]], axis=-1))


if __name__ == "__main__":
    for seed in range(101, 102):
        set_seed(seed)
        res = {'Epoch': [], 'aveT': [], 'aveC': [], 'eATE': [], 'ePEHE': [], 'rjf': [], 'lcff': [], 'ly': []}
        main()
        res = pd.DataFrame(res)
        res.to_csv('result/AdvG_'+args.type+str(seed)+'.csv', index=False)

