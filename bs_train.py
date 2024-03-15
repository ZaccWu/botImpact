import numpy as np
import pandas as pd
import torch
import random
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from baseline_model import DD, CNE_Minus, IgniteGenerator, IgniteDiscriminator, GialGenerator, GialDiscriminator
from geomloss import SamplesLoss
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import argparse

EPS = 1e-15
parser = argparse.ArgumentParser('Benchmark')

# data parameters
parser.add_argument('--type', type=str, help='data used', default='random')
parser.add_argument('--effect_true', type=float, help='ground-truth effect', default=0) # synthetic: -1, empirical: 0
# model parameters
# ignite, gial, cne-, dd
parser.add_argument('--model', type=str, help='model test', default='ignite')
# training parameters
parser.add_argument('--epoch', type=int, help='num epochs', default=300)
parser.add_argument('--gpu', type=int, help='gpu', default=0)
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





def evaluate_metric(pred_0, pred_1, pred_c1, pred_c0):
    tau_pred = torch.cat([pred_c1, pred_1], dim=0) - torch.cat([pred_0, pred_c0], dim=0)
    print("treat ave: {:.4f}".format(torch.mean(torch.cat([pred_1, pred_c1], dim=0)).item()))
    print("control ave: {:.4f}".format(torch.mean(torch.cat([pred_0, pred_c0], dim=0)).item()))
    print("--------------------------------")
    tau_true = torch.ones(tau_pred.shape).to(device) * args.effect_true
    ePEHE = torch.sqrt(torch.mean(torch.square(tau_pred-tau_true)))
    eATE = torch.abs(torch.mean(tau_pred) - torch.mean(tau_true))
    Treat_eff = torch.mean(tau_pred)
    return eATE, ePEHE, Treat_eff

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
    #x = degree(edge_index[:, 0])  # user node degree as feature
    x = torch.FloatTensor(bot_label)
    # target: bot&human, opinion, treat&control
    target_var = torch.tensor(
        np.concatenate([bot_label[:, np.newaxis], outcome[:, np.newaxis], treat_indicator[:, np.newaxis]], axis=-1))  # (num_nodes, 3)
    botData = Data(x=x.unsqueeze(-1), edge_index=edge_index.t().contiguous(), y=target_var).to(device)
    return botData, N, prop_label




def main():
    botData_f, N_train, prop_label_train = load_data('train')
    print("Finish loading data.")
    #treat_idx, control_idx = torch.where(botData_f.y[:, 2] == -1)[0], torch.where(botData_f.y[:, 2] == 1)[0]
    treat_idx, control_idx = torch.nonzero(botData_f.y[:, 2] == -1).squeeze(), torch.nonzero(botData_f.y[:, 2] == 1).squeeze()

    if args.model == 'ignite':
        model_g = IgniteGenerator(in_dim=1, h_dim=16, out_dim=1).to(device)
        model_d = IgniteDiscriminator(h_dim=16).to(device)
        optimizer_g = torch.optim.Adam(model_g.parameters(), lr=0.001)
        optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.001)
    elif args.model == 'gial':
        model_g = GialGenerator(in_dim=1, h_dim=16, out_dim=1).to(device)
        model_d = GialDiscriminator(h_dim=16).to(device)
        optimizer_g = torch.optim.Adam(model_g.parameters(), lr=0.001)
        optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.001)
    elif args.model == 'cne-':
        model = CNE_Minus(in_dim=1, h_dim=16, out_dim=1).to(device)
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.001}])
    elif args.model == 'dd':
        model = DD(in_dim=1, h_dim=16, out_dim=1).to(device)
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.001}])

    # train
    for epoch in range(args.epoch):
        if args.model == 'ignite':
            model_g.train()
            model_d.train()

            # Generator training
            # out_y1, out_yc0: (num_treat_train), out_y0, out_yc1: (num_control_train), *_prob: (num_nodes)
            optimizer_g.zero_grad()
            out_y1, out_yc0, out_y0, out_yc1, z = model_g(botData_f.x, botData_f.edge_index, treat_idx, control_idx)
            # out_y, target_y: (num_treat+control_train)
            out_y = torch.cat([out_y1, out_y0], dim=-1)
            target_y = torch.cat([botData_f.y[:, 1][treat_idx], botData_f.y[:, 1][control_idx]])
            loss_y = F.mse_loss(out_y.float(), target_y.float())
            loss_g = loss_y*1
            loss_g.backward()
            optimizer_g.step()

            # Discriminator training
            optimizer_d.zero_grad()
            treat_prob = model_d(z.detach())
            loss_cb = torch.mean(treat_prob[treat_idx]) - torch.mean(treat_prob[control_idx])
            loss_d = - loss_cb*100
            loss_d.backward()
            optimizer_d.step()
            # Loss function
            #print("{:.4f} {:.4f}".format(loss_y.item(), loss_cb.item()))

        elif args.model == 'gial':
            model_g.train()
            model_d.train()
            # permutation
            rows, cols = botData_f.x.size()
            rand_indices = torch.randperm(rows)
            shuffled_x = botData_f.x[rand_indices]
            # Generator training
            optimizer_g.zero_grad()
            out_y1, out_yc0, out_y0, out_yc1, z, zf = model_g(botData_f.x, botData_f.edge_index,shuffled_x,
                                                              treat_idx, control_idx)
            # out_y, target_y: (num_treat+control_train)
            out_y = torch.cat([out_y1, out_y0], dim=-1)
            target_y = torch.cat([botData_f.y[:, 1][treat_idx], botData_f.y[:, 1][control_idx]])
            loss_y = F.mse_loss(out_y.float(), target_y.float())
            loss_g = loss_y * 1
            loss_g.backward()
            optimizer_g.step()

            # Discriminator training
            optimizer_d.zero_grad()
            fact_prob, fact_prob_f, D, Df = model_d(z.detach(), zf.detach())
            # target_judgefact, out_judgefact: (2*num_train)
            target_judgefact = torch.cat([torch.ones(len(fact_prob)), torch.zeros(len(fact_prob_f))], dim=-1).to(device)
            out_judgefact = torch.cat([fact_prob, fact_prob_f], dim=0)
            # print("pred treat compare:", torch.mean(treat_prob[treat_idx_ok]).item(), torch.mean(treat_prob[control_idx_ok]).item())
            # MI loss
            target_mi = torch.cat([torch.ones(len(D)), torch.zeros(len(Df))], dim=-1).to(device)
            out_mi = torch.cat([D, Df], dim=0)
            loss_jf = torch.nn.CrossEntropyLoss()(out_judgefact, target_judgefact.long())
            loss_m = torch.nn.BCELoss()(out_mi, target_mi.float())
            loss_d = - loss_m * 1 + loss_jf * 1
            loss_d.backward()
            optimizer_d.step()
            #print("{:.4f} {:.4f} {:.4f}".format(loss_y.item(), loss_jf.item(), loss_m.item()))

        elif args.model == 'cne-':
            model.train()
            optimizer.zero_grad()
            out_y1, out_yc0, out_y0, out_yc1, treat_prob, z = model(botData_f.x, botData_f.edge_index,treat_idx, control_idx)
            out_y = torch.cat([out_y1, out_y0], dim=-1)
            target_y = torch.cat([botData_f.y[:, 1][treat_idx], botData_f.y[:, 1][control_idx]])
            target_judgetreat = torch.cat(
                [torch.ones(len(treat_prob[treat_idx])), torch.zeros(len(treat_prob[control_idx]))]).to(device)
            out_judgetreat = torch.cat([treat_prob[treat_idx], treat_prob[control_idx]], dim=0)
            loss_y = F.mse_loss(out_y.float(), target_y.float())
            loss_judgetreat = torch.nn.CrossEntropyLoss()(out_judgetreat, target_judgetreat.long())
            loss = loss_y * 1 + loss_judgetreat * 1
            loss.backward()
            optimizer.step()
            #print("{:.4f} {:.4f}".format(loss_y.item(), loss_judgetreat.item()))

        elif args.model == 'dd':
            model.train()
            optimizer.zero_grad()
            out_y1, out_yc0, out_y0, out_yc1, z = model(botData_f.x, botData_f.edge_index, treat_idx,control_idx)
            out_y = torch.cat([out_y1, out_y0], dim=-1)
            target_y = torch.cat([botData_f.y[:, 1][treat_idx], botData_f.y[:, 1][control_idx]])
            loss_y = F.mse_loss(out_y.float(), target_y.float())
            samples_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.005, backend="tensorized") ## TODO: reduce the computational cost
            imbalance_loss = samples_loss(z[treat_idx], z[control_idx])
            loss = loss_y * 1 + imbalance_loss * 1
            loss.backward()
            optimizer.step()
            #print("{:.4f} {:.4f}".format(loss_y.item(), imbalance_loss.item()))

        if epoch%10 == 0:
            if args.model == 'ignite':
                model_g.eval()
                model_d.eval()
                out_y1, out_yc0, out_y0, out_yc1, z = model_g(botData_f.x, botData_f.edge_index, treat_idx, control_idx)
            elif args.model == 'gial':
                model_g.eval()
                model_d.eval()
                out_y1, out_yc0, out_y0, out_yc1, z, zf = model_g(botData_f.x, botData_f.edge_index, shuffled_x, treat_idx, control_idx)
            elif args.model == 'cne-':
                model.eval()
                out_y1, out_yc0, out_y0, out_yc1, treat_prob = model(botData_f.x, botData_f.edge_index, treat_idx, control_idx)
            elif args.model == 'dd':
                model.eval()
                out_y1, out_yc0, out_y0, out_yc1, z = model(botData_f.x, botData_f.edge_index, treat_idx, control_idx)

            # treatment effect prediction result
            eATE_test, ePEHE_test, treat_eff  = evaluate_metric(out_y0, out_y1, out_yc1, out_yc0)
            print("Epoch: " + str(epoch))
            print('eATE: {:.4f}'.format(eATE_test.detach().cpu().numpy()),
                  'ePEHE: {:.4f}'.format(ePEHE_test.detach().cpu().numpy()),
                  'Effect: {:.4f}'.format(treat_eff.detach().cpu().numpy()))
            print("================================")
            # save embedding
            if epoch == args.save_epoch:
                Z_emb, treat_lb, y_tar = z.detach().cpu().numpy(), botData_f.y[:, 2].detach().cpu().numpy(), botData_f.y[:, 1].detach().cpu().numpy()
                if args.save:
                    savedt_path = 'result/save_emb/'+args.type+'/'
                    if not os.path.isdir(savedt_path):
                        os.makedirs(savedt_path)
                    np.save(savedt_path+args.model+str(seed)+'.npy', np.concatenate([Z_emb,treat_lb[:,np.newaxis], y_tar[:,np.newaxis]], axis=-1)) # emb_H + treat (dim: z+1)




if __name__ == "__main__":
    for seed in range(101, 111):
        set_seed(seed)
        res = {'Epoch': [], 'eATE': [], 'ePEHE': [], 'rjf': [], 'lcff': [], 'ly': []}
        main()
        res = pd.DataFrame(res)
        res.to_csv('result/'+args.model+'_'+str(seed)+'.csv', index=False)
