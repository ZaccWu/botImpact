import numpy as np
import pandas as pd
import torch
import random
from torch.autograd import Function
from typing import Any, Optional, Tuple
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

seed = 101

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MaskEncoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=16):
        super(MaskEncoder, self).__init__()
        self.convM1 = GATConv(in_dim, h_dim)
        self.convM2 = GATConv(h_dim, out_dim)

    def forward(self, x, edge_index):
        # edge_index: (2, num_edge)
        xM1 = F.relu(self.convM1(x, edge_index))
        xM1 = F.dropout(xM1, p=0.5, training=self.training)
        xM2 = self.convM2(xM1, edge_index)  # xM2: (num_nodes, h_dim)

        value = (xM2[edge_index[0]] * xM2[edge_index[1]]).sum(dim=1) # (num_edges)

        _, topk_homo = torch.topk(value, int(len(value)*0.95), largest=True)
        _, topk_hetero = torch.topk(value, int(len(value)*0.05), largest=False)
        return edge_index[:,topk_homo], edge_index[:,topk_hetero]


class botDetect(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=2):
        super(botDetect, self).__init__()
        self.convB1 = GATConv(in_dim, h_dim)
        self.convB2 = GATConv(h_dim, out_dim)

    def forward(self, x, edge_index):
        xB1 = F.relu(self.convB1(x, edge_index))
        xB1 = F.dropout(xB1, p=0.5, training=self.training)
        xB2 = self.convB2(xB1, edge_index)
        return xB2

def generate_counterfactual_edge(edge_pool, var_edge_index, inv_edge_index):
    # iput edge_index: (2, num_edge)
    inv_edge_pool = set(map(tuple, np.array(inv_edge_index.T)))
    var_edge_pool = set(map(tuple, np.array(var_edge_index.T)))

    selected_edge = random.sample(edge_pool - inv_edge_pool - var_edge_pool, int(len(var_edge_pool)/2))
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


def contrastive_loss(target, pred_score, m=5):
    # target: 0-1, pred_score: float
    Rs = torch.mean(pred_score)
    delta = torch.std(pred_score)
    dev_score = (pred_score - Rs)/(delta + 10e-10)
    cont_score = torch.max(torch.zeros(pred_score.shape), m-dev_score)
    loss = dev_score[(1-target).nonzero()].sum()+cont_score[target.nonzero()].sum()
    return loss

def transfer_pred(out, threshold):
    pred = out.clone()
    pred[torch.where(out < threshold)] = 0
    pred[torch.where(out >= threshold)] = 1
    return pred

def load_data(dt):
    # load train data
    edge_index_train = torch.LongTensor(np.load('Dataset/synthetic/'+dt+'_edge.npy'))    # (num_edge, 2)
    bot_label_train = np.load('Dataset/synthetic/'+dt+'_bot_label.npy')
    T_train = np.load('Dataset/synthetic/'+dt+'_T_label.npy')
    outcome_train = np.load('Dataset/synthetic/'+dt+'_y.npy')
    prop_label = np.load('Dataset/synthetic/'+dt+'_prop_label.npy')

    N = len(outcome_train)  # num of nodes
    x = degree(edge_index_train[:, 0])  # user node degree as feature
    target_var = torch.tensor(
        np.concatenate([bot_label_train[:, np.newaxis], outcome_train[:, np.newaxis], T_train[:, np.newaxis]], axis=-1))  # (num_nodes, 3)
    botData = Data(x=x.unsqueeze(-1), edge_index=edge_index_train.t().contiguous(), y=target_var)
    return botData, N, prop_label

def main():
    botData_train, N_train, prop_label_train = load_data('train')
    botData_test, N_test, prop_label_test = load_data('test')
    model1 = MaskEncoder(in_dim=1, h_dim=16, out_dim=16)
    model2 = botDetect(in_dim=1, h_dim=16, out_dim=1)
    optimizer = torch.optim.Adam([{'params': model1.parameters(), 'lr': 0.001},
                                  {'params': model2.parameters(), 'lr': 0.001}])
    # for counterfactual edge generation
    edge_pool_train = set(map(tuple, np.array([[i, j] for i in range(N_train) for j in range(i, N_train)])))|\
                      set(map(tuple, np.array([[j, i] for i in range(N_train) for j in range(i, N_train)])))
    # train
    for epoch in range(300):
        model1.train()
        model2.train()
        optimizer.zero_grad()

        homo_edge_index, hetero_edge_index = model1(botData_train.x, botData_train.edge_index)
        if epoch%5 == 0:   # 每5个epoch换一个环境
            fake_env_graph = generate_counterfactual_edge(edge_pool_train, var_edge_index=hetero_edge_index,
                                                         inv_edge_index=homo_edge_index) # for bot detection
            botData_fake_env = Data(x=botData_train.x, edge_index=fake_env_graph.contiguous(), y=botData_train.y)

        # detect bot
        out_b = model2(botData_train.x, botData_train.edge_index)   # out_b: (num_node, 1)
        out_b_fake_fact = model2(botData_fake_env.x, botData_fake_env.edge_index)   # out_b_fake: (num_node, 1)
        target_b = botData_train.y[:,0]   # target_b: (num_node)
        target_b_fake = botData_fake_env.y[:,0] # target_b_fake: (num_node)

        # 最大化所有环境下bot识别的准确度
        loss_b = contrastive_loss(target_b, out_b)
        #print("mean human: ", torch.mean(out_b[:3000]), " mean bot: ", torch.mean(out_b[3000:]))
        loss_b_fake = contrastive_loss(target_b_fake, out_b_fake_fact)
        print("{:.4f} {:.4f}".format(loss_b.item(),loss_b_fake.item()))
        loss = loss_b*par['lb'] + loss_b_fake*par['lbf']
        loss.backward()
        optimizer.step()

        if epoch%1 == 0:
            model1.eval()
            model2.eval()
            # bot detection result
            out_b = model2(botData_test.x, botData_test.edge_index)
            threshold = torch.quantile(out_b, 0.9677, dim=None, keepdim=False, interpolation='higher')
            pred_b = transfer_pred(out_b, threshold)
            pred_label_b = pred_b
            report_b = classification_report(target_b, pred_label_b.detach().numpy(), target_names=['class0', 'class1'], output_dict=True)
            # bot detection result
            out_b_fake = model2(botData_fake_env.x, botData_fake_env.edge_index)
            threshold = torch.quantile(out_b_fake, 0.9677, dim=None, keepdim=False, interpolation='higher')
            pred_b_fake = transfer_pred(out_b_fake, threshold)
            pred_label_b_fake = pred_b_fake
            report_b_fake = classification_report(target_b_fake, pred_label_b_fake.detach().numpy(), target_names=['class0', 'class1'], output_dict=True)

            # treatment effect prediction result
            print("Epoch: " + str(epoch))
            print('f1-0 {:.4f},'.format(report_b['class0']['f1-score']),
                  'rec-0 {:.4f},'.format(report_b['class0']['recall']),
                  'f1-1 {:.4f},'.format(report_b['class1']['f1-score']),
                  'rec-1 {:.4f},'.format(report_b['class1']['recall']),
                  'f1-macro {:.4f},'.format(report_b['macro avg']['f1-score']),
                  'acc {:.4f},'.format(report_b['accuracy']))
            print('f1-0 {:.4f},'.format(report_b_fake['class0']['f1-score']),
                  'rec-0 {:.4f},'.format(report_b_fake['class0']['recall']),
                  'f1-1 {:.4f},'.format(report_b_fake['class1']['f1-score']),
                  'rec-1 {:.4f},'.format(report_b_fake['class1']['recall']),
                  'f1-macro {:.4f},'.format(report_b_fake['macro avg']['f1-score']),
                  'acc {:.4f},'.format(report_b_fake['accuracy']))


    # model1.eval()
    # model2.eval()
    # out_b = model2(botData_test.x, botData_test.edge_index)
    # target_b = botData_test.y[:, 0]
    #
    # # bot detection
    # threshold = torch.quantile(out_b, 0.9677, dim=None, keepdim=False, interpolation='higher')
    # pred_b = out_b.clone()
    # pred_b[torch.where(out_b < threshold)] = 0
    # pred_b[torch.where(out_b >= threshold)] = 1
    # pred_label_b = pred_b
    # report_b = classification_report(target_b, pred_label_b.detach().numpy(), target_names=['class0', 'class1'], output_dict=True)
    #
    # print('f1-0 {:.4f},'.format(report_b['class0']['f1-score']),
    #       'rec-0 {:.4f},'.format(report_b['class0']['recall']),
    #       'f1-1 {:.4f},'.format(report_b['class1']['f1-score']),
    #       'rec-1 {:.4f},'.format(report_b['class1']['recall']),
    #       'f1-macro {:.4f},'.format(report_b['macro avg']['f1-score']),
    #       'acc {:.4f},'.format(report_b['accuracy']))


if __name__ == "__main__":
    gpu = 0
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    par = {'lb': 1,
           'lbf': 1,}
    print(par)
    main()
