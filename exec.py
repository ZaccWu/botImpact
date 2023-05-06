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

class MaskEncoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=16):
        super(MaskEncoder, self).__init__()
        self.convM1 = GCNConv(in_dim, h_dim)
        self.convM2 = GCNConv(h_dim, out_dim)

    def forward(self, x, edge_index):
        xM1 = F.relu(self.convB1(x, edge_index))
        xM1 = F.dropout(xM1, p=0.5, training=self.training)
        xM2 = self.convB2(xM1, edge_index)

        value = torch.sigmoid((xM2[edge_index[0]] * xM2[edge_index[1]]).sum(dim=1))
        _, topk_homo = torch.topk(value, int(len(value)*0.5), largest=True)
        _, topk_hetero = torch.topk(value, int(len(value)*0.5), largest=False)
        return edge_index[topk_homo], edge_index[topk_hetero]


class botDetect(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=2):
        super(botDetect, self).__init__()
        self.convB1 = GCNConv(in_dim, h_dim)
        self.convB2 = GCNConv(h_dim, out_dim)

    def forward(self, x, edge_index):
        xB1 = F.relu(self.convB1(x, edge_index))
        xB1 = F.dropout(xB1, p=0.5, training=self.training)
        xB2 = self.convB2(xB1, edge_index)
        return xB2

class impactDetect(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=1):
        super(impactDetect, self).__init__()
        self.convZ1 = GCNConv(in_dim, h_dim)
        self.convZ2 = GCNConv(h_dim, h_dim)

        self.yNet = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.ReLU())
        self.propenNet = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.ReLU())
        self.balanceNet = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.ReLU())

    def forward(self, x, edge_index, fake_x, fake_edge_index):
        # generate node embedding
        xZ1, xfZ1 = F.relu(self.convZ1(x, edge_index)), F.relu(self.convZ1(fake_x, fake_edge_index))
        xZ1, xfZ1 = F.dropout(xZ1, p=0.5, training=self.training), F.dropout(xfZ1, p=0.5, training=self.training)
        xZ2, xfZ2 = self.convZ2(xZ1, edge_index), self.convZ2(xfZ1, fake_edge_index)

        # predict outcome
        yi = self.yNet(xZ2)
        # judge the node is from factual & counterfactual graph
        fprob, fprob_f = self.propenNet(xZ2), self.propenNet(xfZ2)
        # judge the node is treated or controled
        treat_prob = self.balanceNet(xZ2)

        return yi, fact_prob, fact_prob_f, treat_prob



def generate_counterfactual_edge(N, var_edge_index, inv_edge_index):
    '''
    去掉var_edge生成反事实的，保留inv_edge
    '''
    return new_edge_index


def anomaly_loss(target, pred_score, m=1):
    # target: 0-1, pred_score: float
    Rs = torch.mean(pred_score)
    delta = torch.std(pred_score)
    dev_score = (pred_score - Rs)/delta
    cont_score = torch.max(torch.zeros(pred_score.shape), m-dev_score)
    loss = dev_score[(1-target).nonzero()].sum()+cont_score[target.nonzero()].sum()
    return loss

def evaluate_metric(pred_0, pred_1, pred_c0, pred_c1):
    tau_pred = torch.cat([pred_1, pred_c1], dim=0) - torch.cat([pred_0, pred_c0], dim=0)
    tau_true = torch.ones(tau_pred.shape)
    ePEHE = torch.sqrt(torch.mean(torch.square(tau_pred-tau_true)))
    eATE = torch.abs(torch.mean(tau_pred) - torch.mean(tau_true))
    return eATE, ePEHE

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

model1 = MaskEncoder(in_dim=1, h_dim=16, out_dim=16)
model2 = botDetect(in_dim=1, h_dim=16, out_dim=1)
model3 = impactDetect()
optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)

def train():
    for epoch in range(100):
        model1.train()
        model2.train()
        optimizer.zero_grad()

        homo_edge_index, hetero_edge_index = model1(botData.x, botData.edge_index)
        fake_env_graph = generate_counterfactual_edge(N, var_edge_index=homo_edge_index,
                                                      inv_edge_index=hetero_edge_index) # for bot detection
        fake_fact_graph = generate_counterfactual_edge(N, var_edge_index=hetero_edge_index,
                                                       inv_edge_index=homo_edge_index) # for effect estimation
        botData_fake_env = Data(x=x.unsqueeze(-1), edge_index=fake_env_graph.t().contiguous(), y=target_var, train_mask=train_mask,
                       test_mask=test_mask)
        botData_fake_fact = Data(x=x.unsqueeze(-1), edge_index=fake_fact_graph.t().contiguous(), y=target_var, train_mask=train_mask,
                       test_mask=test_mask)

        # detect bot
        out_b = model2(botData.x, botData.edge_index)
        out_b_fake_fact = model2(botData_fake_env.x, botData_fake_env.edge_index)
        target_b = botData.y[:,0][botData.train_mask]
        target_b_fake = botData_fake_env.y[:,0][botData.train_mask]

        # 最大化所有环境下bot识别的准确度
        loss_b = anomaly_loss(target_b, out_b[botData.train_mask])
        loss_b_fake = anomaly_loss(target_b_fake, out_b_fake_fact[botData.train_mask])



        # assess bot impact
        out_y, fact_prob, fact_prob_f, treat_prob = model3(botData.x, botData.edge_index,
                                              botData_fake_fact.x, botData_fake_fact.edge_index)
        target_y = botData.y[:, 1][botData.train_mask]
        target_judgefact = torch.cat([torch.ones(len(fact_prob)),torch.zeros(len(fact_prob_f))], dim=-1)
        out_judgefact = torch.cat([fact_prob, fact_prob_f])

        loss_y = anomaly_loss(target_y, out_y[botData.train_mask])

        loss = loss_b + loss_b_fake + loss_y
        loss.backward()
        optimizer.step()

        if epoch%10 == 0:
            model2.eval()
            out_b = model2(botData.x, botData.edge_index)
            target_b, target_y = botData.y[:, 0][botData.train_mask], botData.y[:, 1][botData.train_mask]
            tc_label = botData.y[:, 2][botData.train_mask]

            # bot detection result
            threshold = torch.quantile(out_b, 0.96, dim=None, keepdim=False, interpolation='higher')
            pred_b = out_b.clone()
            pred_b[torch.where(out_b < threshold)] = 0
            pred_b[torch.where(out_b >= threshold)] = 1
            pred_label_b = pred_b[botData.train_mask]
            # outcome prediction result
            threshold = torch.quantile(out_y, 0.83, dim=None, keepdim=False, interpolation='higher')
            pred_y = out_y.clone()
            pred_y[torch.where(out_y < threshold)] = 0
            pred_y[torch.where(out_y >= threshold)] = 1
            pred_label_y = pred_y[botData.train_mask]
            # treatment effect prediction result
            idx_t, idx_c = torch.where(tc_label==1)[0], torch.where(tc_label==0)[0]
            true_t, true_c = target_y[idx_t], target_y[idx_c]
            pred_t, pred_c = pred_label_y[idx_t], pred_label_y[idx_c]
            # TODO: generate counterfactual outcome
            #eATE_test, ePEHE_test = evaluate_metric(pred_0, pred_1, pred_c0, pred_c1)


            report_b = classification_report(target_b, pred_label_b.detach().numpy())
            report_y = classification_report(target_y, pred_label_y.detach().numpy())
            print("Epoch " + str(epoch) + " bot-detect report:", report_b)
            print("Epoch "+str(epoch)+" outcome report:", report_y)



@torch.no_grad()
def test():
    model2.eval()
    out_b = model2(botData.x, botData.edge_index)

    # bot detection
    threshold = torch.quantile(out_b, 0.96, dim=None, keepdim=False, interpolation='higher')
    pred_b = out_b.clone()
    pred_b[torch.where(out_b < threshold)] = 0
    pred_b[torch.where(out_b >= threshold)] = 1
    pred_label_b = pred_b[botData.test_mask]
    target_b = botData.y[:,0][botData.test_mask]

    threshold = torch.quantile(out_y, 0.83, dim=None, keepdim=False, interpolation='higher')
    pred_y = out_y.clone()
    pred_y[torch.where(out_y < threshold)] = 0
    pred_y[torch.where(out_y >= threshold)] = 1
    pred_label_y = pred_y[botData.train_mask]
    target_y = botData.y[:, 1][botData.train_mask]

    report_b = classification_report(target_b, pred_label_b)
    report_y = classification_report(target_y, pred_label_y)
    print("bot-detect report:", report_b)
    print("outcome report:", report_y)


    # purchase prediction


train()
test()
