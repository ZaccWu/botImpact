import numpy as np
import pandas as pd
import torch
import random
from torch.autograd import Function
from typing import Any, Optional, Tuple
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.metrics import classification_report

seed = 101
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GRL_Layer(torch.nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()
    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class MaskEncoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=16):
        super(MaskEncoder, self).__init__()
        self.convM1 = GCNConv(in_dim, h_dim)
        self.convM2 = GCNConv(h_dim, out_dim)

    def forward(self, x, edge_index):
        # edge_index: (2, num_edge)
        xM1 = F.relu(self.convM1(x, edge_index))
        xM1 = F.dropout(xM1, p=0.5, training=self.training)
        xM2 = self.convM2(xM1, edge_index)  # xM2: (num_nodes, h_dim)

        value = (xM2[edge_index[0]] * xM2[edge_index[1]]).sum(dim=1) # (num_edges)

        _, topk_homo = torch.topk(value, int(len(value)*0.5), largest=True)
        _, topk_hetero = torch.topk(value, int(len(value)*0.5), largest=False)
        return edge_index[:,topk_homo], edge_index[:,topk_hetero]


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
        self.convZ1 = SAGEConv(in_dim, h_dim)
        self.convZ2 = SAGEConv(h_dim, h_dim)
        self.yNet1 = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())
        self.yNet0 = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())
        self.balanceNet = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())
        self.propenNet = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())
        self.grl = GRL_Layer()

    def forward(self, x, edge_index, fake_x, fake_edge_index, treat_idx, control_idx, mask):
        # generate node embedding
        xZ1, xfZ1 = F.relu(self.convZ1(x, edge_index)), F.relu(self.convZ1(fake_x, fake_edge_index))
        xZ1, xfZ1 = F.dropout(xZ1, p=0.5, training=self.training), F.dropout(xfZ1, p=0.5, training=self.training)
        xZ2, xfZ2 = self.convZ2(xZ1, edge_index), self.convZ2(xfZ1, fake_edge_index)    # xZ2/xfZ2: (num_nodes, h_dim)

        # predict outcome
        y1, yc0 = self.yNet1(xZ2[mask][treat_idx]), self.yNet1(xfZ2[mask][treat_idx])
        y0, yc1 = self.yNet0(xZ2[mask][control_idx]), self.yNet0(xfZ2[mask][control_idx])
        # judge the node is from factual & counterfactual graph
        fprob, fprob_f = self.propenNet(xZ2), self.propenNet(xfZ2)
        # judge the node is treated or controled
        treat_prob = self.balanceNet(self.grl(xZ2))
        return y1.squeeze(-1), yc0.squeeze(-1), y0.squeeze(-1), yc1.squeeze(-1), fprob.squeeze(-1), fprob_f.squeeze(-1), treat_prob.squeeze(-1)



def generate_counterfactual_edge(edge_pool, var_edge_index, inv_edge_index):
    # iput edge_index: (2, num_edge)
    inv_edge_pool = set(map(tuple, np.array(inv_edge_index.T)))
    var_edge_pool = set(map(tuple, np.array(var_edge_index.T)))
    selected_edge = random.sample(edge_pool - inv_edge_pool - var_edge_pool, len(inv_edge_pool))
    selected_edge = torch.LongTensor(list(map(list, selected_edge)))
    new_edge_index = torch.cat([inv_edge_index, selected_edge.T], dim=1)
    return new_edge_index


def contrastive_loss(target, pred_score, m=1):
    # target: 0-1, pred_score: float
    Rs = torch.mean(pred_score)
    delta = torch.std(pred_score)
    dev_score = (pred_score - Rs)/(delta + 10e-10)
    cont_score = torch.max(torch.zeros(pred_score.shape), m-dev_score)
    loss = dev_score[(1-target).nonzero()].sum()+cont_score[target.nonzero()].sum()
    return loss

def evaluate_metric(pred_0, pred_1, pred_c1, pred_c0):
    tau_pred = torch.cat([pred_c1, pred_1], dim=0) - torch.cat([pred_0, pred_c0], dim=0)
    tau_true = torch.ones(tau_pred.shape) * 0.1
    ePEHE = torch.sqrt(torch.mean(torch.square(tau_pred-tau_true)))
    eATE = torch.abs(torch.mean(tau_pred) - torch.mean(tau_true))
    return eATE, ePEHE

def main():
    edge_index = torch.LongTensor(np.load('Dataset/synthetic/edge.npy'))    # (num_edge, 2)
    bot_label = np.load('Dataset/synthetic/bot_label.npy')
    T = np.load('Dataset/synthetic/T_label.npy')
    outcome = np.load('Dataset/synthetic/y.npy')

    N = len(outcome)  # num of nodes
    x = degree(edge_index[:, 0])  # user node degree as feature
    # 随机选80%训练
    node_idx = [i for i in range(N)]
    random.shuffle(node_idx)
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[node_idx[:int(N * 0.8)]] = 1
    test_mask = ~train_mask

    target_var = torch.LongTensor(
        np.concatenate([bot_label[:, np.newaxis], outcome[:, np.newaxis], T[:, np.newaxis]], axis=-1))  # (num_nodes, 3)

    botData = Data(x=x.unsqueeze(-1), edge_index=edge_index.t().contiguous(), y=target_var, train_mask=train_mask,
                   test_mask=test_mask)

    model1 = MaskEncoder(in_dim=1, h_dim=16, out_dim=16)
    model2 = botDetect(in_dim=1, h_dim=16, out_dim=1)
    model3 = impactDetect(in_dim=1, h_dim=16, out_dim=1)
    optimizer = torch.optim.Adam([{'params': model1.parameters(), 'lr': 0.001},
                                  {'params': model2.parameters(), 'lr': 0.001},
                                  {'params': model3.parameters(), 'lr': 0.001}])
    # for counterfactual edge generation
    edge_pool = set(map(tuple, np.array([[i, j] for i in range(N) for j in range(i, N)])))
    treat_label_train = botData.y[:, 2][botData.train_mask]
    treat_idx, control_idx = torch.where(treat_label_train==1)[0], torch.where(treat_label_train==-1)[0]
    treat_label_test = botData.y[:, 2][botData.test_mask]
    treat_idx_ts, control_idx_ts = torch.where(treat_label_test==1)[0], torch.where(treat_label_test==-1)[0]

    # train
    for epoch in range(300):
        model1.train()
        model2.train()
        model3.train()
        optimizer.zero_grad()

        homo_edge_index, hetero_edge_index = model1(botData.x, botData.edge_index)

        fake_env_graph = generate_counterfactual_edge(edge_pool, var_edge_index=hetero_edge_index,
                                                      inv_edge_index=homo_edge_index) # for bot detection
        fake_fact_graph = generate_counterfactual_edge(edge_pool, var_edge_index=homo_edge_index,
                                                       inv_edge_index=hetero_edge_index) # for effect estimation

        botData_fake_env = Data(x=x.unsqueeze(-1), edge_index=fake_env_graph.contiguous(), y=target_var, train_mask=train_mask,
                       test_mask=test_mask)
        botData_fake_fact = Data(x=x.unsqueeze(-1), edge_index=fake_fact_graph.contiguous(), y=target_var, train_mask=train_mask,
                       test_mask=test_mask)

        # detect bot
        out_b = model2(botData.x, botData.edge_index)   # out_b: (num_node, 1)
        out_b_fake_fact = model2(botData_fake_env.x, botData_fake_env.edge_index)   # out_b_fake: (num_node, 1)
        target_b = botData.y[:,0][botData.train_mask]   # target_b: (num_train_node)
        target_b_fake = botData_fake_env.y[:,0][botData.train_mask] # target_b_fake: (num_train_node)

        # 最大化所有环境下bot识别的准确度
        loss_b = contrastive_loss(target_b, out_b[botData.train_mask])
        loss_b_fake = contrastive_loss(target_b_fake, out_b_fake_fact[botData.train_mask])

        # assess bot
        # out_y1, out_yc0: (num_treat_train), out_y0, out_yc1: (num_control_train), *_prob: (num_nodes)
        out_y1, out_yc0, out_y0, out_yc1, fact_prob, fact_prob_f, treat_prob = model3(botData.x, botData.edge_index,
                                              botData_fake_fact.x, botData_fake_fact.edge_index, treat_idx, control_idx, train_mask)
        # out_y, target_y: (num_treat+control_train)
        out_y = torch.cat([out_y1, out_y0], dim=-1)
        target_y = torch.cat([botData.y[:, 1][botData.train_mask][treat_idx], botData.y[:, 1][botData.train_mask][control_idx]])
        # target_judgetreat: (num_treat+control_train)
        target_judgetreat = torch.cat([torch.ones(len(treat_prob[botData.train_mask][treat_idx])),torch.zeros(len(treat_prob[botData.train_mask][control_idx]))], dim=-1)
        out_judgetreat = torch.cat([treat_prob[botData.train_mask][treat_idx], treat_prob[botData.train_mask][control_idx]], dim=-1)
        # target_judgefact, out_judgefact: (2*num_train)
        target_judgefact = torch.cat([torch.ones(len(fact_prob[botData.train_mask])),torch.zeros(len(fact_prob_f[botData.train_mask]))], dim=-1)
        out_judgefact = torch.cat([fact_prob[botData.train_mask], fact_prob_f[botData.train_mask]], dim=-1)

        loss_y = contrastive_loss(target_y, out_y)
        loss_judgefact = contrastive_loss(target_judgefact, out_judgefact)
        loss_judgetreat = contrastive_loss(target_judgetreat, out_judgetreat)

        print(loss_b.item(), loss_b_fake.item(), loss_y.item(), loss_judgefact.item(), loss_judgetreat.item())
        loss = loss_b + loss_b_fake + loss_y*10 + loss_judgefact*0.01 + loss_judgetreat*0.05
        loss.backward()
        optimizer.step()

        if epoch%10 == 0:
            model1.eval()
            model2.eval()
            model3.eval()
            out_b = model2(botData.x, botData.edge_index)
            out_y1, out_yc0, out_y0, out_yc1, fact_prob, fact_prob_f, treat_prob = model3(botData.x, botData.edge_index,
                                                                        botData_fake_fact.x, botData_fake_fact.edge_index, treat_idx_ts, control_idx_ts, test_mask)

            target_b = botData.y[:, 0][botData.test_mask]
            # bot detection result
            threshold = torch.quantile(out_b, 0.96, dim=None, keepdim=False, interpolation='higher')
            pred_b = out_b.clone()
            pred_b[torch.where(out_b < threshold)] = 0
            pred_b[torch.where(out_b >= threshold)] = 1
            pred_label_b = pred_b[botData.test_mask]
            report_b = classification_report(target_b, pred_label_b.detach().numpy())


            # treatment effect prediction result
            eATE_test, ePEHE_test = evaluate_metric(out_y0, out_y1, out_yc1, out_yc0)
            print("Epoch " + str(epoch) + " bot-detect report:", report_b)
            print('eATE: {:.4f}'.format(eATE_test.detach().numpy()),
                  'ePEHE: {:.4f}'.format(ePEHE_test.detach().numpy()))


    model1.eval()
    model2.eval()
    model3.eval()

    homo_edge_index, hetero_edge_index = model1(botData.x, botData.edge_index)
    fake_fact_graph = generate_counterfactual_edge(edge_pool, var_edge_index=homo_edge_index,
                                                   inv_edge_index=hetero_edge_index)  # for effect estimation
    botData_fake_fact = Data(x=x.unsqueeze(-1), edge_index=fake_fact_graph.contiguous(), y=target_var,
                             train_mask=train_mask,
                             test_mask=test_mask)

    out_b = model2(botData.x, botData.edge_index)
    out_y1, out_yc0, out_y0, out_yc1, fact_prob, fact_prob_f, treat_prob = model3(botData.x, botData.edge_index,
                                                                                  botData_fake_fact.x,
                                                                                  botData_fake_fact.edge_index,
                                                                                  treat_idx_ts, control_idx_ts, test_mask)

    target_b = botData.y[:, 0][botData.test_mask]

    # bot detection
    threshold = torch.quantile(out_b, 0.96, dim=None, keepdim=False, interpolation='higher')
    pred_b = out_b.clone()
    pred_b[torch.where(out_b < threshold)] = 0
    pred_b[torch.where(out_b >= threshold)] = 1
    pred_label_b = pred_b[botData.test_mask]
    report_b = classification_report(target_b, pred_label_b.detach().numpy())
    # treatment effect prediction result
    eATE_test, ePEHE_test = evaluate_metric(out_y0, out_y1, out_yc1, out_yc0)

    print("Testing for bot-detect report:", report_b)
    print("ATE: {:.4f}, PEHE: {:.4f}".format(eATE_test.detach().numpy(), ePEHE_test.detach().numpy()))



if __name__ == "__main__":
    main()