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

        _, topk_homo = torch.topk(value, int(len(value)*0.8), largest=True)
        _, topk_hetero = torch.topk(value, int(len(value)*0.2), largest=False)
        return edge_index[:,topk_homo], edge_index[:,topk_hetero]

class impactDetect(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=1, heads=1):
        super(impactDetect, self).__init__()
        self.convZ1 = GATConv(in_dim, h_dim, heads)
        self.convZ2 = GATConv(h_dim*heads, h_dim, heads)
        self.yNet1 = torch.nn.Sequential(torch.nn.Linear(h_dim*heads, out_dim), torch.nn.LeakyReLU())
        self.yNet0 = torch.nn.Sequential(torch.nn.Linear(h_dim*heads, out_dim), torch.nn.LeakyReLU())
        #self.balanceNet = torch.nn.Sequential(torch.nn.Linear(h_dim, 2), torch.nn.LeakyReLU())
        #self.propenNet = torch.nn.Sequential(torch.nn.Linear(h_dim, 2), torch.nn.LeakyReLU())
        self.balanceNet = torch.nn.Sequential(torch.nn.Linear(h_dim*heads, 2))
        self.propenNet = torch.nn.Sequential(torch.nn.Linear(h_dim*heads, h_dim), torch.nn.LeakyReLU(), torch.nn.Linear(h_dim, 2), torch.nn.LeakyReLU())
        self.grl = GRL_Layer()

    def forward(self, x, edge_index, fake_x, fake_edge_index, treat_idx, control_idx):
        # generate node embedding
        xZ1, xfZ1 = F.relu(self.convZ1(x, edge_index)), F.relu(self.convZ1(fake_x, fake_edge_index))
        xZ1, xfZ1 = F.dropout(xZ1, p=0.5, training=self.training), F.dropout(xfZ1, p=0.5, training=self.training)
        xZ2, xfZ2 = self.convZ2(xZ1, edge_index), self.convZ2(xfZ1, fake_edge_index)    # xZ2/xfZ2: (num_nodes, h_dim)

        # predict outcome
        y1, yc0 = self.yNet1(xZ2[treat_idx]), self.yNet0(xfZ2[treat_idx])       # yc0：如果有bot的node周围没bot会怎样
        y0, yc1 = self.yNet0(xZ2[control_idx]), self.yNet1(xfZ2[control_idx])   # yc1: 如果没bot的node周围有bot会怎么样
        # judge the node is from factual & counterfactual graph
        fprob, fprob_f = self.balanceNet(self.grl(xZ2)), self.balanceNet(self.grl(xfZ2))
        # judge the node is treated or controled
        tprob = self.propenNet(xZ2)
        return y1.squeeze(-1), yc0.squeeze(-1), y0.squeeze(-1), yc1.squeeze(-1), fprob.squeeze(-1), fprob_f.squeeze(-1), tprob.squeeze(-1)



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
    print("pred treat:", torch.mean(pred_1), torch.mean(pred_c1))
    print("pred control:", torch.mean(pred_0), torch.mean(pred_c0))
    tau_true = torch.ones(tau_pred.shape) * -1
    ePEHE = torch.sqrt(torch.mean(torch.square(tau_pred-tau_true)))
    eATE = torch.abs(torch.mean(tau_pred) - torch.mean(tau_true))
    return eATE, ePEHE

def transfer_pred(out, threshold):
    pred = out.clone()
    pred[torch.where(out < threshold)] = 0
    pred[torch.where(out >= threshold)] = 1
    return pred

def load_syn_data(dt):
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

def load_emp_data(stance):
    edge_index = torch.load("Dataset/MGTAB/edge_index.pt")
    edge_index = torch.tensor(edge_index, dtype=torch.int64)  # (2, 1700108)
    # follower, friend, mention, reply, quote, URL, hashtag
    edge_type = torch.load("Dataset/MGTAB/edge_type.pt")  # (1700108)
    # weight: direction or edge weight
    edge_weight = torch.load("Dataset/MGTAB/edge_weight.pt")  # (1700108)
    # neutral: 3776, against: 3637, support: 2786
    stance_label = torch.load("Dataset/MGTAB/labels_stance.pt")  # (10199) 0-neutral, 1-against, 2-support
    # human: 7451, bot: 2748
    bot_label = torch.load("Dataset/MGTAB/labels_bot.pt")  # (10199) 0-human, 1-bot
    # dim 0-19: profile features, dim 20-787: tweet features
    features = torch.load("Dataset/MGTAB/features.pt")  # (10199, 788)
    features = features.to(torch.float32)

    edge_index_0, edge_index_1 = edge_index[:, (0 == edge_type)], edge_index[:, (1 == edge_type)]
    edge_index = np.concatenate([edge_index_0, edge_index_1], axis=1)
    N = len(stance_label)
    x = features[:,:20]
    T, prop_label = np.zeros(N), np.zeros(N)

    print("constructing friend dict:...")
    friend_dict = {}
    for i in range(N):
        u, v = edge_index[0][i], edge_index[1][i]
        friend_dict.setdefault(u, []).append(v)
    id_all = set(np.array([i for i in range(N)]))
    id_s = id_all - set(np.nonzero(stance_label-stance)[0])
    prop_label[list(map(int, id_s))] = 1

    print("constructing treat/control:...")
    treat_id = []
    control_id = []
    for i in range(N):
        if i not in friend_dict.keys():
            continue
        for j in friend_dict[i]:
            if stance_label[j] == stance:
                if bot_label[j] == 1:
                    treat_id.append(i)
                elif bot_label[j] == 0:
                    control_id.append(i)
    treat_id, control_id = set(treat_id), set(control_id)
    intersec = set(treat_id)&set(control_id)
    treat_id = list(map(int, treat_id-intersec))
    control_id = list(map(int, control_id-intersec))
    print("len of treat/control/intersect:", len(treat_id), len(control_id), len(intersec))
    T[treat_id] = -1
    T[control_id] = -1
    target_var = torch.tensor(
        np.concatenate([bot_label[:, np.newaxis], stance_label[:, np.newaxis], T[:, np.newaxis]], axis=-1))  # (num_nodes, 3)
    botData = Data(x=x, edge_index=torch.LongTensor(edge_index).contiguous(), y=target_var)
    return botData, N, prop_label, torch.LongTensor(treat_id), torch.LongTensor(control_id)

def main():
    # botData_train, N_train, prop_label_train = load_syn_data('train')
    # botData_test, N_test, prop_label_test = load_syn_data('train')
    botData_train, N_train, prop_label_train, treat_idx_train, control_idx_train = load_emp_data(stance=2)
    botData_test, N_test, prop_label_test, treat_idx_test, control_idx_test = load_emp_data(stance=2)
    # model1 = MaskEncoder(in_dim=1, h_dim=32, out_dim=32)
    # model3 = impactDetect(in_dim=1, h_dim=32, out_dim=1)
    model1 = MaskEncoder(in_dim=20, h_dim=32, out_dim=32)
    model3 = impactDetect(in_dim=20, h_dim=32, out_dim=1)
    optimizer = torch.optim.Adam([{'params': model1.parameters(), 'lr': 0.001},
                                  {'params': model3.parameters(), 'lr': 0.001}])
    # for counterfactual edge generation
    #edge_pool_train = set(map(tuple, np.array([[i, j] for i in range(N_train) for j in range(i, N_train)])))
    edge_pool_train = {(i,j) for i in range(N_train) for j in range(i, N_train)}
    #treat_idx_train, control_idx_train = torch.where(botData_train.y[:, 2]==-1)[0], torch.where(botData_train.y[:, 2]==1)[0]


    outcome_dict = {}
    # train
    for epoch in range(300):
        model1.train()
        model3.train()
        optimizer.zero_grad()
        homo_edge_index, hetero_edge_index = model1(botData_train.x, botData_train.edge_index)
        fake_fact_graph = generate_counterfactual_edge(edge_pool_train, var_edge_index=homo_edge_index,
                                                       inv_edge_index=hetero_edge_index) # for effect estimation
        botData_fake_fact = Data(x=botData_train.x, edge_index=fake_fact_graph.contiguous(), y=botData_train.y)
        print("original treat/control: ", treat_idx_train.shape, control_idx_train.shape)
        treat_idx_ok, control_idx_ok = match_node(fake_fact_graph, botData_train.y[:, 0], prop_label_train, treat_idx_train, control_idx_train)
        print("matched treat/control: ", treat_idx_ok.shape, control_idx_ok.shape)

        # assess bot
        # out_y1, out_yc0: (num_treat_train), out_y0, out_yc1: (num_control_train), *_prob: (num_nodes)
        ## TODO: 此处有疑问，感觉这里输入和target_y没有必要用treat_idx_ok，用treat_index_train就行
        out_y1, out_yc0, out_y0, out_yc1, fact_prob, fact_prob_f, treat_prob = model3(botData_train.x, botData_train.edge_index,
                                              botData_fake_fact.x, botData_fake_fact.edge_index, treat_idx_train, control_idx_train)
        out_y = torch.cat([out_y1, out_y0], dim=-1)
        target_y = torch.cat([botData_train.y[:, 1][treat_idx_train], botData_train.y[:, 1][control_idx_train]])

        # target_judgetreat: (num_treat+control_train) counterfactual图里面treat标签相反
        target_judgetreat = torch.cat([torch.ones(len(treat_prob[treat_idx_train])),torch.zeros(len(treat_prob[control_idx_train]))])
        out_judgetreat = torch.cat([treat_prob[treat_idx_train],treat_prob[control_idx_train]], dim=0)

        # target_judgefact, out_judgefact: (2*num_train)
        target_judgefact = torch.cat([torch.ones(len(fact_prob)),torch.zeros(len(fact_prob_f))], dim=-1)
        out_judgefact = torch.cat([fact_prob, fact_prob_f], dim=0)
        #print("pred treat compare:", torch.mean(treat_prob[treat_idx_ok]).item(), torch.mean(treat_prob[control_idx_ok]).item())
        loss_y = F.mse_loss(out_y.float(), target_y.float())
        #loss_judgefact = contrastive_loss(target_judgefact, out_judgefact)
        loss_judgefact = torch.nn.CrossEntropyLoss()(out_judgefact, target_judgefact.long())
        loss_judgetreat = torch.nn.CrossEntropyLoss()(out_judgetreat, target_judgetreat.long())
        #loss_judgetreat = contrastive_loss(target_judgetreat, out_judgetreat)

        print("{:.4f} {:.4f} {:.4f}".format(loss_y.item(),loss_judgefact.item(),loss_judgetreat.item()))
        loss = loss_y*par['ly'] + loss_judgefact*par['ljf'] + loss_judgetreat*par['ljt']
        loss.backward()
        optimizer.step()

        if epoch%10 == 0:
            model1.eval()
            model3.eval()
            # 验证/测试时无需新构造fake graph
            out_y1, out_yc0, out_y0, out_yc1, fact_prob, fact_prob_f, treat_prob = model3(botData_test.x, botData_test.edge_index,
                                                                        botData_fake_fact.x, botData_fake_fact.edge_index, treat_idx_test, control_idx_test)
            # for empirical data
            res_y1, res_y0, res_yc0, res_yc1 = out_y1.detach().numpy(), out_y0.detach().numpy(), out_yc0.detach().numpy(), out_yc1.detach().numpy()
            outcome_dict[str(epoch) + 'y1'] = np.pad(res_y1, (0, 500-len(res_y1)))
            outcome_dict[str(epoch) + 'y0'] = np.pad(res_y0, (0, 500-len(res_y0)))
            outcome_dict[str(epoch) + 'yc0'] = np.pad(res_yc0, (0, 500-len(res_yc0)))
            outcome_dict[str(epoch) + 'yc1'] = np.pad(res_yc1, (0, 500-len(res_yc1)))

            # treatment effect prediction result
            eATE_test, ePEHE_test = evaluate_metric(out_y0, out_y1, out_yc1, out_yc0)
            print("Epoch: " + str(epoch))
            print('eATE: {:.4f}'.format(eATE_test.detach().numpy()),
                  'ePEHE: {:.4f}'.format(ePEHE_test.detach().numpy()))

    # for empirical data
    res = pd.DataFrame(outcome_dict)
    res.to_csv('result_outcome.csv', index=False)



if __name__ == "__main__":
    type = 'random'
    stance = 2  # for empirical data
    gpu = 0
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    par = {'ly': 10,
           'ljf': 10,
           'ljt': 1}
    print(par)
    main()
