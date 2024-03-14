import torch
from torch_geometric.nn import GATConv, GCNConv
import torch.nn.functional as F

# DD
class DD(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=1):
        super(DD, self).__init__()
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


# CNE-
class CNE_Minus(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=1):
        super(CNE_Minus, self).__init__()
        self.convZ1 = GCNConv(in_dim, h_dim)
        self.convZ2 = GCNConv(h_dim, h_dim)
        self.yNet1 = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())
        self.yNet0 = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())
        self.balanceNet = torch.nn.Sequential(torch.nn.Linear(h_dim, 2))
        self.propenNet = torch.nn.Sequential(torch.nn.Linear(h_dim, h_dim), torch.nn.LeakyReLU(), torch.nn.Linear(h_dim, 2), torch.nn.LeakyReLU())
    def forward(self, x, edge_index, treat_idx, control_idx):
        # generate node embedding
        xZ1 = F.relu(self.convZ1(x, edge_index))
        xZ1 = F.dropout(xZ1, p=0.5, training=self.training)
        xZ2 = self.convZ2(xZ1, edge_index)    # xZ2: (num_nodes, h_dim)
        # predict outcome
        y1, yc0 = self.yNet1(xZ2[treat_idx]), self.yNet0(xZ2[treat_idx])       # yc0：如果有bot的node周围没bot会怎样
        y0, yc1 = self.yNet0(xZ2[control_idx]), self.yNet1(xZ2[control_idx])   # yc1: 如果没bot的node周围有bot会怎么样
        # judge the node is treated or controled
        tprob = self.propenNet(xZ2)
        return y1.squeeze(-1), yc0.squeeze(-1), y0.squeeze(-1), yc1.squeeze(-1), tprob.squeeze(-1), xZ2

# Ignite
class IgniteGenerator(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=1, heads=8):
        super(IgniteGenerator, self).__init__()
        self.convZ1 = GATConv(in_dim, h_dim, heads=heads)
        self.convZ2 = GATConv(h_dim*heads, h_dim, heads=heads)
        self.yNet1 = torch.nn.Sequential(torch.nn.Linear(h_dim*heads, out_dim), torch.nn.LeakyReLU())
        self.yNet0 = torch.nn.Sequential(torch.nn.Linear(h_dim*heads, out_dim), torch.nn.LeakyReLU())
    def forward(self, x, edge_index, treat_idx, control_idx):
        # generate node embedding
        xZ1 = F.relu(self.convZ1(x, edge_index))
        xZ1 = F.dropout(xZ1, p=0.5, training=self.training)
        xZ2 = self.convZ2(xZ1, edge_index)    # xZ2: (num_nodes, h_dim)
        # predict outcome
        y1, yc0 = self.yNet1(xZ2[treat_idx]), self.yNet0(xZ2[treat_idx])       # yc0：如果有bot的node周围没bot会怎样
        y0, yc1 = self.yNet0(xZ2[control_idx]), self.yNet1(xZ2[control_idx])   # yc1: 如果没bot的node周围有bot会怎么样
        return y1.squeeze(-1), yc0.squeeze(-1), y0.squeeze(-1), yc1.squeeze(-1), xZ2

class IgniteDiscriminator(torch.nn.Module):
    def __init__(self, h_dim, heads=8):
        super(IgniteDiscriminator, self).__init__()
        self.propenNet = torch.nn.Sequential(torch.nn.Linear(h_dim*heads, h_dim), torch.nn.LeakyReLU(),
                                             torch.nn.Linear(h_dim, 1), torch.nn.LeakyReLU())
    def forward(self, xZ2):
        # judge the node is treat/control
        tprob = self.propenNet(xZ2)
        return tprob.squeeze(-1)

# Gial
class GialGenerator(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=1):
        super(GialGenerator, self).__init__()
        self.convZ1 = GATConv(in_dim, h_dim)
        self.convZ2 = GATConv(h_dim, h_dim)
        self.yNet1 = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())
        self.yNet0 = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU())
    def forward(self, x, edge_index, fake_x, treat_idx, control_idx):
        # generate node embedding
        xZ1, xfZ1 = F.relu(self.convZ1(x, edge_index)), F.relu(self.convZ1(fake_x, edge_index))
        xZ1, xfZ1 = F.dropout(xZ1, p=0.5, training=self.training), F.dropout(xfZ1, p=0.5, training=self.training)
        xZ2, xfZ2 = self.convZ2(xZ1, edge_index), self.convZ2(xfZ1, edge_index)    # xZ2/xfZ2: (num_nodes, h_dim)
        # predict outcome
        y1, yc0 = self.yNet1(xZ2[treat_idx]), self.yNet0(xfZ2[treat_idx])       # yc0：如果有bot的node周围没bot会怎样
        y0, yc1 = self.yNet0(xZ2[control_idx]), self.yNet1(xfZ2[control_idx])   # yc1: 如果没bot的node周围有bot会怎么样
        return y1.squeeze(-1), yc0.squeeze(-1), y0.squeeze(-1), yc1.squeeze(-1), xZ2, xfZ2

class GialDiscriminator(torch.nn.Module):
    def __init__(self, h_dim):
        super(GialDiscriminator, self).__init__()
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