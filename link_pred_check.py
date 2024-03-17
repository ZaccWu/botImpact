import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import networkx as nx
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import GCNConv
EPS = 1e-15


class SimpleGCN(torch.nn.Module):
	def __init__(self,in_dim, z_dim, h_dim=32):
		super().__init__()
		self.z_dim = z_dim
		self.training = True
		self.conv1 = GCNConv(in_dim, h_dim, add_self_loops=True)
		self.conv2 = GCNConv(h_dim, self.z_dim, add_self_loops=True)
		self.ll_y = torch.nn.Linear(self.z_dim + 2, 1)
		self.act = torch.nn.LeakyReLU()
	def forward(self, x, edge_index):
		x_1 = self.conv1(x, edge_index)
		x_1 = x_1.relu()
		x_1 = F.dropout(x_1, p=0.5, training=self.training)
		emb_out = self.act(self.conv2(x_1, edge_index))
		return emb_out


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

def evaluate(z, pos_edge_index, neg_edge_index):
	pos_y = z.new_ones(pos_edge_index.size(1))
	neg_y = z.new_zeros(neg_edge_index.size(1))
	y = torch.cat([pos_y, neg_y], dim=0)
	decoder = InnerProductDecoder()
	pos_pred = decoder(z, pos_edge_index, sigmoid=True)
	neg_pred = decoder(z, neg_edge_index, sigmoid=True)
	pred = torch.cat([pos_pred, neg_pred], dim=0)
	y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
	return roc_auc_score(y, pred), average_precision_score(y, pred)



def train():

    # topic 1 pos
    bot_label = np.load('Dataset/twi22/t1/t1_pos_bot_label.npy')
    #treat_indicator = np.load('Dataset/twi22/t1/t1_pos_T_label.npy')
    outcome = np.load('Dataset/twi22/t1/t1_pos_y.npy')
    prop_label = np.load('Dataset/twi22/t1/t1_pos_prop_label.npy')
    edge_index = np.load('Dataset/twi22/t1/t1_pos_edge.npy')

    botData = Data(x=torch.FloatTensor(outcome).unsqueeze(-1), edge_index=torch.LongTensor(edge_index).t().contiguous())
    transform = RandomLinkSplit(is_undirected=False, num_val=0, num_test=0.3, neg_sampling_ratio=1.0)
    train_edge, _, test_edge = transform(botData)

    test_edge_select = []
    for i in range(len(test_edge.edge_index[0])):
        test_edge_select.append((test_edge.edge_index[0][i].item(), test_edge.edge_index[1][i].item()))

    G = nx.DiGraph()
    for e in range(len(edge_index)):
        G.add_edge(edge_index[e][0], edge_index[e][1])
    G = G.reverse() # for checking precessor

    human_ego_edge, bot_ego_edge = [], []
    for node in G.nodes:
        if prop_label[node] == 1:
            if bot_label[node] == 0:
                ego_graph = nx.ego_graph(G, node)
                ego_graph = ego_graph.reverse()
                human_ego_edge.extend(list(ego_graph.edges()))
            elif bot_label[node] == 1:
                ego_graph = nx.ego_graph(G, node)
                ego_graph = ego_graph.reverse()
                bot_ego_edge.extend(list(ego_graph.edges()))

    print(len(test_edge_select), len(human_ego_edge), len(bot_ego_edge))
    human_ego_test = set(test_edge_select)&set(human_ego_edge) # remove edge in the train set
    bot_ego_test = set(test_edge_select)&set(bot_ego_edge)  # remove edge in the train set
    print(len(human_ego_test), len(bot_ego_test))

    human_ego_edge, bot_ego_edge = [], []
    for (e0, e1) in human_ego_test:
        human_ego_edge.append([e0, e1])
    for (e0, e1) in bot_ego_test:
        bot_ego_edge.append([e0, e1])

    human_ego_edge, bot_ego_edge = torch.LongTensor(human_ego_edge).t(), torch.LongTensor(bot_ego_edge).t()
    print(train_edge.edge_index.shape, human_ego_edge.shape, bot_ego_edge.shape)


    model = SimpleGCN(in_dim=1, z_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        model.training = True
        optimizer.zero_grad()
        pos_edge, neg_edge = train_edge.edge_label_index[:, train_edge.edge_label.nonzero()].squeeze(-1), \
                             train_edge.edge_label_index[:, (1 - train_edge.edge_label).nonzero()].squeeze(-1)
        emb_out = model(botData.x, pos_edge)  # semi-supervised
        loss = recon_loss(emb_out, pos_edge, neg_edge)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            model.training = False

            embH, embB = model(botData.x, human_ego_edge), model(botData.x, bot_ego_edge)

            decoder = InnerProductDecoder()
            pos_pred_human, pos_pred_bot = decoder(embH, human_ego_edge, sigmoid=True), decoder(embB, bot_ego_edge, sigmoid=True)

            y_human = embH.new_ones(human_ego_edge.size(1))
            y_bot = embB.new_ones(human_ego_edge.size(1))

            print("Human: ", roc_auc_score(y_human, pos_pred_human.detach().numpy()), average_precision_score(y_human, pos_pred_human.detach().numpy()))
            print("Bot: ", roc_auc_score(y_bot, pos_pred_human.detach().numpy()), average_precision_score(y_bot, pos_pred_human.detach().numpy()))


if __name__ == "__main__":
    train()