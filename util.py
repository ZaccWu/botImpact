import torch
import numpy as np

EPS = 1e-15

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
        if prop_label[friend].sum()>0 and bot_label[friend].sum()==0: # the node is in the control group in Gcf
            treat_idx_ok.append(id)

    for id in control_idx.tolist():
        if id not in friend_dict:
            continue
        friend = friend_dict[id]  # id's friend
        if bot_label[friend].sum()>0 and prop_label[friend].sum()==0: # the node is in the treated group in Gcf
            control_idx_ok.append(id)

    return torch.LongTensor(treat_idx_ok), torch.LongTensor(control_idx_ok)

def pairwise_similarity(matrix1, matrix2):
    norm1 = torch.norm(matrix1, p=2, dim=1, keepdim=True)
    norm2 = torch.norm(matrix2, p=2, dim=1, keepdim=True)
    normalized_matrix1 = matrix1 / (norm1 + EPS)
    normalized_matrix2 = matrix2 / (norm2 + EPS)

    # 计算pair-wise相似度
    # similarity_matrix = torch.mm(normalized_matrix1, normalized_matrix2.t())
    similarity_matrix = torch.cdist(normalized_matrix1, normalized_matrix2, p=2)
    return similarity_matrix


def similarity_check(matrix1, matrix2):
    norm1 = torch.norm(matrix1, p=2, dim=1, keepdim=True)
    norm2 = torch.norm(matrix2, p=2, dim=1, keepdim=True)
    normalized_matrix1 = matrix1 / norm1
    normalized_matrix2 = matrix2 / norm2
    dist = torch.cdist(normalized_matrix1, normalized_matrix2, p=2).mean().item()
    cosim = torch.mm(normalized_matrix1, normalized_matrix2.t()).mean().item()
    print('Sim_dist: {:.4f}, Sim_cos: {:.4f}'.format(dist * 1000, cosim))
