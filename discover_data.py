import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''
0: 'followers'
1: 'friends'
2: 'mention'
3: 'reply'
4: 'quoted'
5: 'url'
6: 'hashtag'
'''

edge_index = torch.load("Dataset/MGTAB/edge_index.pt")
edge_index = torch.tensor(edge_index, dtype = torch.int64)  # (2, 1700108)
# follower, friend, mention, reply, quote, URL, hashtag
edge_type = torch.load("Dataset/MGTAB/edge_type.pt")        # (1700108)
# weight: direction or edge weight
edge_weight = torch.load("Dataset/MGTAB/edge_weight.pt")    # (1700108)
# neutral: 3776, against: 3637, support: 2786
stance_label = torch.load("Dataset/MGTAB/labels_stance.pt") # (10199) 0-neutral, 1-against, 2-support
# human: 7451, bot: 2748
bot_label = torch.load("Dataset/MGTAB/labels_bot.pt")       # (10199) 0-human, 1-bot

# dim 0-19: profile features, dim 20-787: tweet features
features = torch.load("Dataset/MGTAB/features.pt")          # (10199, 788)
features = features.to(torch.float32)

def check_homo(edge_idx, Str):
    node_degree, node_homo = [], []
    for i in range(len(bot_label)):
        followers_of_i = edge_idx[1, :][i == edge_idx[0, :]]    # 0-focal, 1-friend
        num_friends = len(followers_of_i)
        if num_friends == 0:
            node_degree.append(0)
            node_homo.append(None)
            continue
        same_propor = stance_label[followers_of_i].eq(stance_label[i]).sum() / len(followers_of_i)
        node_degree.append(num_friends)
        node_homo.append(same_propor.item())
    return node_degree, node_homo

edge_index_0, edge_index_1 = edge_index[:,(0==edge_type)], edge_index[:,(1==edge_type)]
edge_index_2, edge_index_3, edge_index_4 = edge_index[:,(2==edge_type)], edge_index[:,(3==edge_type)], edge_index[:,(4==edge_type)]

nd1, nh1 = check_homo(edge_index_0, "user's followers")    # degree: 2.09, 40.58, homo: 0.606, 0.728
nd2, nh2 = check_homo(edge_index_1, "user's following")    # degree: 14.19, 50.14, homo: 0.471, 0.617
nd3, nh3 = check_homo(edge_index_2, "user's mention")      # degree: 5.71, 13.26, homo: 0.575, 0.729
nd4, nh4 = check_homo(torch.concat([edge_index_2[1,:].unsqueeze(0), edge_index_2[0,:].unsqueeze(0)], dim=0),
           "user's mentioner")                  # degree: 0.17, 15.31, homo: 0.926, 0.888
nd5, nh5 = check_homo(edge_index_3, "user's replying")     # degree: 2.12, 29.20, homo: 0.880, 0.904
nd6, nh6 = check_homo(torch.concat([edge_index_3[1,:].unsqueeze(0), edge_index_3[0,:].unsqueeze(0)], dim=0),
           "user's replyer")                    # degree: 1.63, 29.39, homo: 0.968, 0.942
nd7, nh7 = check_homo(edge_index_4, "user's quoting")      # degree: 1.13, 10.00, homo: 0.719, 0.897
nd8, nh8 = check_homo(torch.concat([edge_index_4[1,:].unsqueeze(0), edge_index_4[0,:].unsqueeze(0)], dim=0),
           "user's quoter")                     # degree: 0.09, 10.39, homo: 1, 0.958

bot_data = pd.DataFrame({
    'bot': bot_label.numpy(),
    'stance': stance_label.numpy(),
    'followers_deg': nd1,
    'followers_hom': nh1,
    'followings_deg': nd2,
    'followings_hom': nh2,
    'mentionings_deg': nd3,
    'mentionings_hom': nh3,
    'mentioners_deg': nd4,
    'mentioners_hom': nh4,
    'replyings_deg': nd5,
    'replyings_hom': nh5,
    'replyers_deg': nd6,
    'replyers_hom': nh6,
    'quotings_deg': nd7,
    'quotings_hom': nh7,
    'quoters_deg': nd8,
    'quoters_hom': nh8,
})
bot_data.to_csv('bot_data.csv',index=False)