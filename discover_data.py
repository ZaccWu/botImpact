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
    human_degree, bot_degree, human_homo, bot_homo = [], [], [], []
    for i in range(len(bot_label)):
        followers_of_i = edge_idx[1, :][i == edge_idx[0, :]]    # 0-focal, 1-friend
        num_friends = len(followers_of_i)
        if num_friends == 0:
            continue
        same_propor = stance_label[followers_of_i].eq(stance_label[i]).sum() / len(followers_of_i)
        if bot_label[i] == 1:
            bot_degree.append(len(followers_of_i))
            bot_homo.append(same_propor.item())
        else:
            human_degree.append(len(followers_of_i))
            human_homo.append(same_propor.item())
    print("Analysis of "+Str)
    print("Num of friends: ", np.mean(bot_degree), np.mean(human_degree))
    print("Friends' homo:", np.mean(bot_homo), np.mean(human_homo))

check_homo(edge_index[:,(0==edge_type)], "user's followers")
check_homo(edge_index[:,(1==edge_type)], "user's following")
check_homo(edge_index[:,(2==edge_type)], "user's mention")
check_homo(edge_index[:,(3==edge_type)], "user's replying")
check_homo(edge_index[:,(4==edge_type)], "user's quoting")
