import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data



edge_index = torch.load("Dataset/MGTAB/edge_index.pt")
edge_index = torch.tensor(edge_index, dtype = torch.int64)  # (2, 1700108)
# follower, friend, mention, reply, quote, URL, hashtag
edge_type = torch.load("Dataset/MGTAB/edge_type.pt")        # (1700108)
# weight: direction or edge weight
edge_weight = torch.load("Dataset/MGTAB/edge_weight.pt")    # (1700108)
# neutral: 3776, against: 3637, support: 2786
stance_label = torch.load("Dataset/MGTAB/labels_stance.pt") # (10199)
# human: 7451, bot: 2748
bot_label = torch.load("Dataset/MGTAB/labels_bot.pt")       # (10199)

# dim 0-19: profile features, dim 20-787: tweet features
features = torch.load("Dataset/MGTAB/features.pt")          # (10199, 788)
features = features.to(torch.float32)

print(edge_type)
print(edge_weight)
print(stance_label)
print(bot_label)


