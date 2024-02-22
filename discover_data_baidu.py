import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

user = pd.read_json('user.json')



