import dgl
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import pickle as pkl
import networkx as nx
import random
import numpy as np
from dgl.data.utils import load_graphs, save_graphs
from argparse import ArgumentParser 


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--feat_type", type=str, default="avg", help="way to generate group feature")
    args = parser.parse_args()
    return args 


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



args = parse_args()
if args.feat_type == "gc":
    print("Generating group feature using graph classification ...")
elif args.feat_type == "avg":
    print("Generating group feature using naive average ...")

## creat qq dataset
with open("labels.pk", "rb") as f:
    labels = pkl.load(f)
with open("ind.group.graph", "rb") as f:
    graph = pkl.load(f)
    nxgraph = nx.from_dict_of_lists(graph)
if args.feat_type == "avg":
    features = np.load("norm_x.npy")
elif args.feat_type == "gc":
    features = np.load("norm_x_gc.npy")

g = dgl.from_networkx(nxgraph)
g.ndata['feat'] = torch.tensor(features).double()
# print(torch.argmax(torch.tensor(labels),1))
# print(torch.argmax(torch.tensor(labels),0))

g.ndata['label'] = torch.argmax(torch.tensor(labels),1)

random.seed(123)
test_id = list(range(len(labels)))[-10000:]
tmp_id = list(range(len(labels)))[:-10000]
random.shuffle(tmp_id)
train_id, val_id = tmp_id[:1000], tmp_id[1000:]
g.ndata['train_mask'] = torch.tensor(sample_mask(train_id, len(labels)))
g.ndata['val_mask'] = torch.tensor(sample_mask(val_id, len(labels)))
g.ndata['test_mask'] = torch.tensor(sample_mask(test_id, len(labels)))

if args.feat_type == "avg":
    save_graphs('qq_graph', [g])
elif args.feat_type == "gc":
    save_graphs("qq_graph_gc", [g])
