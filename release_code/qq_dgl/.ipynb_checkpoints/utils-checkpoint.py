#!/usr/bin/env python
# coding=utf-8
import json 
import pandas as pd 
import networkx as nx 
from texttable import Texttable 

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", "").capitalize(), args[k]] for k in keys])
    print(t.draw())


def hierarchical_graph_reader(path):
    edges = pd.read_csv(path, delimiter="\t", usecols=[1,2]).values.tolist()
    graph = nx.from_edgelist(edges)
    return graph 


def adjust_learning_rate(optimizer, epoch, init_lr):
    lr = init_lr - (0.0001 * (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr 


def graph_level_reader(path):
    data = json.load(open(path))
    return data 
