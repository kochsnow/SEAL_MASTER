#!/usr/bin/env python
# coding=utf-8
import torch 
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv 


class SAGE(torch.nn.Module):
    """
    SAGE layer class.
    """
    def __init__(self, args, number_of_features, number_of_labels):
        super(SAGE, self).__init__()
        self.args = args 
        self.number_of_features = number_of_features 
        self.number_of_labels = number_of_labels 

        self._setup() 
        
    def _setup(self):
        self.gc1 = GCNConv(self.number_of_features, 
                          self.args.first_gcn_dimensions)
        self.gc2 = GCNConv(self.args.first_gcn_dimensions, 
                          self.args.second_gcn_dimensions)
        self.fc1 = torch.nn.Linear(self.args.second_gcn_dimensions, 
                                  self.args.first_dense_neurons * self.args.second_dense_neurons)
        self.fc2 = torch.nn.Linear(self.args.first_dense_dimensions * self.args.second_dense_neurons, 
                                  self.number_of_labels)

    def forward(self, data):
        edges = data.edge_index 
        features = data.x 
        batch = data.batch 
        node_features_1 = F.relu(self.gc1(features, edges))
        node_features_2 = F.relu(self.gc2(node_features_1, edges))
        abstract_features_1 = self.fc1(node_features_2)
        abstract_features_1 = F.dropout(abstract_features_1, p=0.5, training=self.training)
        graph_embedding = global_mean_pool(abstract_features_1, batch)
        ic_predictions = self.fc2(graph_embedding)
        penalty = 0.0 
        return graph_embedding, penalty, ic_predictions 


# global encoder 
class SuperMacroGCN(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, 
                 dropout=0.5):
        super(SuperMacroGCN, self).__init__() 

        self.convs = torch.nn.ModuleList() 
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList() 
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True)
                )
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout 

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters() 

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # classify 
        global_embeddings = x 
        hc_predictions = self.convs[-1](global_embeddings, adj_t)
        return global_embeddings, hc_predictions 


class MacroGCN(torch.nn.Module):
    
    def __init__(self, args, number_of_features, number_of_labels):
        super(MacroGCN, self).__init__() 
        self.args = args 
        self.number_of_features = number_of_features 
        self.number_of_labels = number_of_labels 
        self._setup() 

    def _setup(self):
        self.gc1 = GCNConv(self.number_of_features, self.macro_gcn_dimensions)
        self.gc2 = GCNConv(self.macro_gcn_dimensions, self.number_of_labels)

    def forward(self, features, edges):
        node_features_1 = F.relu(self.gc1(features, edges))
        node_features_2 = self.gc2(node_features_1, edges)
        predictions = F.log_softmax(node_features_2, dim=1)
        return predictions 
