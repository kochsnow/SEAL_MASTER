#!/usr/bin/env python
# coding=utf-8
import torch 

from gnn import SAGE, SuperMacroGCN 


class SEAL(torch.nn.Module):
    """
    SEAL-CI model layer class. 
    """

    def __init__(self, args, number_of_features, number_of_labels, device):
        super(SEAL, self).__init__() 
        self.args = args 
        self.number_of_features = number_of_features 
        self.number_of_labels = number_of_labels 
        self.device = device 
        self._setup() 

    def _setup(self):
        self.graph_level_model = SAGE(self.args, self.number_of_features, self.number_of_labels)
        self.hierarchical_model = SuperMacroGCN(self.args.first_dense_neurons * self.args.second_dense_neurons, 8, self.number_of_labels, 3)
        self.graph_level_model.to(self.device)
        self.hierarchical_model.to(self.device)

    def forward(self, graphs, macro_edges):
        embeddings = []
        ic_predictions = []
        penalties = []
        for batch_graph in graphs:
            embedding, penalty, ic_prediction = self.graph_level_model(batch_graph)
            embeddings.append(embedding)
            ic_predictions.append(ic_prediction)
            penalties = penalties + penalty 

        loc_embeddings = torch.cat(tuple(embeddings))
        ic_predictions = torch.cat(tuple(ic_predictions))
        penalties = penalties / len(graphs)
        global_embeddings, hc_predictions = self.hierarchical_model(loc_embeddings, macro_edges)
        return hc_predictions, ic_predictions, loc_embeddings, global_embeddings, penalties 

