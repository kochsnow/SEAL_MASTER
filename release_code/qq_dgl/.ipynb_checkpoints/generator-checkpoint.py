#!/usr/bin/env python
# coding=utf-8
import json 
import torch 
import glob 
import numpy as np 
from tqdm import tqdm 
from torch_geometric.data import Data 

from utils import graph_level_reader


class GraphDatasetGenerator(object):

    def __init__(self, path, device):
        self.path = path 
        self.device = device 
        self._enumerate_graphs()
        self._count_features_and_labels()
        self._create_target()
        self._create_dataset() 

    def _enumerate_graphs(self):
        """
        Listing the graph files and creating the respective label and feature maps.
        """
        graph_count = len(glob.glob(self.path + "*.json"))
        features = set() 
        labels = set() 
        self.graphs = []
        for index in tqdm(range(graph_count)):
            graph_file = self._concatenate_name(index)
            data = graph_level_reader(graph_file)
            self.graphs.append(data)
            labels = labels.union(set([data["labels"]]))
        self.label_map = {v: i for i, v in enumerate(labels)}

    def _count_features_and_labels(self):
        """
        Counting the number of unique feature and labels.:w
        """
        # the number of features for raw QQ-Group data is 10.
        self.number_of_features = 10 
        self.number_of_labels = len(self.label_map)

    def _transform_edges(self, raw_data):
        """
        Transforming an edge list from the data dictionary to a tensor :w
        :param raw_data: dictionary with edge list. 
        :return: edge list matrix.
        """
        edges = [[edge[0], edge[1]] for edge in raw_data["edges"]]
        return torch.t(torch.LongTensor(edges)).to(self.device)
    
    def _concatenate_name(self, index):
        """
        Creating a file name from an index. 
        """
        return self.path + str(index) + ".json"

    def _transform_features(self, raw_data):
        """
        Creating a feature matrix from the raw data. 
        :param raw_data: dictionary with features 
        :return: feature_matrix: FloatTensor of features
        """
        number_of_nodes = len(raw_data["features"])
        feature_matrix = [raw_data["features"][str(k)] for k in range(number_of_nodes)] 
        feature_matrix = np.array(feature_matrix) 
        _range = np.max(feature_matrix, axis=0) - np.min(feature_matrix, axis=0)
        feature_matrix = (feature_matrix - np.min(feature_matrix, axis=0)) / _range 
        feature_matrix = torch.FloatTensor(feature_matrix) 
        return feature_matrix.to(self.device) 

    def _data_transform(self, raw_data):
        """
        Creating a dictionary with the edge list matrix and the feature 
        """
        clean_data = dict() 
        clean_data["edges"] = self._transform_edges(raw_data) 
        clean_data["features"] = self._transform_features(raw_data) 
        return clean_data 

    def _create_target(self):
        """
        Creating a target vector.
        """
        self.target = [self.label_map[graph["labels"]] for graph in self.graphs]
        self.target = torch.FloatTensor(self.target).view(-1, 1).to(self.device)
    
    def _create_dataset(self):
        """
        Creating a list of dictionaries with edge list matrices and feature matrices.
        """
        graphs = [self._data_transform(graph) for graph in self.graphs] 
        data_list = []
        for k in range(len(self.target)):
            g = Data(x=graphs[k]["features"], edge_index=graphs[k]["edges"],
                    y=self.target[k])
            data_list.append(g) 
        
        # create dataset obj 
        # create dataloader obj 



