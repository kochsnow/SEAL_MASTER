#ready data mode
import json
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from texttable import Texttable
import pdb

import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
class ArxivDataset(InMemoryDataset):
    def __init__(self, root, graphs, transform=None, pre_transform=None):
        super(ArxivDataset, self).__init__(root, transform, pre_transform)
        self.graphs = graphs
        self.data, self.slices = self.process()#torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        download_url(url, self.raw_dir)


    def process(self):
        # Read data into huge `Data` list.
        data_list = self.graphs

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        return data, slices
        #torch.save((data, slices), self.processed_paths[0])
    def _download(self):
        return

    def _process(self):
        return
    
def hierarchical_graph_reader(path):
    """
    Reading the macro-level graph from disk.
    :param path: Path to the edge list.
    :return graph: Hierarchical graph as a NetworkX object.
    """
    edges = pd.read_csv(path, delimiter = "\t", usecols = [1, 2]).values.tolist()
#     for k in range(680):
#         edges.append([k, k])
    graph = nx.from_edgelist(edges)
    return graph

def graph_level_reader(path):
    """
    Reading a single graph from disk.
    :param path: Path to the JSON file.
    :return data: Dictionary of data.
    """
    data = json.load(open(path))
    return data

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

class GraphDatasetGenerator(object):
    """
    Creating an in memory version of the graphs.
    :param path: Folder with json files.
    """
    def __init__(self, path):
        self.path = path
        self._enumerate_graphs()
        self._count_features_and_labels()
        self._create_target()
        self._create_dataset()
        

    def _enumerate_graphs(self):
        """
        Listing the graph files and creating the respective label and feature maps.
        """
        graph_count = len(glob.glob(self.path + "*.json"))
        labels = set()
        features = set()
        self.graphs = []
        for index in tqdm(range(graph_count)):
            graph_file = self._concatenate_name(index)
            data = graph_level_reader(graph_file)
            self.graphs.append(data)
            labels = labels.union(set([data["labels"]]))
#             features = features.union(set([val for k, v in data["features"].items() for val in v]))
        self.label_map = {v: i for i, v in enumerate(labels)}
#         self.feature_map = {v: i for i, v in enumerate(features)}

    def _count_features_and_labels(self):
        """
        Counting the number of unique features and labels.
        """
        self.number_of_features = 768#len(self.feature_map)
        self.number_of_labels = len(self.label_map)

    def _transform_edges(self, raw_data):
        """
        Transforming an edge list from the data dictionary to a tensor.
        :param raw_data: Dictionary with edge list.
        :return : Edge list matrix.
        """
        edges = [[edge[0], edge[1]] for edge in raw_data["edges"]]
        #edges = edges + [[edge[1], edge[0]] for edge in raw_data["edges"]]
        return torch.t(torch.LongTensor(edges)).to(device)

    def _concatenate_name(self, index):
        """
        Creating a file name from an index.
        :param index: Graph index.
        :return : File name.
        """
        return self.path + str(index) + ".json"

    def _transform_features(self, raw_data):
        """
        Creating a feature matrix from the raw data.
        :param raw_data: Dictionary with features.
        :return feature_matrix: FloatTensor of features.
        """
        number_of_nodes = len(raw_data["features"])
#         feature_matrix = np.zeros((number_of_nodes, self.number_of_features))
#         index_1 = [int(n) for n, feats in raw_data["features"].items() for f in feats]
#         index_2 = [int(self.feature_map[f]) for n, feats in raw_data["features"].items() for f in feats]
#         feature_matrix[index_1, index_2] = 1.0
        feature_matrix = [raw_data["features"][str(k)] for k in range(number_of_nodes)]
        feature_matrix = torch.FloatTensor(np.array(feature_matrix))
        return feature_matrix.to(device)

    def _data_transform(self, raw_data):
        """
        Creating a dictionary with the edge list matrix and the features matrix.
        """
        clean_data = dict()
        clean_data["edges"] = self._transform_edges(raw_data)
        clean_data["features"] = self._transform_features(raw_data)
#         clean_data["labels"] = self.label_map(raw_data["labels"])
#         pdb.set_trace()
        return clean_data

    def _create_target(self):
        """
        Creating a target vector.
        """
        
        self.target = [self.label_map[graph["labels"]] for graph in self.graphs]
        self.target = torch.LongTensor(self.target).view(-1, 1).to(device)

    def _create_dataset(self):
        """
        Creating a list of dictionaries with edge list matrices and feature matrices.
        """
        graphs = [self._data_transform(graph) for graph in self.graphs]
        data_list = []
        for k in range(len(self.target)):
            g = Data(x = graphs[k]["features"], edge_index = graphs[k]["edges"], y = self.target[k])
            data_list.append(g)
        arxiv_dataset = ArxivDataset("root", data_list)
        self.graphs_loader = DataLoader(arxiv_dataset, batch_size=256, shuffle=False)
