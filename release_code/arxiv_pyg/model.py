#local encoder
# 主要增加batch mode
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
class SAGE(torch.nn.Module):
    """
    SAGE layer class.
    """
    def __init__(self, args, number_of_features, num_of_labels):
        """
        Creating a SAGE layer.
        :param args: Arguments object.
        :param number_of_features: Number of node features.
        """
        super(SAGE, self).__init__()
        self.args = args
        self.number_of_features = number_of_features
        self.num_of_labels = num_of_labels
        self._setup()
#         self.training = True

    def _setup(self):
        """
        Setting up upstream and pooling layers.
        """
        self.graph_convolution_1 = GCNConv(self.number_of_features,
                                           self.args.first_gcn_dimensions)

        self.graph_convolution_2 = GCNConv(self.args.first_gcn_dimensions,
                                           self.args.second_gcn_dimensions)

        self.fully_connected_1 = torch.nn.Linear(self.args.second_gcn_dimensions,
                                                 self.args.first_dense_neurons*self.args.second_dense_neurons)

        self.fully_connected_2 = torch.nn.Linear(self.args.first_dense_neurons*self.args.second_dense_neurons,
                                                 self.num_of_labels)

#     def forward(self, data):
#         """
#         Making a forward pass with the graph level data.
#         :param data: Data feed dictionary.
#         :return graph_embedding: Graph level embedding.
#         :return penalty: Regularization loss.
#         """
#         edges = data["edges"]
#         features = data["features"]
#         node_features_1 = torch.nn.functional.relu(self.graph_convolution_1(features, edges))
#         node_features_2 = self.graph_convolution_2(node_features_1, edges)
#         abstract_features_1 = torch.tanh(self.fully_connected_1(node_features_2))
#         attention = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=0)
#         graph_embedding = torch.mm(torch.t(attention), node_features_2)
#         graph_embedding = graph_embedding.view(1, -1)
#         penalty = torch.mm(torch.t(attention), attention)-torch.eye(self.args.second_dense_neurons)
#         penalty = torch.sum(torch.norm(penalty, p=2, dim=1))
#         return graph_embedding, penalty
    #batch mode
    def forward(self, data):
        """
        Making a forward pass with the graph level data.
        :param data: Data feed dictionary.
        :return graph_embedding: Graph level embedding.
        :return penalty: Regularization loss.
        """
        edges = data.edge_index
        features = data.x
        batch = data.batch
        node_features_1 = torch.nn.functional.relu(self.graph_convolution_1(features, edges))
        node_features_2 = torch.nn.functional.relu(self.graph_convolution_2(node_features_1, edges))
        abstract_features_1 = self.fully_connected_1(node_features_2)
        abstract_features_1 = F.dropout(abstract_features_1, p=0.5, training=self.training)
        graph_embedding = global_mean_pool(abstract_features_1, batch)
        ic_predictions = self.fully_connected_2(graph_embedding)
#         attention = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=0)
#         graph_embedding = torch.mm(torch.t(attention), node_features_2)
#         graph_embedding = graph_embedding.view(1, -1)
#         penalty = torch.mm(torch.t(attention), attention)-torch.eye(self.args.second_dense_neurons)
#         penalty = torch.sum(torch.norm(penalty, p=2, dim=1))
        penalty = 0.0
        return graph_embedding, penalty, ic_predictions

#global encoder
class SuperMacroGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers = 3,
                 dropout = 0.5):
        super(SuperMacroGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout
#         self.training = True

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
#             pdb.set_trace()
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        #classifiy
        global_embeddings = x
        hc_predictions = self.convs[-1](global_embeddings, adj_t)
        return global_embeddings, hc_predictions
    
    
class MacroGCN(torch.nn.Module):
    """
    Macro Hierarchical GCN layer.
    """
    def __init__(self, args, number_of_features, number_of_labels):
        super(MacroGCN, self).__init__()
        """
        Creating a GCN layer for hierarchical learning.
        :param args: Arguments object.
        :param number_of_features: Graph level embedding size.
        :param number_of_labels: Number of node level labels.
        """
        self.args = args
        self.number_of_features = number_of_features
        self.number_of_labels = number_of_labels
        self._setup()

    def _setup(self):
        """
        We define two GCN layers, the downstram does classification.
        """
        self.graph_convolution_1 = GCNConv(self.number_of_features, self.args.macro_gcn_dimensions)
        self.graph_convolution_2 = GCNConv(self.args.macro_gcn_dimensions, self.number_of_labels)

    def forward(self, features, edges):
        """
        Making a forward pass.
        :param features: Node level embedding.
        :param egdes: Edge matrix of macro-model.
        :return predictions: Predictions for nodes.
        """
        node_features_1 = torch.nn.functional.relu(self.graph_convolution_1(features, edges))
        node_features_2 = self.graph_convolution_2(node_features_1, edges)
        predictions = torch.nn.functional.log_softmax(node_features_2, dim=1)
        return predictions
        
class SEAL(torch.nn.Module):
    """
    SEAL-CI model layer.
    """
    def __init__(self, args, number_of_features, number_of_labels):
        super(SEAL, self).__init__()
        """
        Creating a SEAl-CI layer.
        :param args: Arguments object.
        :param number_of_features: Number of features per graph.
        :param number_of_labels: Number of node level labels.
        """
        self.args = args
        self.number_of_features = number_of_features
        self.number_of_labels = number_of_labels
        self._setup()

    def _setup(self):
        """
        Creating a two stage model/
        """
        self.graph_level_model = SAGE(self.args, self.number_of_features, self.number_of_labels)
#         self.hierarchical_model = MacroGCN(self.args,
#                                            self.args.second_gcn_dimensions*self.args.second_dense_neurons,
#                                            self.number_of_labels)
        self.hierarchical_model = SuperMacroGCN(self.args.first_dense_neurons*self.args.second_dense_neurons,
                                                128,
                                           self.number_of_labels, 3)
        self.graph_level_model.to(device)
        self.hierarchical_model.to(device)

    def forward(self, graphs, macro_edges):
        """
        Making a forward pass.
        :param graphs: Graph data instance.
        :param macro_edges: Macro edge list matrix.
        :return predictions: Predicted scores.
        :return penalties: Average penalty on graph representations.
        """
        embeddings = []
        ic_predictions = []
        penalties = 0
        for batch_graph in graphs:
#             graph_embedding, penalty, ic_predictions
            embedding, penalty, ic_prediction = self.graph_level_model(batch_graph)
            embeddings.append(embedding)
#             pdb.set_trace()
            ic_predictions.append(ic_prediction)
            penalties = penalties + penalty
#         pdb.set_trace()
        loc_embeddings = torch.cat(tuple(embeddings))
        ic_predictions = torch.cat(tuple(ic_predictions))
        penalties = penalties/len(graphs)
        global_embeddings, hc_predictions = self.hierarchical_model(loc_embeddings, macro_edges)
        return hc_predictions, ic_predictions, loc_embeddings, global_embeddings, penalties