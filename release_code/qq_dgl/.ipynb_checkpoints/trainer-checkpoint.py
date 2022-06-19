"""SEAL-CI model."""
import torch
import random
from tqdm import trange
from collections import Counter
from torch_geometric.utils import to_dense_batch, to_dense_adj
from sklearn.metrics import precision_recall_fscore_support

from utils import hierarchical_graph_reader, adjust_learning_rate 


class SEALCITrainer(object):
    """
    Semi-Supervised Graph Classification: A Hierarchical Graph Perspective Cautious Iteration model.
    """
    def __init__(self, args, dataset_generator, model, dgi_model, device):
        """
        Creating dataset, doing dataset split, creating target and node index vectors.
        :param args: Arguments object.
        """
        self.args = args
        self.device = device 
        self.macro_graph = hierarchical_graph_reader(self.args.hierarchical_graph)
        print()
        self.dataset_generator = dataset_generator#GraphDatasetGenerator(self.args.graphs)
        self.model = model
        self.dgi_model = dgi_model
        self._setup_macro_graph()
        self._create_split()
        self._create_labeled_target()
        self._create_node_indices()

#     def _setup_model(self):
#         """
#         Creating a SEAL model.
#         """
#         self.model = SEAL(self.args, self.dataset_generator.number_of_features,
#                           self.dataset_generator.number_of_labels)

    def _setup_macro_graph(self):
        """
        Creating an edge list for the hierarchical graph.
        """
        self.macro_graph_edges = [[edge[0], edge[1]] for edge in self.macro_graph.edges()]
        self.macro_graph_edges = self.macro_graph_edges + [[edge[1], edge[0]] for edge in self.macro_graph.edges()]
        self.macro_graph_edges = torch.t(torch.LongTensor(self.macro_graph_edges)).to(self.device)

    def _create_split(self):
        """
        Creating a labeled-unlabeled split.
        """
        graph_indices = [index for index in range(len(self.dataset_generator.graphs))]
#         random.shuffle(graph_indices)
#         self.labeled_indices = graph_indices[0:self.args.labeled_count]
#         self.unlabeled_indices = graph_indices[self.args.labeled_count:]
#         data = io.loadmat("/home/workspace/fanyang/work_/Impa/data/processed_arxiv_dataset/hyper_graph/indices.mat")
        self.labeled_indices = graph_indices[0:self.args.labeled_count]
        self.unlabeled_indices = graph_indices[-10000:]
#         tmp_indexs = self.unlabeled_indices[0:100]
#         self.labeled_indices = self.labeled_indices + tmp_indexs
#         self.unlabeled_indices = self.unlabeled_indices[100:]

    def _create_labeled_target(self):
        """
        Creating a mask for labeled instances and a target for them.
        """
        self.labeled_mask = torch.LongTensor([-1 for node in self.macro_graph.nodes()]).to(self.device)
        self.labeled_target = torch.LongTensor([0 for node in self.macro_graph.nodes()]).to(self.device)
        indices = torch.LongTensor(self.labeled_indices)
        unlabeled_indices = torch.LongTensor(self.unlabeled_indices)
#         pdb.set_trace()
        self.labeled_mask[indices] = 1
        self.labeled_target[indices] = self.dataset_generator.target.view(-1)[indices]
        self.labeled_mask[unlabeled_indices] = 0
        self.labeled_target[unlabeled_indices] = self.dataset_generator.target.view(-1)[unlabeled_indices]

    def _create_node_indices(self):
        """
        Creating an index of nodes.
        """
        self.node_indices = [index for index in range(self.macro_graph.number_of_nodes())]
        self.node_indices = torch.LongTensor(self.node_indices)

    def fit_a_single_model(self):
        """
        Fitting a single SEAL model.
        """
#         self._setup_model()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)

        for k in range(self.args.epochs):
            optimizer.zero_grad()
            self.model.train()
#             self.model.graph_level_model = True
#             self.model.hierarchical_model = True
            hc_predictions, ic_predictions, loc_embeddings, global_embeddings, penalties = self.model(self.dataset_generator.graphs_loader, self.macro_graph_edges)
            ic_loss = torch.nn.functional.nll_loss(ic_predictions[self.labeled_mask == 1].softmax(dim=-1).log(),
                                                self.labeled_target[self.labeled_mask == 1],
                                                reduction='mean' )
            hc_loss = torch.nn.functional.nll_loss(hc_predictions[self.labeled_mask == 1].softmax(dim=-1).log(),
                                                self.labeled_target[self.labeled_mask == 1],
                                                reduction='mean')
            kl_loss = torch.nn.functional.kl_div(ic_predictions[self.labeled_mask == 1].softmax(dim=-1).log(), 
                                                 hc_predictions[self.labeled_mask == 1].softmax(dim=-1), 
                                                 reduction='mean')
            num_drugs = loc_embeddings.shape[0]
            adj_tensor = to_dense_adj(self.macro_graph_edges)
            dgi_loss = self.dgi_model.forward(global_embeddings, loc_embeddings, adj_tensor, num_drugs)
            use_ic = False
            adjust_learning_rate(optimizer, k, self.args.learning_rate)
            if k < 200:
                loss = ic_loss#loss + self.args.gamma*penalty
                use_ic = True
            else:
#                 loss = ic_loss*0.1 + hc_loss*0.9 + kl_loss*0.01
                loss = ic_loss*0.1 + hc_loss*0.9 + dgi_loss*0.01
                use_ic = False
            print("epoch: {}, lr: {}, ic_loss: {}, hc_loss: {}, dgi_loss: {}, loss: {}".format(k, 
                                                                                      optimizer.state_dict()['param_groups'][0]['lr'],
                                                                                      ic_loss.detach().cpu().numpy(),
                                                                                      hc_loss.detach().cpu().numpy(),
                                                                                      dgi_loss.detach().cpu().numpy(),
                                                                                      loss.detach().cpu().numpy()))
#             print(k, ic_loss, hc_loss, kl_loss, loss) 
            scores, prediction_indices, accuracy = self.score_a_single_model(mode = 0, use_ic = use_ic)
            print("Unlabeled Accuracy:%g" % round(accuracy, 4))
#             pdb.set_trace()
            print(Counter(prediction_indices.cpu().numpy()))
            loss.backward()
            optimizer.step()

    def score_a_single_model(self, mode = 0, use_ic = False):
        """
        Scoring the SEAL model.
        """
#         self.model.graph_level_model = False
#         self.model.hierarchical_model = False
        self.model.eval()
        hc_predictions, ic_predictions, loc_embeddings, global_embeddings, penalties  = self.model(self.dataset_generator.graphs_loader, self.macro_graph_edges)
        if use_ic:
            scores, prediction_indices = ic_predictions.max(dim=1)
        else:
            scores, prediction_indices = hc_predictions.max(dim=1)
        correct = prediction_indices[self.labeled_mask == mode]
        correct = correct.eq(self.labeled_target[self.labeled_mask == mode]).sum().item()
        normalizer = prediction_indices[self.labeled_mask == mode].shape[0]
        accuracy = float(correct)/float(normalizer)
        sub_predictions = prediction_indices[self.labeled_mask == mode].cpu().numpy()
        sub_labels = self.labeled_target[self.labeled_mask == mode].cpu().numpy()
        pr, re, f1, _ = precision_recall_fscore_support(sub_labels, sub_predictions, average='macro')
        print("precision:{}, recall:{}, f1 score:{}".format(pr, re, f1))
        return scores, prediction_indices, accuracy
            
    def _choose_best_candidate(self, predictions, indices):
        """
        Choosing the best candidate based on predictions.
        :param predictions: Scores.
        :param indices: Vector of likely labels.
        :return candidate: Node chosen.
        :return label: Label of node.
        """
        nodes = self.node_indices[self.labeled_mask == 0]
        sub_predictions = predictions[self.labeled_mask == 0]
        sub_predictions, candidate = sub_predictions.max(dim=0)
        candidate = nodes[candidate]
        label = indices[candidate]
        return candidate, label

    def _update_target(self, candidate, label):
        """
        Adding the new node to the mask and the target is updated with the predicted label.
        :param candidate: Candidate node identifier.
        :param label: Label of candidate node.
        """
        self.labeled_mask[candidate] = 1
        self.labeled_target[candidate] = label      

    def fit(self):
        """
        Training models sequentially.
        """
        print("\nTraining started.\n")
        budget_size = trange(self.args.budget, desc='Unlabeled Accuracy: ', leave=True)
        for _ in budget_size:
            self.fit_a_single_model()
            scores, prediction_indices, accuracy = self.score_a_single_model()
            candidate, label = self._choose_best_candidate(scores, prediction_indices)
            self._update_target(candidate, label)
            budget_size.set_description("Unlabeled Accuracy:%g" % round(accuracy, 4))

    def score(self):
        """
        Scoring the model.
        """
        print("\nModel scoring.\n")
        scores, prediction_indices, accuracy = self.score_a_single_model()
        print("Unlabeled Accuracy:%g" % round(accuracy, 4))
