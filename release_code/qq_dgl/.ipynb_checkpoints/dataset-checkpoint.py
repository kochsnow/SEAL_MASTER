from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl


class Dataset:
    def __init__(self, name='qq'): # qq, amazon, yelp  binary classify
        self.name = name
        graph = None
        if name == 'qq':
            graph, label_dict = load_graphs('qq/qq_graph')
            graph = graph[0]
            graph.ndata['feature'] = graph.ndata['feat']
        elif name == "qq_gc":
            graph, label_dict = load_graphs("qq/qq_graph_gc")
            graph = graph[0]
            graph.ndata["feature"] = graph.ndata["feat"]
        elif name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        else:
            print('no such dataset')
            exit(1)
        # print(graph.ndata['label'])
        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        self.graph = graph

            # g = glist[0]
            # train_mask = g.ndata['train_mask']  # (node, xxx ,xxx)
            # val_mask = g.ndata['val_mask']
            # test_mask = g.ndata['test_mask']
            # in_feats = g.ndata['feat'].shape[1]


# dataset = FraudAmazonDataset()
# graph = dataset[0]
# print(graph)
# dgraph = dgl.to_homogeneous(graph)
# print(dgraph)
# num_classes = dataset.num_classes
# feat = graph.ndata['feature']
# label = graph.ndata['label']
# print(graph.ndata['feature'].shape)
# print(label.sum())
# # test_mask = graph.ndata['test_mask']
# # print(graph.ndata['label'][test_mask].sum())
# # print(graph.edata)
# print(graph['net_upu'])
# print(num_classes, feat, label)

# dataset = FraudAmazonDataset()
# graph = dataset[0]
# num_classes = dataset.num_classes
# feat = graph.ndata['feature']
# label = graph.ndata['label']
# print(num_classes, feat, label)
