import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from model import *
import random
import numpy as np
from dgl.data.utils import load_graphs, save_graphs
from dataset import Dataset
from sklearn.metrics import f1_score
from argparse import ArgumentParser 

from infomax import GcnInfoMax 


def parse_args():

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="qq", 
                       help="name of dataset (default: qq)")
    parser.add_argument("--model_type", type=str, default="gcn",         
                        help="model type")

    args = parser.parse_args()
    return args


def cal_f1(outputs, labels, masks):
    # outputs = outputs.numpy()
    # labels = labels.numpy()
    # outputs = numpy.argmax(outputs, 1)
    # labels = numpy.argmax(labels, 1)
    # print(outputs, labels)
    all_preds, all_labels, corrects = 0., 0., 0.
    recall, precision, f1, mf1 = 0., 0., 0., 0.
    for i in range(len(masks)):
        if masks[i]:
            # print(outputs[i],labels[i])
            all_preds += outputs[i]
            all_labels += labels[i]
            if outputs[i] == 1 and labels[i] == 1:
                corrects += 1
    if corrects:
        recall = corrects/all_labels
        precision = corrects/all_preds
        f1 = 2*recall*precision/(recall+precision)

    return recall, precision, f1


def train(model, g):

    train_mask = g.ndata['train_mask'].bool()
    val_mask = g.ndata['val_mask'].bool()
    test_mask = g.ndata['test_mask'].bool()
    print('train/dev/test: ', train_mask.sum(), val_mask.sum(), test_mask.sum())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1 = 0, 0, 0, 0, 0
    # g.ndata['feature'] = g.ndata['feature'].float()
    features = g.ndata['feature']
    labels = g.ndata['label']

    for e in range(200):
        model.train()
        hc_loss = 0
        ic_loss = 0
        kl_loss = 0
        if args.model_type == "gcn":
            logits = model(features)
        elif args.model_type == "seal":
            local_logits, logits =  model(features)
        # print(logits.size(), labels.size())
        # print(logits[train_mask].size(), labels[train_mask].size())
        # print(labels[train_mask])
        hc_loss += F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        if args.model_type == "seal":
            ic_loss += F.cross_entropy(local_logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
            kl_loss = torch.nn.functional.kl_div(local_logits[train_mask].softmax(dim=-1).log(), logits[train_mask].softmax(dim=-1), reduction='mean')
        loss = hc_loss#0.9*hc_loss + 0.1*ic_loss + 0.01*kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        logits = model(features)
        pred = logits.argmax(1).numpy()
        # val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        # test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        # Save the best validation accuracy and the corresponding test accuracy.
        rec, pre, f1 = cal_f1(pred, labels, val_mask)
        trec, tpre, tf1 = cal_f1(pred, labels, test_mask)
        print(pred)
        tmf1 = f1_score(labels[test_mask], pred[test_mask], average='macro')
        print(len(labels[test_mask]), len(pred[test_mask]))
        if best_f1 < f1:
            best_f1 = f1
            final_tf1 = tf1
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1

        print('In epoch {}, loss: {:.3f}, val f1: {:.3f} (best {:.3f}), test f1: {:.3f} (best {:.3f}, M {:.3f}, R {:.3f}, P {:.3f})'.format(
            e, loss, f1, best_f1, tf1, final_tf1, final_tmf1, final_trec, final_tpre))


class HierarchicalGNN(nn.Module):

    def __init__(self, in_feats, h_feats, num_classes, glist):

        super(HierarchicalGNN, self).__init__()
        self.local_model = MLP(in_feats, h_feats, num_classes, glist)
        self.global_model = GCN(in_feats, h_feats, num_classes, glist)

    def forward(self, features):
        local_logits = self.local_model(features)
        global_logits = self.global_model(features)
        return local_logits, global_logits



args = parse_args()
# training config
# g = glist[0]
dataset_name = args.dataset 
g = Dataset(dataset_name).graph
in_feats = g.ndata['feature'].shape[1]
# print(g)
# print(g.ndata['feature'].shape)
# print(g.ndata['label'].sum(0))

h_feats = 64
num_classes = 2
weight = 20.
gamma = 0.1 
prior = True 
# graph_path = '{}/our_gwn1'.format(dataset_name)
# glist, _ = load_graphs(graph_path)
# print(g.ndata.keys())
if args.model_type == "gcn":
    model = GCN(in_feats, h_feats, num_classes, [g])
elif args.model_type == "seal":
    model = HierarchicalGNN(in_feats, h_feats, num_classes, [g])
    
# model = GWNN(in_feats, h_feats, num_classes, glist)
train(model, g)
