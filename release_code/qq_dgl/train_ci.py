import dgl
import os 
import random 
import pdb
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
import warnings

from infomax import GcnInfoMax 
from tqdm import tqdm
warnings.filterwarnings("ignore")


class HierGCN(torch.nn.Module):

    def __init__(self, in_feats, h_feats, num_classes, glist):
        super(HierGCN, self).__init__()
        self.local_model = MLP(in_feats, h_feats, num_classes, glist)
        self.global_model = GCN(in_feats, h_feats, num_classes, glist)

    def forward(self, features):
        local_logits = self.local_model(features)
        global_logits, global_embeddings = self.global_model(features, return_emb=True)
        return local_logits, global_logits, global_embeddings  


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="qq", 
                        help="name of dataset (default: qq)")
    parser.add_argument("--model", type=str, default="gcn", 
                       help="type of model (default: gcn)")
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--alpha", type=float, default=0.9, 
                       help="weight for hierarchical prediction loss (default: 0.9)")
    parser.add_argument("--beta", type=float, default=0.1, 
                       help="weight for internal prediction loss (default: 0.1)")
    parser.add_argument("--eta", type=float, default=0.01, 
                       help="weight for contrastive learning (default: 0.01)")
    parser.add_argument("--lam", type=int, default=100, 
                        help="iteration num of labels for catious learning")
    parser.add_argument("--T", type=int, default=100,
                        help="iteration times t for catious learning")
    parser.add_argument("--epoch", type=int, default=100,
                        help="update epochs")

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


def ci_mask(model, train_mask, val_mask, features):
    model.eval()
    local_logits, logits, embs = model(features) 
    scores = F.kl_div(local_logits.softmax(dim=-1).log(), 
                logits.softmax(dim=-1), reduce = False)
    val_mask_indexs = (val_mask == True).nonzero(as_tuple=False).numpy().reshape(-1,)#.tolist()
    sub_scores = scores[val_mask]
    sub_scores = np.mean(sub_scores.detach().numpy(), axis = -1)
    lam = args.lam
    chosen_index = np.argsort(sub_scores, axis = 0)[0:lam]
    print("chosen_index:", chosen_index)
    candidate = val_mask_indexs[chosen_index]
    train_mask[candidate] = True
    val_mask[candidate] = False 
    pred = logits.argmax(1)#.numpy()
    return candidate, pred
    

def train_ci(g, model, cl_model=None):
    train_mask = g.ndata['train_mask'].bool()
    val_mask = g.ndata['val_mask'].bool()
    test_mask = g.ndata['test_mask'].bool()
    features = g.ndata['feature']
    labels = g.ndata['label']
    train(g, model, train_mask, val_mask, test_mask, 156, labels, cl_model)
    for t in tqdm(range(args.T), total=args.T):
        candidate, pred = ci_mask(model, train_mask, val_mask, features)
        labels[candidate] = pred[candidate]
        train(g, model, train_mask, val_mask, test_mask, args.epoch, labels, cl_model)
    train(g, model, train_mask, val_mask, test_mask, 200, labels, cl_model)

def train(g, model, train_mask, val_mask, test_mask, epochs, labels, cl_model=None):
    print('train/dev/test: ', train_mask.sum(), val_mask.sum(), test_mask.sum(), flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1 = 0, 0, 0, 0, 0
    # g.ndata['feature'] = g.ndata['feature'].float()
    features = g.ndata['feature']
    #labels = g.ndata['label']

    for e in range(epochs):
        model.train()

        if args.model == "gcn":
            logits = model(features)
        elif args.model in ["miracle", "seal"]:
            local_logits, logits, embs = model(features)

        hc_loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        if args.model == "miracle":
            ic_loss = F.cross_entropy(
                local_logits[train_mask], 
                labels[train_mask], weight=torch.tensor([1., weight]))
            # cl_loss, need to be filled.
            # global embedding, local_embedding, adj, num_nodes  
            cl_loss = cl_model(embs, features, g.adjacency_matrix(), num_nodes)
            loss = ic_loss * args.beta + hc_loss * args.alpha + cl_loss * args.eta
        elif args.model == "seal":
            ic_loss = F.cross_entropy(
                local_logits[train_mask], 
                labels[train_mask], weight=torch.tensor([1., weight])
                )
            kl_loss = F.kl_div(local_logits[train_mask].softmax(dim=-1).log(), 
                               logits[train_mask].softmax(dim=-1),
                               reduction="mean")
            loss = ic_loss * args.beta + hc_loss * args.alpha + kl_loss * args.eta 
        elif args.model == "gcn":
            loss = hc_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        if args.model == "gcn":
            logits = model(features)
        elif args.model in ["miracle", "seal"]:
            local_logits, logits, embs = model(features)
        pred = logits.argmax(1).numpy()
        # val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        # test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        # Save the best validation accuracy and the corresponding test accuracy.
        rec, pre, f1 = cal_f1(pred, labels, val_mask)
        trec, tpre, tf1 = cal_f1(pred, labels, test_mask)
        print(pred, flush=True)
        tmf1 = f1_score(labels[test_mask], pred[test_mask], average='macro')
        print(len(labels[test_mask]), len(pred[test_mask]), flush=True)
        if best_f1 < f1:
            best_f1 = f1
            final_tf1 = tf1
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1

        if args.model == "gcn":
            print('In epoch {}, loss: {:.3f}, val f1: {:.3f} (best {:.3f}), test f1: {:.3f} (best {:.3f}, M {:.3f}, R {:.3f}, P {:.3f})'.format(
                e, loss, f1, best_f1, tf1, final_tf1, final_tmf1, final_trec, final_tpre), flush=True)
        elif args.model == "miracle":
            #print('In epoch {}, loss: {:.3f}, val f1: {:.3f} (best {:.3f}), test f1: {:.3f} (best {:.3f}, M {:.3f}, R {:.3f}, P {:.3f})'.format(
            #    e, loss, f1, best_f1, tf1, final_tf1, final_tmf1, final_trec, final_tpre))
            print('In epoch {}, loss: {:.3f}, hc_loss: {:.3f}, ic_loss: {:.3f}, cl loss: {:.3f}, val f1: {:.3f} (best {:.3f}), test f1: {:.3f} (best {:.3f}, M {:.3f}, R {:.3f}, P {:.3f})'.format(
                e, loss, args.alpha * hc_loss, args.beta * ic_loss, args.eta * cl_loss, f1, best_f1, tf1, final_tf1, final_tmf1, final_trec, final_tpre), flush=True)
        elif args.model == "seal":
            print('In epoch {}, loss: {:.3f}, hc_loss: {:.3f}, ic_loss: {:.3f}, kl loss: {:.3f}, val f1: {:.3f} (best {:.3f}), test f1: {:.3f} (best {:.3f}, M {:.3f}, R {:.3f}, P {:.3f})'.format(
                e, loss, args.alpha * hc_loss, args.beta * ic_loss, args.eta * kl_loss, f1, best_f1, tf1, final_tf1, final_tmf1, final_trec, final_tpre), flush=True)


def seed_everything(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


args = parse_args()
print(args)
seed_everything(args.seed)
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# training config
dataset_name = args.dataset 
g = Dataset(dataset_name).graph
num_nodes = g.ndata["feature"].shape[0]
in_feats = g.ndata['feature'].shape[1]
# print(g)
# print(g.ndata['feature'].shape)
# print(g.ndata['label'].sum(0))

h_feats = 64
num_classes = 2
weight = 20.

# graph_path = '{}/our_gwn1'.format(dataset_name)
# glist, _ = load_graphs(graph_path)
# print(g.ndata.keys())
if args.model == "gcn":
    model = GCN(in_feats, h_feats, num_classes, [g])
    train(g, model)
elif args.model == "miracle":
    gamma = 0.1 
    prior = 0 
    model = HierGCN(in_feats, h_feats, num_classes, [g])
    cl_model = GcnInfoMax(gamma, prior, in_feats, h_feats, device=device)
    train(g, model, cl_model=cl_model)
elif args.model == "seal":
    model = HierGCN(in_feats, h_feats, num_classes, [g])
    train_ci(g, model)
