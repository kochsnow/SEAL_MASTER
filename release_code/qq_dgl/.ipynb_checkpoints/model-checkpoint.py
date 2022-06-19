import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, EdgeWeightNorm
import dgl.function as fn
import math
import dgl


class MLP(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, glist=None):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(in_feats, h_feats)
        self.linear2 = torch.nn.Linear(h_feats, num_classes)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, in_feat):
        in_feat = self.dropout(in_feat)
        h = self.linear1(in_feat)
        h = F.relu(h)
        h = self.linear2(h)
        return h


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, glist):
        super(GCN, self).__init__()

        self.g = glist[0]
        self.conv1 = GraphConv(in_feats, h_feats, activation=nn.ReLU())
        self.conv2 = GraphConv(h_feats, num_classes, activation=nn.ReLU())
        # self.linear = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(0.5)
        for param in self.parameters():
            print(type(param), param.size())


    def forward(self, in_feat, return_emb=False):
        in_feat = self.dropout(in_feat)
        h = self.conv1(self.g, in_feat)
        emb = h 
        # h = F.relu(h)
        # h = self.dropout(h)
        # h = self.linear(h)
        h = self.conv2(self.g, h)
        # h = self.linear(h)
        # print(h.shape)
        if return_emb:
            return h, emb
        return h
    
    def graph_embedding(self, in_feat):
        in_feat = self.dropout(in_feat)
        h = self.conv1(self.g, in_feat)
        return h


class GWNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, glist):
        super(GWNN, self).__init__()
        # self.gwn2 = NNConv(h_feats*2, h_feats, 'sum')
        self.glist = glist
        self.gnums = len(glist)
        for g in self.glist:
            g.edata['w'] = g.edata['w'].float()
        self.g = glist[0]
        self.ginv = glist[1]
        self.conv1 = GraphConv(h_feats, h_feats, weight=False, bias=False)
        self.conv2 = GraphConv(h_feats, h_feats, weight=False, bias=False)
        # self.conv2 = GraphConv(h_feats, h_feats, activation=nn.ReLU())
        # self.conv3 = GraphConv(h_feats, num_classes, activation=nn.ReLU())
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.kernel = torch.nn.Parameter(torch.randn(h_feats)/math.sqrt(h_feats))
        # print(self.kernel.requires_grad)
        # self.kernel2 = torch.nn.Parameter(torch.randn(h_feats)/math.sqrt(h_feats))
        # self.kernel_linear = torch.nn.Linear(h_feats, h_feats)

    def forward(self, in_feat):
        in_feat = self.dropout(in_feat.float())
        h = self.linear1(in_feat)
        h = self.conv1(self.g, h, edge_weight=self.g.edata['w'])
        h = self.kernel*h
        h = self.conv2(self.ginv, h, edge_weight=self.ginv.edata['w'])
        # print(self.kernel)
        h = F.relu(h)
        h = self.linear3(h)

        # h = self.conv1(self.g, h, edge_weight=self.g.edata['w'])
        # h = self.kernel2*h
        # h = self.conv1(self.ginv, h, edge_weight=self.ginv.edata['w'])

        # h = F.relu(h)
        # h = self.linear3(h)
        # print(torch.argmax(h,1).sum())
        return h


class GWNN2(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, glist):
        super(GWNN2, self).__init__()
        # self.gwn2 = NNConv(h_feats*2, h_feats, 'sum')
        self.glist = glist
        self.gnums = len(glist)
        for g in self.glist:
            g.edata['w'] = g.edata['w'].float()
            # dgl.add_self_loop(g)
            # print(g)
            # print((g.edata['w']>1e-5).sum())
        # self.kernel_linear = torch.nn.Linear(h_feats, h_feats)
        # , norm='none'
        self.conv1 = GraphConv(in_feats, h_feats, bias=False, activation=nn.ReLU())
        self.conv2 = GraphConv(h_feats*self.gnums, h_feats, bias=False, activation=nn.ReLU())
        self.linear1 = nn.Linear(h_feats*self.gnums, h_feats*self.gnums)
        self.linear2 = nn.Linear(h_feats*self.gnums, h_feats*self.gnums)
        self.linear3 = nn.Linear(h_feats*self.gnums, num_classes)

        self.dropout = torch.nn.Dropout(0.5)
        for param in self.parameters():
            print(type(param), param.size())
        # print(self.parameters())
        # self.norm = EdgeWeightNorm(norm='both')
        # for g in self.glist:
        #     g.edata['nw'] = self.norm(g, g.edata['w'].float()) #g.edata['nw'] =
        # self.kernel = torch.nn.Parameter(torch.randn(h_feats))

    def forward(self, in_feat):
        in_feat = self.dropout(in_feat.float())
        # norm_edge_weight = self.norm(self.glist[0], g.edata['w'])
        # h = [self.conv1(g, in_feat) for g in self.glist]
        h = [self.conv1(g, in_feat, edge_weight=g.edata['w']) for g in self.glist]
        # print(h[0].size())
        h = torch.cat(h, -1)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.linear2(h)
        h = F.relu(h)
        # print(h.size())
        # h = [self.conv2(g, h) for g in self.glist]
        h = [self.conv2(g, h, edge_weight=g.edata['w']) for g in self.glist]
        h = torch.cat(h, -1)
        h = self.linear3(h)

        # h1 = self.gwn2(g1, h)
        # h2 = self.gwn2(g2, h)
        # h = torch.cat([h1,h2],-1)
        # h = self.linear2(h)
        # h = F.relu(h)
        return h
