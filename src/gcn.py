# Based on https://github.com/tkipf/pygcn/blob/master/pygcn/

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv, GATConv



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        #print(adj.shape)
        #print(input.shape)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNSynthetic(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GCNSynthetic, self).__init__()
        self.num_layers = 3
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)
        self.lin = nn.Linear(nhid + nhid + nout, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, adj)
        x = self.lin(torch.cat((x1, x2 ,x3), dim=1))
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GCN2Layer(nn.Module):
    """
    2-layer GCN
    """
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GCN2Layer, self).__init__()
        self.num_layers = 2
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.lin = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x = self.lin(x2)#x = self.lin(torch.cat((x1, x2 ,x3), dim=1))
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class GCN1Layer_PyG(nn.Module):
    """
    1-layer GCN
    """
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GCN1Layer_PyG, self).__init__()
        self.num_layers = 1
        self.gc1 = GCNConv(nfeat, nhid)
        self.lin = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x1 = F.relu(self.gc1(x, edge_index))
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x = self.lin(x1)#x = self.lin(torch.cat((x1, x2 ,x3), dim=1))
        #x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GCN2Layer_TG(nn.Module):
    """
    2-layer GCN
    """
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GCN2Layer_TG, self).__init__()
        self.num_layers = 2
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.lin = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x1 = F.relu(self.gc1(x, edge_index))
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, edge_index))
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        x = self.lin(x2)#x = self.lin(torch.cat((x1, x2 ,x3), dim=1))
        #x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class GCN3Layer_PyG(nn.Module):
    """
    3-layer GCN
    """
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GCN3Layer_PyG, self).__init__()
        self.num_layers = 3
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.gc3 = GCNConv(nhid, nhid)
        self.lin = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x1 = F.relu(self.gc1(x, edge_index))
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, edge_index))
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        x3= self.gc3(x2, edge_index)
        x = self.lin(x3) #x = self.lin(torch.cat((x1, x2 ,x3), dim=1))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class GCN4Layer_PyG(nn.Module):
    """
    4-layer GCN
    """
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GCN4Layer_PyG, self).__init__()
        self.num_layers = 4
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.gc3 = GCNConv(nhid, nhid)
        self.gc4 = GCNConv(nhid, nhid)
        self.lin = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x1 = F.relu(self.gc1(x, edge_index))
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gc2(x1, edge_index)
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = F.relu(self.gc3(x2, edge_index))
        x4 = self.gc4(x3, edge_index)
        x = self.lin(x4) #x = self.lin(torch.cat((x1, x2 ,x3), dim=1))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class GAT2Layer_PyG(nn.Module):
    """
    2-layer GAT
    """
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GAT2Layer_PyG, self).__init__()
        self.num_layers = 4
        self.gc1 = GATConv(nfeat, nhid)
        self.gc2 = GATConv(nhid, nhid)
        self.lin = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x1 = F.relu(self.gc1(x, edge_index))
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, edge_index))
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        x = self.lin(x2) #x = self.lin(torch.cat((x1, x2 ,x3), dim=1))
        #x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GraphConvolution_sparse(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, edge_index):

        #adjMatrix = [[0 for i in range(len(input))] for k in range(len(input))]
        adj = torch.zeros(len(input),len(input))
        # scan the arrays edge_u and edge_v
        for i in range(len(edge_index[0])):
            u = edge_index[0][i]
            v = edge_index[1][i]
            adj[u][v] = 1
        #adj = torch.Tensor(adjMatrix).squeeze()

        support = torch.mm(input, self.weight)
        #print(adj.shape)
        #print(input.shape)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN2Layer_sparse(nn.Module):
    """
    2-layer GCN
    """
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GCN2Layer_sparse, self).__init__()
        self.num_layers = 2
        self.gc1 = GraphConvolution_sparse(nfeat, nhid)
        self.gc2 = GraphConvolution_sparse(nhid, nhid)
        self.lin = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x1 = F.relu(self.gc1(x, edge_index))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, edge_index))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x = self.lin(x2)#x = self.lin(torch.cat((x1, x2 ,x3), dim=1))
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
