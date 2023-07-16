# Based on https://github.com/tkipf/pygcn/blob/master/pygcn/train.py

from __future__ import division
from __future__ import print_function
import sys
sys.path.append('..')
import argparse
import pickle
import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from gcn import GCNSynthetic
from utils.utils import normalize_adj
#from torch_geometric.utils import accuracy

# Defaults based on GNN Explainer
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--clip', type=float, default=2.0, help='Gradient clip).')
parser.add_argument('--device', default='cpu', help='CPU or GPU.')
args = parser.parse_args()

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)


from data.CitationDatasets import *
cora = get_cora()
print(cora.data)

adjMatrix = [[0 for i in range(len(cora.data.y))] for k in range(len(cora.data.y))]

# scan the arrays edge_u and edge_v
for i in range(len(cora.data.edge_index[0])):
	u = cora.data.edge_index[0][i]
	v = cora.data.edge_index[1][i]
	adjMatrix[u][v] = 1


# For models trained using our GCN_synethic from GNNExplainer,
# using hyperparams from GNN explainer tasks
adj = torch.Tensor(adjMatrix).squeeze()
features = torch.Tensor(cora.data.x).squeeze()
labels = torch.tensor(cora.data.y).squeeze()

node_idx = [i for i in range(0, len(cora.data.y))]
idx_train = torch.masked_select(torch.Tensor(node_idx), cora.data.train_mask)
idx_test = torch.masked_select(torch.Tensor(node_idx), cora.data.test_mask)
idx_train = idx_train.type(torch.int64)
idx_test = idx_test.type(torch.int64)
print(adj.shape)
print(features.shape)
print(labels.shape)
print(idx_train.shape)
print(idx_test.shape)
print("___________________")

norm_adj = normalize_adj(adj)

model = GCNSynthetic(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
                     nclass=len(labels.unique()), dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


if args.device == 'cuda':
	model.cuda()
	features = features.cuda()
	norm_adj = norm_adj.cuda()
	labels = labels.cuda()
	idx_train = idx_train.cuda()
	idx_test = idx_test.cuda()

def train(epoch):
	t = time.time()
	model.train()
	optimizer.zero_grad()
	output = model(features, norm_adj)
	loss_train = model.loss(output[idx_train], labels[idx_train])
	y_pred = torch.argmax(output, dim=1)
	acc_train = (y_pred[idx_train] == labels[idx_train]).sum().item() / labels[idx_train].numel()
	loss_train.backward()
	clip_grad_norm(model.parameters(), args.clip)
	optimizer.step()

	print('Epoch: {:04d}'.format(epoch+1),
		  'loss_train: {:.4f}'.format(loss_train.item()),
		  'acc_train: {:.4f}'.format(acc_train),
		  'time: {:.4f}s'.format(time.time() - t))


def test():
	model.eval()
	output = model(features, norm_adj)
	loss_test = F.nll_loss(output[idx_test], labels[idx_test])
	y_pred = torch.argmax(output, dim=1)
	acc_test = (y_pred[idx_test] == labels[idx_test]).sum().item() / labels[idx_test].numel()
	print("Test set results:",
		  "loss= {:.4f}".format(loss_test.item()),
		  "accuracy= {:.4f}".format(acc_test))
	return y_pred


# Train model
t_total = time.time()
for epoch in range(args.epochs):
	train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

torch.save(model.state_dict(), "../models/gcn_3layer_{}".format(args.dataset) + ".pt")

# Testing
y_pred = test()

print("y_true counts: {}".format(np.unique(labels.numpy(), return_counts=True)))
print("y_pred_orig counts: {}".format(np.unique(y_pred.numpy(), return_counts=True)))
print("Finished training!")
