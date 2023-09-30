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
from gcn import GCNSynthetic, GCN2Layer, GCN2Layer_TG, GCN2Layer_sparse, GCN3Layer_PyG, GCN1Layer_PyG, GCN4Layer_PyG, GAT2Layer_PyG

from utils.utils import normalize_adj
#from torch_geometric.utils import accuracy
from torch_geometric.utils import dense_to_sparse

"""

This is the file for training all models (dense and sparse)
Just modify the --model parameter with: GCN2Layer, GCN2Layer_TG, GCNSynthetic, GCN2Layer_sparse, GCN3Layer_PyG
 
 optimal training epochs according to Early Stopping:
 - GCN1Layer_PyG 73 (without Dropout 199)
 - GCN2Layer_TG 46
 - GCN3Layer_PyG 48
 - GCN4Layer_PyG 34 Dropout Problem!
 
"""

# Defaults based on GNN Explainer
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='citeseer')
parser.add_argument('--model', default='GCN3Layer_PyG')#or GCNSynthetic
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--clip', type=float, default=2.0, help='Gradient clip).')
parser.add_argument('--device', default='cpu', help='CPU or GPU.')
args = parser.parse_args()

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)


from data.CitationDatasets import *
if args.dataset == "cora":
	dataset = get_cora()
if args.dataset == "pubmed":
	dataset = get_pubmed()
if args.dataset == "citeseer":
	dataset = get_citeseer()
print(dataset.data)

adjMatrix = [[0 for i in range(len(dataset.data.y))] for k in range(len(dataset.data.y))]

# scan the arrays edge_u and edge_v
for i in range(len(dataset.data.edge_index[0])):
	u = dataset.data.edge_index[0][i]
	v = dataset.data.edge_index[1][i]
	adjMatrix[u][v] = 1


# For models trained using our GCN_synethic from GNNExplainer,
# using hyperparams from GNN explainer tasks
adj = torch.Tensor(adjMatrix).squeeze()

features = torch.Tensor(dataset.data.x).squeeze()
labels = torch.tensor(dataset.data.y).squeeze()

node_idx = [i for i in range(0, len(dataset.data.y))]
idx_train = torch.masked_select(torch.Tensor(node_idx), dataset.data.train_mask)
idx_test = torch.masked_select(torch.Tensor(node_idx), dataset.data.test_mask)
idx_train = idx_train.type(torch.int64)
idx_test = idx_test.type(torch.int64)
print(adj.shape)
print(features.shape)
print(labels.shape)
#print(idx_train)
print(idx_train.shape)
print(idx_test.shape)
print("___________________")
#print(cora.data.y[:140])

# if we use the PTG models, we have to change the adjacency matrix to the edge list
norm_adj = normalize_adj(adj)
print(torch.count_nonzero(adj))
print(torch.count_nonzero(norm_adj))
if args.model == "GCN2Layer_TG" or args.model == "GCN2Layer_sparse":
	norm_edge_index = dense_to_sparse(norm_adj)[0]
	#print(norm_adj)
	#print(norm_adj.shape)
print(dataset.data.edge_index)
print(len(adj))
print(len(norm_adj))
#print(torch.eq(cora.data.edge_index,norm_edge_index))
if args.model == "GCN2Layer":
	model = GCN2Layer(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
					 nclass=len(labels.unique()), dropout=args.dropout)
elif args.model == "GCNSynthetic":
	model = GCNSynthetic(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
						 nclass=len(labels.unique()), dropout=args.dropout)
elif args.model == "GCN2Layer_TG":
	model = GCN2Layer_TG(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
					 nclass=len(labels.unique()), dropout=args.dropout)
elif args.model == "GCN2Layer_sparse":
	model = GCN2Layer_sparse(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
					 nclass=len(labels.unique()), dropout=args.dropout)
elif args.model == "GCN3Layer_PyG":
	model = GCN3Layer_PyG(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
					 nclass=len(labels.unique()), dropout=args.dropout)
elif args.model == "GCN1Layer_PyG":
	model = GCN1Layer_PyG(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
					 nclass=len(labels.unique()), dropout=args.dropout)
elif args.model == "GCN4Layer_PyG":
	model = GCN4Layer_PyG(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
					 nclass=len(labels.unique()), dropout=args.dropout)
elif args.model == "GAT2Layer_PyG":
	model = GAT2Layer_PyG(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
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
	if args.model == "GCN2Layer" or args.model == "GCNSynthetic":
		output = model(features, norm_adj)  # adj or norm_adj
	else:
		output = model(features, dataset.data.edge_index)
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
	return loss_train.item(), acc_train


def test():
	model.eval()
	if args.model == "GCN2Layer" or args.model == "GCNSynthetic":
		output = model(features, norm_adj)  # adj or norm_adj
	else:
		output = model(features, dataset.data.edge_index)
	print(torch.exp(output[155]))
	loss_test = F.nll_loss(output[idx_test], labels[idx_test])
	y_pred = torch.argmax(output, dim=1)
	acc_test = (y_pred[idx_test] == labels[idx_test]).sum().item() / labels[idx_test].numel()
	print("Test set results:",
		  "loss= {:.4f}".format(loss_test.item()),
		  "accuracy= {:.4f}".format(acc_test))
	return y_pred, loss_test.item(), acc_test


# Train model
t_total = time.time()
history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
early_stopping = 10
print_interval = 10
for epoch in range(args.epochs):
	loss_train, acc_train = train(epoch)
	y_pred, loss_test, acc_test = test()
	history["loss"].append(loss_train)
	history["acc"].append(acc_train)
	history["val_loss"].append(loss_test)
	history["val_acc"].append(acc_test)
	# The official implementation in TensorFlow is a little different from what is described in the paper...
	if epoch > early_stopping and loss_test > np.mean(history["val_loss"][-(early_stopping + 1): -1]):

		print("\nEarly stopping...")

		break

	if epoch % print_interval == 0:
		print(f"\nEpoch: {epoch}\n----------")
		print(f"Train loss: {loss_train:.4f} | Train acc: {acc_train:.4f}")
		print(f"  Val loss: {loss_test:.4f} |   Val acc: {acc_test:.4f}")

y_pred, loss_test, acc_test = test()
print(f"\nEpoch: {epoch}\n----------")
print(f"Train loss: {loss_train:.4f} | Train acc: {acc_train:.4f}")
print(f"  Val loss: {loss_test:.4f} |   Val acc: {acc_test:.4f}")
#print(f" Test loss: {test_loss:.4f} |  Test acc: {test_acc:.4f}")

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

torch.save(model.state_dict(), "../models/{}_{}".format(args.model, args.dataset) + ".pt")

# Testing
#y_pred = test()

print("y_true counts: {}".format(np.unique(labels.numpy(), return_counts=True)))
print("y_pred_orig counts: {}".format(np.unique(y_pred.numpy(), return_counts=True)))
print("Finished training!")
