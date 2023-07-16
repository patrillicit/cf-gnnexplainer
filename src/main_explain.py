from __future__ import division
from __future__ import print_function
import sys
sys.path.append('..')
import argparse
import pickle
import numpy as np
import time
import torch
from gcn import GCNSynthetic
from cf_explanation.cf_explainer import CFExplainer
from utils.utils import normalize_adj, get_neighbourhood, safe_open
from torch_geometric.utils import dense_to_sparse

""" first run train.py to train the 3-layer GCN with the Cora dataset. Then run this to start the CF-GNNExplainer """

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora')

# Based on original GCN models -- do not change
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--n_layers', type=int, default=3, help='Number of convolutional layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (between 0 and 1)')

# For explainer
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for explainer')#0.005
parser.add_argument('--optimizer', type=str, default="SGD", help='SGD or Adadelta')
parser.add_argument('--n_momentum', type=float, default=0.9, help='Nesterov momentum')
parser.add_argument('--beta', type=float, default=0.5, help='Tradeoff for dist loss')
parser.add_argument('--num_epochs', type=int, default=20, help='Num epochs for explainer')
parser.add_argument('--device', default='cpu', help='CPU or GPU.')
args = parser.parse_args()

print(args)

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.autograd.set_detect_anomaly(True)


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

edge_index = dense_to_sparse(adj)

print(adj.shape)
print(features.shape)
print(labels.shape)
print(idx_train.shape)
print(idx_test.shape)
print(len(edge_index))
print("___________________")

norm_adj = normalize_adj(adj)       # According to reparam trick from GCN paper


# Set up original model, get predictions
model = GCNSynthetic(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
					 nclass=len(labels.unique()), dropout=args.dropout)

model.load_state_dict(torch.load("../models/gcn_3layer_{}.pt".format(args.dataset)))
model.eval()
output = model(features, norm_adj)
y_pred_orig = torch.argmax(output, dim=1)
print("y_true counts: {}".format(np.unique(labels.numpy(), return_counts=True)))
print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.numpy(), return_counts=True)))      # Confirm model is actually doing something
print(idx_test)
idx_test = torch.Tensor([8,88])
idx_test = idx_test.type(torch.int64)
print(idx_test)
# Get CF examples in test set
test_cf_examples = []
start = time.time()
for i in idx_test[:]:
	sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i), edge_index, args.n_layers + 1, features, labels)
	new_idx = node_dict[int(i)]

	# Check that original model gives same prediction on full graph and subgraph
	with torch.no_grad():
		print("Output original model, full adj: {}".format(output[i]))
		print("Output original model, sub adj: {}".format(model(sub_feat, normalize_adj(sub_adj))[new_idx]))


	# Need to instantitate new cf model every time because size of P changes based on size of sub_adj
	explainer = CFExplainer(model=model,
							sub_adj=sub_adj,
							sub_feat=sub_feat,
							n_hid=args.hidden,
							dropout=args.dropout,
							sub_labels=sub_labels,
							y_pred_orig=y_pred_orig[i],
							num_classes = len(labels.unique()),
							beta=args.beta,
							device=args.device)

	if args.device == 'cuda':
		model.cuda()
		explainer.cf_model.cuda()
		adj = adj.cuda()
		norm_adj = norm_adj.cuda()
		features = features.cuda()
		labels = labels.cuda()
		idx_train = idx_train.cuda()
		idx_test = idx_test.cuda()

	cf_example = explainer.explain(node_idx=i, cf_optimizer=args.optimizer, new_idx=new_idx, lr=args.lr,
	                               n_momentum=args.n_momentum, num_epochs=args.num_epochs)
	test_cf_examples.append(cf_example)
	print(cf_example)

	subg_edge_index = [[],[]]
	print(len(cf_example[0][2]))
	print(cf_example[0][2])
	print(len(cf_example[0][3]))
	cf_example[0][2] = (np.rint(cf_example[0][2])).astype(int)
	cf_example[0][3] = (np.rint(cf_example[0][3])).astype(int)

	for row in range(len(cf_example[0][2])):
		for column in range(len(cf_example[0][2])):
			#print(type(cf_example[0][2][row][column]))
			#print(cf_example[0][3][row][column])


			if np.not_equal(cf_example[0][2][row][column],cf_example[0][3][row][column]):
				#print(row)
				#print(column)
				subg_edge_index[0].append(row)
				subg_edge_index[1].append(column)
	print(cf_example[0][2] != cf_example[0][3])
	print(np.count_nonzero(cf_example[0][2] == 1))
	print(np.count_nonzero(cf_example[0][3] == 1))

	node_subset = []
	for i in subg_edge_index:
		for j in i:
			node_subset.append(j)
	node_subset = list(dict.fromkeys(node_subset))
	print(subg_edge_index)
	print(node_subset)
	print("Time for {} epochs of one example: {:.4f}min".format(args.num_epochs, (time.time() - start)/60))
print("Total time elapsed: {:.4f}s".format((time.time() - start)/60))
print("Number of CF examples found: {}/{}".format(len(test_cf_examples), len(idx_test)))

# Save CF examples in test set

with safe_open("../results/{}/{}/{}_cf_examples_lr{}_beta{}_mom{}_epochs{}_seed{}".format(args.dataset, args.optimizer, args.dataset,
																	args.lr, args.beta, args.n_momentum, args.num_epochs, args.seed), "wb") as f:
	pickle.dump(test_cf_examples, f)
