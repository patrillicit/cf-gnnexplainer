from __future__ import division
from __future__ import print_function
import sys
sys.path.append('../../')
import argparse
import pickle
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
#from torch_geometric.utils import accuracy
from torch.nn.utils import clip_grad_norm
#from gnn_explainer.explainer import explain
from gnnexplainer import GNNExplainer


from src.gcn import GCNSynthetic, GCN2Layer
from src.utils.utils import normalize_adj, get_neighbourhood, safe_open, get_degree_matrix, create_symm_matrix_from_vec, create_vec_from_symm_matrix
from torch_geometric.utils import dense_to_sparse



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora')
parser.add_argument('--model', default='GCN2Layer')#or GCNSynthetic, GCN2Layer dont forget to change n_layers!!!


# Based on original GCN models -- do not change
parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of convolutional layers.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (between 0 and 1)')

# For explainer
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_epochs', type=int, default=500, help='Num epochs for explainer')
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
if args.model == "GCN2Layer":
	model = GCN2Layer(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
					 nclass=len(labels.unique()), dropout=args.dropout)
elif args.model == "GCNSynthetic":
	model = GCNSynthetic(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
						 nclass=len(labels.unique()), dropout=args.dropout)

model.load_state_dict(torch.load("../models/{}_{}.pt".format(args.model, args.dataset)))
model.eval()
output = model(features, norm_adj)
print(output)
y_pred_orig = torch.argmax(output, dim=1)
print(y_pred_orig)
print("y_true counts: {}".format(np.unique(labels.numpy(), return_counts=True)))
print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.numpy(), return_counts=True)))      # Confirm model is actually doing something
print(idx_test)
idx_test = torch.Tensor([146, 155, 167])#([8, 88, 1, 10, 98])
idx_test = idx_test.type(torch.int64)
print(idx_test)

# Get CF examples in test set
test_cf_examples = []
start = time.time()
for i in idx_test[:]:

	sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i), edge_index, args.n_layers + 1, features,
																 labels)
	new_idx = node_dict[int(i)]

	# Create explainer
	explainer = GNNExplainer(
		model=model,
		epochs=200
	)
	_, edge_mask = explainer.explain_node(i, x=features, adj=norm_adj, edge_index=edge_index[0])

	print(edge_mask)

#
# 	best_loss = np.inf
#
# 	for n in range(args.num_epochs):
# 		sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i), edge_index, args.n_layers + 1, features, labels)
# 		new_idx = node_dict[int(i)]
#
# 		# Get CF adj, new prediction
# 		num_nodes = sub_adj.shape[0]
#
# 		# P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
# 		P_vec_size = int((num_nodes * num_nodes - num_nodes) / 2)  + num_nodes
#
# 		# Randomly initialize P_vec in [-1, 1]
# 		r1 = -1
# 		r2 = 1
# 		P_vec = torch.FloatTensor((r1 - r2) * torch.rand(P_vec_size) + r2)
# 		P_hat_symm = create_symm_matrix_from_vec(P_vec, num_nodes)      # Ensure symmetry
# 		P = (F.sigmoid(P_hat_symm) >= 0.5).float()      # threshold P_hat
#
# 		# Get cf_adj, compute prediction for cf_adj
# 		cf_adj = P * sub_adj
# 		A_tilde = cf_adj + torch.eye(num_nodes)
#
# 		D_tilde = get_degree_matrix(A_tilde)
# 		# Raise to power -1/2, set all infs to 0s
# 		D_tilde_exp = D_tilde ** (-1 / 2)
# 		D_tilde_exp[torch.isinf(D_tilde_exp)] = 0
#
# 		# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
# 		cf_norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
#
# 		pred_cf = torch.argmax(model(sub_feat, cf_norm_adj), dim=1)[new_idx]
# 		pred_orig = torch.argmax(model(sub_feat, normalize_adj(sub_adj)), dim=1)[new_idx]
# 		loss_graph_dist = sum(sum(abs(cf_adj - sub_adj))) / 2      # Number of edges changed (symmetrical)
# 		print("Node idx: {}, original pred: {}, cf pred: {}, graph loss: {}".format(i, pred_orig, pred_cf, loss_graph_dist))
#
# 		if (pred_cf != pred_orig) & (loss_graph_dist < best_loss):
# 			best_loss = loss_graph_dist
# 			print("best loss: {}".format(best_loss))
# 			best_cf_example = [i.item(), new_idx.item(),
# 							cf_adj.detach().numpy(), sub_adj.detach().numpy(),
# 							pred_cf.item(), pred_orig.item(), sub_labels[new_idx].numpy(),
# 							sub_adj.shape[0], node_dict,
# 							   loss_graph_dist.item()]
# 	test_cf_examples.append(best_cf_example)
# 	print("Time for {} epochs of one example: {:.4f}min".format(args.num_epochs, (time.time() - start)/60))
# print("Total time elapsed: {:.4f}min".format((time.time() - start)/60))
#
# # Save CF examples in test set
# with safe_open("../results/random_perturb/{}_baseline_cf_examples_epochs{}".format(args.dataset, args.num_epochs), "wb") as f:
# 		pickle.dump(test_cf_examples, f)
