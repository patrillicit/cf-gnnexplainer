from __future__ import division
from __future__ import print_function
import sys
sys.path.append('..')
import argparse
import pickle
import numpy as np
import time
import torch
from gcn import GCNSynthetic, GCN2Layer, GCN2Layer_TG, GCN2Layer_sparse, GCN3Layer_PyG, GCN1Layer_PyG, GCN4Layer_PyG
from cf_explanation.cf_explainer import CFExplainer
from utils.utils import normalize_adj, get_neighbourhood, safe_open
from torch_geometric.utils import dense_to_sparse

""" This file runs the CF-GNNExplainer. In the Argument --model the
 model type of the trained model needs to be given. 
   
Only dense (adjacency matrix) models can be handled by the CF-GNNExplainer
"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora')
parser.add_argument('--model', default='GCN3Layer_PyG')#or GCNSynthetic, GCN2Layer dont forget to change n_layers!!!

# Based on original GCN models -- do not change
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--n_layers', type=int, default=3, help='Number of convolutional layers.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (between 0 and 1)')

# For explainer
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for explainer')#0.005
parser.add_argument('--optimizer', type=str, default="SGD", help='SGD or Adadelta')
parser.add_argument('--n_momentum', type=float, default=0.9, help='Nesterov momentum')
parser.add_argument('--beta', type=float, default=0.8, help='Tradeoff for dist loss')
parser.add_argument('--num_epochs', type=int, default=500, help='Num epochs for explainer')
parser.add_argument('--device', default='cpu', help='CPU or GPU.')
args = parser.parse_args()

print(args)

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.autograd.set_detect_anomaly(True)


from data.CitationDatasets import *
if args.dataset == "cora":
	dataset = get_cora()
if args.dataset == "pubmed":
	dataset = get_pubmed()
if args.dataset == "citeseer":
	dataset = get_citeseer()
print(dataset.data)
print(dataset.data.edge_index)

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

norm_edge_index = dense_to_sparse(adj)
print(norm_edge_index)
print(norm_edge_index[0])


print(adj.shape)
print(features.shape)
print(labels.shape)
print(idx_train.shape)
print(idx_test.shape)
print(len(norm_edge_index))
print("___________________")

norm_adj = normalize_adj(adj)       # According to reparam trick from GCN paper


# Set up original model, get predictions
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

model.load_state_dict(torch.load("../models/{}_{}.pt".format(args.model, args.dataset)))
model.eval()

if args.model == "GCN2Layer" or args.model == "GCNSynthetic":
	output = model(features, norm_adj)  # adj or norm_adj
else:
	output = model(features, dataset.data.edge_index)


print(output)
y_pred_orig = torch.argmax(output, dim=1)
print(y_pred_orig)
print("y_true counts: {}".format(np.unique(labels.numpy(), return_counts=True)))
print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.numpy(), return_counts=True)))      # Confirm model is actually doing something
print(idx_test)
"""
idx_test = torch.Tensor([3235, 3027, 2519, 2401, 2920, 2670, 2624, 3274, 2544, 2539, 3158, 2386,
        3078, 2802, 2778, 3015, 2402, 3148, 3128, 2694, 2950, 2619, 2362, 2982,
        2354, 2581, 3021, 2674, 2414, 2476, 2936, 2466, 2649, 3287, 2779, 2432,
        3201, 3262, 2441, 2370, 3312, 2965, 2908, 3071, 3233, 2800, 2475, 2971,
        2817, 2888, 3281, 2583, 2720, 3228, 3011, 2868, 2913, 2372, 2793, 2387,
        2410, 2713, 2608, 2863, 3141, 3278, 2512, 3049, 2981, 2598, 2505, 3029,
        2897, 2648, 2644, 2998, 3325, 3126, 2960, 2733, 2835, 3057, 2507, 2791,
        2446, 3234, 2922, 2589, 2701, 3123, 3010, 2330, 3108, 2966, 3289, 2852,
        2838, 2660, 2734, 2429])
"""


idx_test = torch.Tensor([2333, 2266, 2560, 1797, 2706, 2684, 2443, 2177, 1788,
                         1825, 2353, 1888, 2494, 2305, 1710, 2189, 2071,
        2606, 2038, 1725, 1800, 2693, 2114, 1894,
        1930, 2299, 2212, 1829, 2084, 2222, 2155, 2178, 1966, 2478, 1915, 2179,
        2093, 2021, 2304, 1782, 1813, 2651, 2621, 2292, 2059, 1970, 2057, 1738,
        1721, 2640, 1997, 1892, 1895, 2461, 1992, 2149, 2425, 2151, 2573, 2269,
        1998, 2642, 1819, 1843, 2337, 2232, 1855, 2167, 2325, 2349, 1823, 2062,
        1913, 1914, 2165, 2007, 2100, 2224, 2508, 2563, 2009, 2466, 2340, 2441,
        2519, 2491, 1812, 1961, 2538, 2502, 2378, 2459, 2589, 2704, 1799, 2024,
        1882, 2499, 1974, 1948])

idx_test = idx_test.type(torch.int64)
print(idx_test)
# Get CF examples in test set
test_cf_examples = []
results_table = []
start = time.time()
for i in idx_test[:]:
	print(i)
	sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i), norm_edge_index, args.n_layers + 1, features, labels)
	new_idx = node_dict[int(i)]
	print(node_dict)
	print(len(node_dict))
	if sub_adj.nelement() == 0:
		print("EUREKA")
		sub_adj = torch.Tensor(0)
	print(sub_adj)


	# Check that original model gives same prediction on full graph and subgraph
	with torch.no_grad():
		print("Output original model, full adj: {}".format(output[i]))
		#print("Output original model, sub adj: {}".format(model(sub_feat, normalize_adj(sub_adj))[new_idx]))


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
							device=args.device,
							model_type=args.model)

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
	if len(cf_example) == 0:
		continue
	if len(cf_example) != 0:
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
		for ele in subg_edge_index:
			for j in ele:
				node_subset.append(j)
		node_subset = list(dict.fromkeys(node_subset))
		print(subg_edge_index)
		print(node_subset)
		print(sub_labels)
		print(len(sub_labels))
		node_subset_reindex = []
		for item in node_subset:
			node_subset_reindex.append(list(node_dict.keys())[list(node_dict.values()).index(item)])

		subg_edge_index_reindex = [[],[]]
		for col in range(len(subg_edge_index_reindex)):
			for item in subg_edge_index[col]:
				subg_edge_index_reindex[col].append(list(node_dict.keys())[list(node_dict.values()).index(item)])
		node_subset_reindex = torch.Tensor(node_subset_reindex)
		subg_edge_index_reindex = torch.Tensor(subg_edge_index_reindex)
		print(node_subset_reindex)
		print(subg_edge_index_reindex)
		#print(cf_example[7])

		# Fidelity+
		def fidelity_p():
			subg_edge_index_reindex_t = torch.transpose(subg_edge_index_reindex,0,1)
			print(subg_edge_index_reindex_t)
			edge_pairs = torch.transpose(dataset.data.edge_index,0,1).tolist()
			print(edge_pairs)
			for pair in subg_edge_index_reindex_t:
				pair = pair.type(torch.int64)
				print(pair)
				print(pair.tolist())
				edge_pairs.remove(pair.tolist())
			edge_pairs = torch.Tensor(edge_pairs)
			edge_index_fp = torch.transpose(edge_pairs,0,1).type(torch.int64)
			print(edge_index_fp)
			print(edge_index_fp.shape)

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

			model.load_state_dict(torch.load("../models/{}_{}.pt".format(args.model, args.dataset)))
			model.eval()
			output_fid = model(features, edge_index_fp)
			print(output_fid.shape)
			print(output_fid)

			y_pred_fid = torch.argmax(output_fid, dim=1)
			print(y_pred_orig.shape)
			print(y_pred_orig)
			print(y_pred_orig[0])
			print(y_pred_orig[1])
			print(i)
			print(y_pred_orig[i])




			prob1 = torch.exp(output[i][y_pred_orig[i]])
			print(prob1)
			prob2 = torch.exp(output_fid[i][y_pred_orig[i]])
			print(prob2)

			fid_p = prob1 - prob2

			return fid_p.item()


		def fidelity_n():

			edge_index_fp = torch.Tensor(subg_edge_index_reindex).type(torch.int64)
			print(edge_index_fp)

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

			model.load_state_dict(torch.load("../models/{}_{}.pt".format(args.model, args.dataset)))
			model.eval()
			output_fid = model(features, edge_index_fp)
			print(output_fid)
			y_pred_fid = torch.argmax(output_fid, dim=1)

			prob1 = torch.exp(output[i][y_pred_orig[i]])
			print(prob1)
			prob2 = torch.exp(output_fid[i][y_pred_orig[i]])
			print(prob2)

			fid_p = prob1 - prob2
			print(fid_p)

			return fid_p.item()

		fid_p = fidelity_p()

		fid_n = fidelity_n()

		# Characterization Score
		w_p = 0.5
		w_n = 0.5
		char_score =((w_p + w_n) * fid_p * (1 - fid_n))/(w_p * (1 - fid_n) + w_n * fid_p)

		# Sparsity

		def sparsity(edge_list_k):
			sparsity = 1 - (len(edge_list_k[0]) /  len(dataset.data.edge_index[0]))

			num_edg_in_neigh = torch.count_nonzero(sub_adj)
			sparsity_neig = 1 - (len(edge_list_k[0]) / num_edg_in_neigh)
			if torch.is_tensor(sparsity_neig):
				sparsity_neig = sparsity_neig.item()
			if sparsity_neig < 0:
				sparsity_neig = 0
			return sparsity, sparsity_neig


		spar, spar_neig = sparsity(subg_edge_index_reindex)



		def contrastivity(i):

			y_i = dataset.data.y[i]
			lab = list(set(dataset.data.y.tolist()))
			print(lab)
			lab.remove(y_i)
			lab = torch.tensor(lab)
			edge_masks_cont = []
			hamming_list_k3 = []
			for labell in lab:
				explainer_cont = CFExplainer(model=model,
											 sub_adj=sub_adj,
											 sub_feat=sub_feat,
											 n_hid=args.hidden,
											 dropout=args.dropout,
											 sub_labels=sub_labels,
											 y_pred_orig=labell,
											 num_classes=len(labels.unique()),
											 beta=args.beta,
											 device=args.device,
											 model_type=args.model)
				#targets = dataset.data.y.clone().detach()
				#targets[i] = labell
				#(targets)
				#print(targets[i])
				#print(targets[i + 1])
				cf_example = explainer_cont.explain(node_idx=i, cf_optimizer=args.optimizer, new_idx=new_idx, lr=args.lr,
											   n_momentum=args.n_momentum, num_epochs=args.num_epochs)

				if len(cf_example) != 0:
					subg_edge_index_cont = [[], []]
					print(len(cf_example[0][2]))
					print(cf_example[0][2])
					print(len(cf_example[0][3]))
					cf_example[0][2] = (np.rint(cf_example[0][2])).astype(int)
					cf_example[0][3] = (np.rint(cf_example[0][3])).astype(int)

					for row in range(len(cf_example[0][2])):
						for column in range(len(cf_example[0][2])):
							if np.not_equal(cf_example[0][2][row][column], cf_example[0][3][row][column]):
								subg_edge_index_cont[0].append(row)
								subg_edge_index_cont[1].append(column)
					print(cf_example[0][2] != cf_example[0][3])
					print(np.count_nonzero(cf_example[0][2] == 1))
					print(np.count_nonzero(cf_example[0][3] == 1))

					node_subset_cont = []
					for ele in subg_edge_index_cont:
						for j in ele:
							node_subset_cont.append(j)
					node_subset_cont = list(dict.fromkeys(node_subset))

					node_subset_cont_reindex = []
					for item in node_subset_cont:
						node_subset_cont_reindex.append(list(node_dict.keys())[list(node_dict.values()).index(item)])

					subg_edge_index_cont_reindex = [[], []]
					for col in range(len(subg_edge_index_cont_reindex)):
						for item in subg_edge_index_cont[col]:
							subg_edge_index_cont_reindex[col].append(
								list(node_dict.keys())[list(node_dict.values()).index(item)])
					node_subset_cont_reindex = torch.Tensor(node_subset_cont_reindex)
					subg_edge_index_cont_reindex = torch.Tensor(subg_edge_index_cont_reindex)
					print(node_subset_cont_reindex)
					print(subg_edge_index_cont_reindex)


				# calculate Hamming distance for k=5
				aa_k3 = np.transpose(subg_edge_index_reindex)
				aa_k3 = [tuple(lst) for lst in aa_k3]
				print(aa_k3)

				bb_k3 = np.transpose(subg_edge_index_cont_reindex)
				bb_k3 = [tuple(lst) for lst in bb_k3]
				print(bb_k3)
				union_k3 = set(aa_k3 + bb_k3)
				print(union_k3)
				intersection_k3 = set(aa_k3).intersection(bb_k3)
				print(intersection_k3)
				# Contrastivity: 100% all edges are different, 0% the subgraph is the same
				resu_k3 = 1 - (len(intersection_k3) / len(aa_k3))
				print(resu_k3)

				hamming_list_k3.append(resu_k3)




			# calculate mean of hamming distance of all instances
			cont_k3 = sum(hamming_list_k3) / len(hamming_list_k3)
			print(cont_k3)
			return cont_k3

		cont = contrastivity(i)


		def stability(i):
			hamming_list_k3_stab = []
			hamming_list_k5_stab = []

			while len(hamming_list_k5_stab) < 6:
				adj_attacked = torch.clone(adj)
				print(adj_attacked)

				import random
				adj_ones = adj.nonzero()
				adj_zeros = (adj == 0).nonzero()
				print(len(adj_ones))
				ones_kept = random.sample(range(len(adj_ones)), int(len(adj_ones) * 0.9))
				print(len(ones_kept))
				ones_kept = adj_ones[ones_kept]
				print(ones_kept)
				print(len(adj_zeros))
				ones_added = random.sample(range(len(adj_zeros)), int(len(adj_zeros) * 0.001))
				print(len(ones_added))
				ones_added = adj_zeros[ones_added]
				print(ones_added)

				attacked_edge_index = torch.transpose(torch.cat((ones_kept, ones_added)), 0, 1)
				print(attacked_edge_index)

				if args.model == "GCN2Layer":
					model_stab = GCN2Layer(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
										   nclass=len(labels.unique()), dropout=args.dropout)
				elif args.model == "GCNSynthetic":
					model_stab = GCNSynthetic(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
											  nclass=len(labels.unique()), dropout=args.dropout)
				elif args.model == "GCN1Layer_PyG":
					model_stab = GCN1Layer_PyG(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
											   nclass=len(labels.unique()), dropout=args.dropout)
				elif args.model == "GCN2Layer_TG":
					model_stab = GCN2Layer_TG(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
											  nclass=len(labels.unique()), dropout=args.dropout)
				elif args.model == "GCN2Layer_sparse":
					model_stab = GCN2Layer_sparse(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
												  nclass=len(labels.unique()), dropout=args.dropout)
				elif args.model == "GCN3Layer_PyG":
					model_stab = GCN3Layer_PyG(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
											   nclass=len(labels.unique()), dropout=args.dropout)

				model_stab.load_state_dict(torch.load("../models/{}_{}.pt".format(args.model, args.dataset)))
				model_stab.eval()
				print(model)
				output_stab = model(features, attacked_edge_index)
				print(output_stab)
				y_pred_stab = torch.argmax(output_stab, dim=1)
				print(y_pred_stab)
				print(y_pred_orig)

				print(y_pred_stab[i])
				print(y_pred_orig[i])
				if torch.equal(y_pred_stab[i], y_pred_orig[i]):
					"""Does the Attacked Edge Index need to be normalized????"""
					sub_adj_stab, sub_feat_stab, sub_labels_stab, node_dict_stab = get_neighbourhood(int(i),(attacked_edge_index,attacked_edge_index),
																				 args.n_layers + 1, features, labels)
					new_idx = node_dict[int(i)]
					print(node_dict)
					print(len(node_dict))


					# Need to instantitate new cf model every time because size of P changes based on size of sub_adj
					explainer = CFExplainer(model=model_stab,
											sub_adj=sub_adj_stab,
											sub_feat=sub_feat_stab,
											n_hid=args.hidden,
											dropout=args.dropout,
											sub_labels=sub_labels_stab,
											y_pred_orig=y_pred_stab[i],
											num_classes=len(labels.unique()),
											beta=args.beta,
											device=args.device,
											model_type=args.model)


					cf_example = explainer.explain(node_idx=i, cf_optimizer=args.optimizer, new_idx=new_idx, lr=args.lr,
												   n_momentum=args.n_momentum, num_epochs=args.num_epochs)

					if len(cf_example) != 0:
						subg_edge_index_cont = [[], []]
						print(len(cf_example[0][2]))
						print(cf_example[0][2])
						print(len(cf_example[0][3]))
						cf_example[0][2] = (np.rint(cf_example[0][2])).astype(int)
						cf_example[0][3] = (np.rint(cf_example[0][3])).astype(int)

						for row in range(len(cf_example[0][2])):
							for column in range(len(cf_example[0][2])):
								if np.not_equal(cf_example[0][2][row][column], cf_example[0][3][row][column]):
									subg_edge_index_cont[0].append(row)
									subg_edge_index_cont[1].append(column)
						print(cf_example[0][2] != cf_example[0][3])
						print(np.count_nonzero(cf_example[0][2] == 1))
						print(np.count_nonzero(cf_example[0][3] == 1))

						node_subset_cont = []
						for ele in subg_edge_index_cont:
							for j in ele:
								node_subset_cont.append(j)
						node_subset_cont = list(dict.fromkeys(node_subset))

						node_subset_cont_reindex = []
						for item in node_subset_cont:
							node_subset_cont_reindex.append(
								list(node_dict_stab.keys())[list(node_dict_stab.values()).index(item)])

						subg_edge_index_cont_reindex = [[], []]
						for col in range(len(subg_edge_index_cont_reindex)):
							for item in subg_edge_index_cont[col]:
								subg_edge_index_cont_reindex[col].append(
									list(node_dict_stab.keys())[list(node_dict_stab.values()).index(item)])
						node_subset_cont_reindex = torch.Tensor(node_subset_cont_reindex)
						subg_edge_index_cont_reindex = torch.Tensor(subg_edge_index_cont_reindex)
						print(node_subset_cont_reindex)
						print(subg_edge_index_cont_reindex)



					# calculate Hamming distance for k=3
					aa_k3 = np.transpose(subg_edge_index_reindex)
					aa_k3 = [tuple(lst) for lst in aa_k3]
					print(aa_k3)

					print(subg_edge_index_cont_reindex)
					bb_k3 = []
					if len(subg_edge_index_cont_reindex[0]) > 0:
						bb_k3 = np.transpose(subg_edge_index_cont_reindex)
						bb_k3 = [tuple(lst) for lst in bb_k3]
					print(bb_k3)
					union_k3 = set(aa_k3 + bb_k3)
					print(union_k3)
					intersection_k3 = set(aa_k3).intersection(bb_k3)
					print(intersection_k3)
					# Stability: 100%: exactly the same subgraph, 0% totally different edges selected
					resu_k3 = len(intersection_k3) / len(aa_k3)
					print(resu_k3)

					hamming_list_k3_stab.append(resu_k3)


			# calculate mean of hamming distance of all instances
			stab_k3 = sum(hamming_list_k3_stab) / len(hamming_list_k3_stab)

			print(stab_k3)


			return stab_k3


		#stab = stability(i)
		stab = 0

		""" Create Output File """

		colors = ["red", "orange", "lightblue", "green", "blue", "purple", "grey"]
		path = r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\CF-GNN Experiments\{}_node{}_layers{}_gnnexp".format(
			args.dataset,
			i,
			args.n_layers)

		d = [subg_edge_index_reindex[0], subg_edge_index_reindex[1], [spar], [spar_neig],
			 [fid_p, fid_n], [char_score], [cont], [stab]]
		from itertools import zip_longest

		df = pd.DataFrame(zip_longest(*d, fillvalue=''),
						  columns=['Edge-source', 'Edge-target', 'Sparsity', 'Neig-Sparsity', 'Fid + / - ', 'Char-Score', 'Contrastivity', 'Stability'])
		# df.rename(columns={'0':'Nodes', '1':'Edge source', '2':'Edge:target', '3':'Sparsity'}, inplace=True)
		print(df.head(15))
		df.to_csv(path + ".csv")

		""" Visualization of the Explanations """
		from visualization.subgraph_plotting import *

		def visualize_graph(nodes, edges, metrics):

			colors = ["red", "orange", "lightblue", "green", "blue", "purple", "grey"]
			path = r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\CF-GNN Experiments\{}_node{}_layers{}_cfgnnexp".format(
				args.dataset,
				i,
				args.n_layers)

			prediction_labels = []
			prediction_prob = []
			for item in output.tolist():
				prediction_prob.append(max(item))
				prediction_labels.append(item.index(max(item)))
			draw_graph(node_index=nodes, edge_index=edges, y=dataset.data.y,
					   prediction=prediction_labels, colors=colors, path=path, label=metrics)


		visualize_graph( node_subset_reindex.type(torch.int64).tolist(), subg_edge_index_reindex.type(torch.int64).tolist(),
						' Spar: ' + str(d[2][0]) + 'Spar-Neig: ' + str(d[3][0]) + ' Fid+: ' + str(
							d[4][0]) + ' Fid-: ' + str(d[4][1])
						+ ' Char: ' + str(d[5][0]) + ' Contrast: ' + str(d[6][0]) + ' Stability: ' + str(d[7][0]))

		instance = [i, spar, spar_neig, fid_p, fid_n, char_score, cont, stab]

		results_table.append(instance)


		df_results = pd.DataFrame(results_table,
									 columns=['Node', 'Sparsity', 'Sparsity-Neig', 'Fid3+', 'Fid3-', 'Char3 ',
											  'Contrastivity', 'Stability'])
		# df.rename(columns={'0':'Nodes', '1':'Edge source', '2':'Edge:target', '3':'Sparsity'}, inplace=True)

		df_results.to_csv(
			r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\CF-GNN Experiments\aa_results_layers{}_cfgnnexp".format(
				args.n_layers) + ".csv")

