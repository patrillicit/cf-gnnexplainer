from __future__ import division
from __future__ import print_function
import sys

sys.path.append('..')
import argparse
import pickle
import numpy as np
import time
import torch
from gcn import GCNSynthetic, GCN2Layer, GCN2Layer_TG, GCN2Layer_sparse, GCN3Layer_PyG, GCN1Layer_PyG
from cf_explanation.cf_explainer import CFExplainer
from utils.utils import normalize_adj, get_neighbourhood, safe_open
from torch_geometric.utils import dense_to_sparse

""" 
    This file runs applies the Pytorch Geometric GNNExplainer for only the sparse models
    
    Use only sparse models: GCN2Layer_TG, GCN2Layer_sparse
    
    We pass the edge_index to the GNNExplainer

"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='citeseer')
parser.add_argument('--model', default='GCN2Layer_TG')  # or GCNSynthetic, GCN2Layer dont forget to change n_layers!!!

# Based on original GCN models -- do not change
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of convolutional layers.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (between 0 and 1)')

# For explainer
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for explainer')  # 0.005
parser.add_argument('--optimizer', type=str, default="SGD", help='SGD or Adadelta')
parser.add_argument('--n_momentum', type=float, default=0.9, help='Nesterov momentum')
parser.add_argument('--beta', type=float, default=0.8, help='Tradeoff for dist loss')
parser.add_argument('--num_epochs', type=int, default=800, help='Num epochs for explainer')
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

edge_index = dense_to_sparse(adj)

print(adj.shape)
print(features.shape)
print(labels.shape)
print(idx_train.shape)
print(idx_test.shape)
print(len(edge_index))
print("___________________")

norm_adj = normalize_adj(adj) # According to reparam trick from GCN paper
if args.model == "GCN2Layer_TG" or args.model == "GCN2Layer_sparse":
    norm_edge_index = dense_to_sparse(norm_adj)[0]

# Set up original model, get predictions
if args.model == "GCN2Layer":
    model = GCN2Layer(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
                      nclass=len(labels.unique()), dropout=args.dropout)
elif args.model == "GCNSynthetic":
    model = GCNSynthetic(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
                         nclass=len(labels.unique()), dropout=args.dropout)
elif args.model == "GCN1Layer_PyG":
	model = GCN1Layer_PyG(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
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

model.load_state_dict(torch.load("../models/{}_{}.pt".format(args.model, args.dataset)))
model.eval()
output = model(features, dataset.data.edge_index)
print(output)
y_pred_orig = torch.argmax(output, dim=1)
print(y_pred_orig)
print("y_true counts: {}".format(np.unique(labels.numpy(), return_counts=True)))
print("y_pred_orig counts: {}".format(
    np.unique(y_pred_orig.numpy(), return_counts=True)))  # Confirm model is actually doing something
print(idx_test)

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
"""

idx_test = idx_test.type(torch.int64)



"""
import random
print(idx_test)
test_indices = random.sample(range(len(idx_test)), 100)
print(idx_test[test_indices])
quit()
"""
# Get CF examples in test set
test_cf_examples = []
start = time.time()
results_table_k3 = []
results_table_k5 = []


for i in idx_test[:]:

    # neighborhood is not needed for the GNNExplainer 
    sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i), edge_index, args.n_layers + 1, features,
                                                                 labels)


    def plot_k_hop_subgraph(node, hops, dataset_edge_index, y, prediction, colors):
        print(node)
        print(hops)
        print(dataset_edge_index)
        nodes, edge_list, mapping, edge_mask = k_hop_subgraph(node, hops, dataset_edge_index)

        subg = graphviz.Digraph('node' + str(node) + '_' + str(hops) + 'hop', comment='Neighborhood-Subgraph')
        print(str(hops) + "-Subgraph of Node " + str(node) + " has " + str(len(nodes)) + " nodes and " + str(
            len(edge_list[0])) + " edges.")
        for node in nodes.tolist():
            subg.node(str(node), fontcolor=colors[prediction[node]], color=colors[y[node]])
        strings0 = []
        for ele in edge_list[0].tolist():
            strings0.append(str(ele))
        strings1 = []
        for ele in edge_list[1].tolist():
            strings1.append(str(ele))

        edge_list = [strings0, strings1]
        edge_list_trans = tuple(zip(*edge_list))
        subg.edges(edge_list_trans)
        
        subg.render(
            directory=r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\Neighborhood with Labels CiteSeer")  # r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\Experiments"
    plot_k_hop_subgraph(i.item(),1,dataset.data.edge_index, dataset.data.y, y_pred_orig, ["red", "orange", "lightblue", "green", "blue", "purple", "grey"])
    continue


    """
    new_idx = node_dict[int(i)]
    print(node_dict)
    print(len(node_dict))

    # Check that original model gives same prediction on full graph and subgraph
    
    with torch.no_grad():
        print("Output original model, full adj: {}".format(output[i]))
        print("Output original model, sub adj: {}".format(model(sub_feat, normalize_adj(sub_adj))[new_idx]))
    """

    from torch_geometric.explain import Explainer, GNNExplainer, Explanation, PGExplainer, ThresholdConfig

    # Need to instantitate new cf model every time because size of P changes based on size of sub_adj
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',  # Model returns log probabilities.
        ),
        threshold_config=None#ThresholdConfig("topk", 5),
    )

    # Generate explanation for the node
    explanation = explainer(dataset.data.x, dataset.data.edge_index, index=i) #norm or not (norm_edge_index)
    print(explanation)
    print(explanation.edge_mask)
    print(torch.equal(explanation.edge_index,dataset.data.edge_index))


    adjmatrix = [[0 for i in range(len(dataset.data.y))] for k in range(len(dataset.data.y))]

    # scan the arrays edge_u and edge_v
    edge_i = 0
    for score in explanation.edge_mask:
        u = dataset.data.edge_index[0][edge_i]
        v = dataset.data.edge_index[1][edge_i]
        edge_i = edge_i + 1
        adjmatrix[u][v] = score
    print(adjmatrix)
    edge_mask_acc = []
    #edge_index_uni = [[],[]]
    row_it = 0
    #element_count = 0
    for roww in range(len(adjmatrix)):
        row_it = row_it + 1
        for columnn in range(row_it):
            importance = adjmatrix[roww][columnn] = adjmatrix[columnn][roww] = (adjmatrix[roww][columnn]+adjmatrix[columnn][roww])/2
            #if importance > 0:
                #element_count = element_count + 1
                #edge_mask_acc.append(importance)
                #edge_index_uni[0].append(roww)
                #edge_index_uni[1].append(columnn)

    for ele in range(len(explanation.edge_index[0])):
        edge_mask_acc.append(adjmatrix[explanation.edge_index[0][ele]][explanation.edge_index[1][ele]])
    edge_mask_acc = torch.tensor(edge_mask_acc)
    # Extract the top-k nodes
    top_k = torch.topk(edge_mask_acc, len(dataset.data.edge_index[0]))
    #print(edge_index_uni)

    edge_list = [[],[]]
    for index in top_k.indices:
        edge_list[0].append(explanation.edge_index[0][index].item())
        edge_list[1].append(explanation.edge_index[1][index].item())
    print(edge_list)

    """ Evaluation Metrics """
    """
    edge_list_k3 = [edge_list[0][:3], edge_list[1][:3]]
    edge_list_k5 = [edge_list[0][:5], edge_list[1][:5]]

    node_subset_k3 = []
    node_subset_k5 = []
    for ele in edge_list_k3:
        for j in ele:
            node_subset_k3.append(j)
    node_subset_k3 = list(dict.fromkeys(node_subset_k3))

    for ele in edge_list_k5:
        for j in ele:
            node_subset_k5.append(j)
    node_subset_k5 = list(dict.fromkeys(node_subset_k5))
    """

    # Fidelity+
    def create_subsets(k):
        edge_list_k = [edge_list[0][:k], edge_list[1][:k]]
        node_subset_k = []
        for ele in edge_list_k:
            for j in ele:
                node_subset_k.append(j)
        node_subset_k = list(dict.fromkeys(node_subset_k))

        bool_mask_topk = torch.zeros(len(dataset.data.edge_index[0])).bool()
        #print(bool_mask)

        for item in range(k):
            bool_mask_topk[top_k.indices[item].item()] = True

        bool_mask_without_topk = torch.ones(len(dataset.data.edge_index[0])).bool()
        #print(bool_mask)

        for item in range(k):
            bool_mask_without_topk[top_k.indices[item].item()] = False

        return [k, edge_list_k, node_subset_k, bool_mask_topk, bool_mask_without_topk]


    def fidelity_p(k, bool_mask):
        """
        edge_list_k = [edge_list[0][:k], edge_list[1][:k]]
        bool_mask = torch.ones(10556).bool()
        print(bool_mask)

        for item in range(k):
            bool_mask[top_k.indices[item].item()] = False
        """
        edge_index_fp = [[],[]]
        edge_index_fp[0] = torch.masked_select(dataset.data.edge_index[0], bool_mask).tolist()
        edge_index_fp[1] = torch.masked_select(dataset.data.edge_index[1], bool_mask).tolist()
        edge_index_fp = torch.Tensor(edge_index_fp).type(torch.int64)

        if args.model == "GCN2Layer":
            model = GCN2Layer(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
                              nclass=len(labels.unique()), dropout=args.dropout)
        elif args.model == "GCNSynthetic":
            model = GCNSynthetic(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
                                 nclass=len(labels.unique()), dropout=args.dropout)
        elif args.model == "GCN1Layer_PyG":
            model = GCN1Layer_PyG(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
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

        model.load_state_dict(torch.load("../models/{}_{}.pt".format(args.model, args.dataset)))
        model.eval()
        output_fid = model(features, edge_index_fp)
        print("FIDELITY")
        print(len(output_fid))
        y_pred_fid = torch.argmax(output_fid, dim=1)

        prob1 = torch.exp(output[i][y_pred_orig[i]])
        print(prob1)
        prob2 = torch.exp(output_fid[i][y_pred_orig[i]])
        print(prob2)

        fid_p = prob1 - prob2
        print(fid_p)

        return fid_p.item()

    def fidelity_n(k, bool_mask):
        """
        edge_list_k = [edge_list[0][:k], edge_list[1][:k]]
        bool_mask = torch.zeros(10556).bool()
        print(bool_mask)

        for item in range(k):
            bool_mask[top_k.indices[item].item()] = True
        """
        edge_index_fp = [[],[]]
        edge_index_fp[0] = torch.masked_select(dataset.data.edge_index[0], bool_mask).tolist()
        edge_index_fp[1] = torch.masked_select(dataset.data.edge_index[1], bool_mask).tolist()
        edge_index_fp = torch.Tensor(edge_index_fp).type(torch.int64)
        print(edge_index_fp.shape)

        if args.model == "GCN2Layer":
            model = GCN2Layer(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
                              nclass=len(labels.unique()), dropout=args.dropout)
        elif args.model == "GCNSynthetic":
            model = GCNSynthetic(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
                                 nclass=len(labels.unique()), dropout=args.dropout)
        elif args.model == "GCN1Layer_PyG":
            model = GCN1Layer_PyG(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
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

    data_k3 = create_subsets(6)
    data_k5 = create_subsets(10)

    fid_p_k3 = fidelity_p(6, data_k3[4])
    fid_p_k5 = fidelity_p(10, data_k5[4])

    fid_n_k3 = fidelity_n(6, data_k3[3])
    fid_n_k5 = fidelity_n(10, data_k5[3])

    # Characterization Score
    w_p = 0.5
    w_n = 0.5
    char_score_k3 = ((w_p + w_n) * fid_p_k3 * (1 - fid_n_k3)) / (w_p * (1 - fid_n_k3) + w_n * fid_p_k3)
    char_score_k5 = ((w_p + w_n) * fid_p_k5 * (1 - fid_n_k5)) / (w_p * (1 - fid_n_k5) + w_n * fid_p_k5)

    # Sparsity

    def sparsity( edge_list_k, node_subset_k):
        #sparsity = 1 - ((len(edge_list_k[0]) + len(node_subset_k)) / (
        #            len(cora.data.x) + len(cora.data.edge_index[0])))
        sparsity = 1 - ( len(edge_list_k[0]) /  len(dataset.data.edge_index[0]))
        #num_edg_in_neigh = torch.count_nonzero(sub_adj)
        edges_in_neigh = 0
        for value in top_k.values:
            if value == 0.:
                break  # Exit the loop when zero is encountered
            edges_in_neigh += 1
        if edges_in_neigh == 0:
            sparsity_neig = 0
        else:
            sparsity_neig = 1 - (len(edge_list_k[0]) / edges_in_neigh)
        if torch.is_tensor(sparsity_neig):
            sparsity_neig = sparsity_neig.item()
        if sparsity_neig < 0:
            sparsity_neig = 0
        return sparsity, sparsity_neig
    spar_k3, spar_neig_k3 = sparsity(data_k3[1], data_k3[2])
    spar_k5, spar_neig_k5 = sparsity(data_k5[1], data_k5[2])


    def contrastivity(org_edge_mask, i):

        nodes, neig_edge_list, mapping, neig_edge_mask = k_hop_subgraph(i.item(), args.n_layers + 1, dataset.data.edge_index)

        explainer_cont = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='phenomenon',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',  # Model returns log probabilities.
            ),
            threshold_config=None  # ThresholdConfig("topk", 5),
        )

        y_i = dataset.data.y[i]
        lab = list(set(dataset.data.y.tolist()))
        print(lab)
        lab.remove(y_i)
        edge_masks_cont = []
        hamming_list_k3 = []
        hamming_list_k5 = []
        for labell in lab:
            targets = dataset.data.y.clone().detach()
            targets[i] = labell
            print(targets)
            print(targets[i])
            print(targets[i+1])

            explanation_cont = explainer_cont(dataset.data.x, dataset.data.edge_index, index=i, target=targets)  # norm or not (norm_edge_index)
            edge_masks_cont.append(explanation_cont.edge_mask)
            print(explanation_cont)
            print(explanation_cont.edge_mask)

            adjmatrix_cont = [[0 for i_row in range(len(dataset.data.y))] for i_column in range(len(dataset.data.y))]

            # scan the arrays edge_u and edge_v
            edge_i = 0
            for score in explanation_cont.edge_mask:
                u = dataset.data.edge_index[0][edge_i]
                v = dataset.data.edge_index[1][edge_i]
                edge_i = edge_i + 1
                adjmatrix_cont[u][v] = score
                print(adjmatrix_cont[u][v])
            #print(adjmatrix_cont)
            edge_mask_acc_cont = []

            row_item = 0
            for rowwww in range(len(adjmatrix)):
                row_item = row_item + 1
                for columnnnn in range(row_it):
                    importance_cont = adjmatrix_cont[rowwww][columnnnn] = adjmatrix_cont[columnnnn][rowwww] = (adjmatrix_cont[rowwww][columnnnn] +
                                                                                        adjmatrix_cont[columnnnn][rowwww]) / 2
            for ele in range(len(explanation.edge_index[0])):
                edge_mask_acc_cont.append(adjmatrix_cont[explanation_cont.edge_index[0][ele]][explanation_cont.edge_index[1][ele]])
            edge_mask_acc_cont = torch.tensor(edge_mask_acc_cont)
            # Extract the top-k nodes
            top_k_cont = torch.topk(edge_mask_acc_cont, len(dataset.data.edge_index[0]))


            """
            bool_mask_top3 = torch.zeros(10556).bool()
            bool_mask_top5 = torch.zeros(10556).bool()

            for item in range(6):
                bool_mask_top3[top_k_cont.indices[item].item()] = True
            for item in range(10):
                bool_mask_top5[top_k_cont.indices[item].item()] = True
            import scipy
            hamming_k3 = scipy.spatial.distance.hamming(data_k3[3].tolist(), bool_mask_top3.tolist()) * len(data_k3[3])
            hamming_k5 = scipy.spatial.distance.hamming(data_k5[3].tolist(), bool_mask_top5.tolist()) * len(data_k5[3])
            print(hamming_k3)
            print(hamming_k5)
            hamming_list_k3.append(hamming_k3)
            hamming_list_k5.append(hamming_k5)
            """

            # calculate Hamming distance for k=5
            aa_k3 = np.transpose(data_k3[1])
            aa_k3 = [tuple(lst) for lst in aa_k3]
            print(aa_k3)
            edges_subset_0_k3 = explanation_cont.edge_index[0][top_k_cont.indices[:6]]
            edges_subset_1_k3 = explanation_cont.edge_index[1][top_k_cont.indices[:6]]
            bb_k3 = np.transpose([edges_subset_0_k3.tolist(), edges_subset_1_k3.tolist()])
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

            # calculate Hamming distance for k=5
            aa_k5 = np.transpose(data_k5[1])
            aa_k5 = [tuple(lst) for lst in aa_k5]
            print(aa_k5)
            edges_subset_0_k5 = explanation_cont.edge_index[0][top_k_cont.indices[:10]]
            edges_subset_1_k5 = explanation_cont.edge_index[1][top_k_cont.indices[:10]]
            bb_k5 = np.transpose([edges_subset_0_k5.tolist(), edges_subset_1_k5.tolist()])
            bb_k5 = [tuple(lst) for lst in bb_k5]
            print(bb_k5)
            union_k5 = set(aa_k5 + bb_k5)
            print(union_k5)
            intersection_k5 = set(aa_k5).intersection(bb_k5)
            print(intersection_k5)
            # Contrastivity: 100% all edges are different, 0% the subgraph is the same
            resu_k5 = 1 - (len(intersection_k5)/len(aa_k5))
            print(resu_k5)

            hamming_list_k5.append(resu_k5)


        #calculate mean of hamming distance of all instances
        cont_k3 = sum(hamming_list_k3) / len(hamming_list_k3)
        cont_k5 = sum(hamming_list_k5) / len(hamming_list_k5)
        print(cont_k3)
        print(cont_k5)

        return cont_k3,cont_k5
        """
        print(edge_masks_cont)
        print(edge_masks_cont[0])
        edge_masks = torch.stack(edge_masks_cont)
        print(edge_masks)
        mean = torch.mean(edge_masks, axis=0)
        print(mean)
        print(mean.shape)

        import scipy.stats as stats
        rho, p_value = stats.spearmanr(org_edge_mask, mean)
        print(rho)

        org_edge_mask_neig = torch.masked_select(org_edge_mask, neig_edge_mask)
        cont_edge_mask_neig = torch.masked_select(mean, neig_edge_mask)
        rho_neig, p_value_neig = stats.spearmanr(org_edge_mask_neig, cont_edge_mask_neig)



        return rho, rho_neig
        """

    cont_k3, cont_k5 = contrastivity(explanation.edge_mask, i)

    def stability(i):
        hamming_list_k3_stab = []
        hamming_list_k5_stab = []

        while len(hamming_list_k5_stab) < 6:
            adj_attacked = torch.clone(adj)
            print(adj_attacked)
            """
            probability = 0.2
            random_matrix = torch.rand_like(adj_attacked)
            testscore = (random_matrix < probability).float() * (1. - adj_attacked) + (random_matrix >= probability).float() * adj_attacked
            """
            """
            row_item = 0
            import random

            # Generate all random probabilities outside the loop
            random_probs = [[random.random() for _ in range(len(adj_attacked))] for _ in range(len(adj_attacked))]

            for rowww in range(len(adj_attacked)):
                for columnnn in range(rowww + 1):  # Use rowww + 1 to avoid modifying diagonal elements
                    prob = random_probs[rowww][columnnn]
                    if adj_attacked[rowww][columnnn] == 1.:
                        if prob <= 0.1:
                            adj_attacked[rowww][columnnn] = adj_attacked[columnnn][rowww] = 0.
                    elif adj_attacked[rowww][columnnn] == 0.:
                        if prob <= 0.001:
                            adj_attacked[rowww][columnnn] = adj_attacked[columnnn][rowww] = 1.
            """

            """
            for rowww in range(len(adj_attacked)):
                row_item = row_item + 1
                for columnnn in range(row_item):
                    import random
                    prob = random.random()
                    if adj_attacked[rowww][columnnn] == 1.:
                        if prob <= 0.1:
                            adj_attacked[rowww][columnnn]  = adj_attacked[columnnn][rowww] = 0.
                    elif adj_attacked[rowww][columnnn] == 0.:
                        if prob <= 0.001:
                            adj_attacked[rowww][columnnn] = adj_attacked[columnnn][rowww] = 1.
            """
            """                                                                      
            for rowww in range(len(adj_attacked)):
                for columnnn in range(len(adj_attacked)):
                    import random
                    prob = random.random()
                    if adj_attacked[rowww][columnnn] == 1.:
                        if prob <= 0.1:
                            adj_attacked[rowww][columnnn] = 0.
                    elif adj_attacked[rowww][columnnn] == 0.:
                        if prob <= 0.01:
                            adj_attacked[rowww][columnnn] = 1.
            """

            #print(adj_attacked)

            #attacked_edge_index = dense_to_sparse(adj_attacked)
            #attacked_edge_index = attacked_edge_index[0]
            #print(attacked_edge_index)

            import random
            adj_ones = adj.nonzero()
            adj_zeros = (adj == 0).nonzero()
            print(len(adj_ones))
            ones_kept = random.sample(range(len(adj_ones)), int(len(adj_ones)*0.9))
            print(len(ones_kept))
            ones_kept =adj_ones[ones_kept]
            print(ones_kept)
            print(len(adj_zeros))
            ones_added = random.sample(range(len(adj_zeros)), int(len(adj_zeros)*0.001))
            print(len(ones_added))
            ones_added = adj_zeros[ones_added]
            print(ones_added)

            attacked_edge_index = torch.transpose(torch.cat((ones_kept, ones_added)),0,1)
            print(attacked_edge_index)
            """
            ones_mask = torch.full((1, len(adj_ones)), 0.2)
            zeros_mask = torch.full((1, len(adj_zeros)), 0.2)
            ones_idx = ones_mask.multinomial(num_samples=int(0.8 * len(adj_ones)), replacement=True)
            ones_kept = adj_ones[ones_idx]
            zeros_idx = zeros_mask.multinomial(num_samples=int(0.2 * len(adj_zeros)), replacement=True)
            ones_added = adj_zeros[zeros_idx]

            attacked_edge_index = torch.cat(ones_kept, ones_added)

            print(adj.nonzero())
            print((adj == 0).nonzero())
            """
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
                explainer_stab = Explainer(
                    model=model_stab,
                    algorithm=GNNExplainer(epochs=200),
                    explanation_type='model',
                    node_mask_type='object',
                    edge_mask_type='object',
                    model_config=dict(
                        mode='multiclass_classification',
                        task_level='node',
                        return_type='log_probs',  # Model returns log probabilities.
                    ),
                    threshold_config=None  # ThresholdConfig("topk", 5),
                )

                explanation_stab = explainer_stab(dataset.data.x, attacked_edge_index, index=i)
                """
                aa = torch.transpose(explanation.edge_index,0,1)
                bb = torch.transpose(explanation_stab.edge_index,0,1)
                print(aa)
                print(bb)
                xxx = set(map(frozenset, aa)).intersection(map(frozenset, bb))
                print(list(map(tuple, xxx)))
                quit()
                """

                adjmatrix_stab = [[0 for i_row in range(len(dataset.data.y))] for i_column in range(len(dataset.data.y))]

                # scan the arrays edge_u and edge_v
                edge_i = 0
                for score in explanation_stab.edge_mask:
                    u = attacked_edge_index[0][edge_i] #before cora.data.edge_index
                    v = attacked_edge_index[1][edge_i]
                    edge_i = edge_i + 1
                    adjmatrix_stab[u][v] = score
                #print(adjmatrix_stab)
                edge_mask_acc_stab = []

                row_item = 0
                for rowwww in range(len(adjmatrix)):
                    row_item = row_item + 1
                    for columnnnn in range(row_item):
                        importance_cont = adjmatrix_stab[rowwww][columnnnn] = adjmatrix_stab[columnnnn][rowwww] = (
                                                           adjmatrix_stab[rowwww][columnnnn] + adjmatrix_stab[columnnnn][rowwww]) / 2
                for ele in range(len(explanation.edge_index[0])):
                    edge_mask_acc_stab.append(
                        adjmatrix_stab[explanation_stab.edge_index[0][ele]][explanation_stab.edge_index[1][ele]])
                edge_mask_acc_stab = torch.tensor(edge_mask_acc_stab)
                # Extract the top-k nodes
                top_k_stab = torch.topk(edge_mask_acc_stab, len(dataset.data.edge_index[0]))
                print(top_k_stab.indices[:20])
                print(top_k_stab.values[:20])


                #calculate Hamming distance for k=3
                aa_k3 = np.transpose(data_k3[1])
                aa_k3 = [tuple(lst) for lst in aa_k3]
                print(aa_k3)
                edges_subset_0_k3 = explanation_stab.edge_index[0][top_k_stab.indices[:6]]
                edges_subset_1_k3 = explanation_stab.edge_index[1][top_k_stab.indices[:6]]
                bb_k3 = np.transpose([edges_subset_0_k3.tolist(),edges_subset_1_k3.tolist()])
                bb_k3 = [tuple(lst) for lst in bb_k3]
                print(bb_k3)
                union_k3 = set(aa_k3 + bb_k3)
                print(union_k3)
                intersection_k3 = set(aa_k3).intersection(bb_k3)
                print(intersection_k3)
                # Stability: 100%: exactly the same subgraph, 0% totally different edges selected
                resu_k3 = len(intersection_k3) / len(aa_k3)
                print(resu_k3)

                # calculate Hamming distance for k=5
                aa_k5 = np.transpose(data_k5[1])
                aa_k5 = [tuple(lst) for lst in aa_k5]
                print(aa_k5)
                edges_subset_0_k5 = explanation_stab.edge_index[0][top_k_stab.indices[:10]]
                edges_subset_1_k5 = explanation_stab.edge_index[1][top_k_stab.indices[:10]]
                bb_k5 = np.transpose([edges_subset_0_k5.tolist(), edges_subset_1_k5.tolist()])
                bb_k5 = [tuple(lst) for lst in bb_k5]
                print(bb_k5)
                union_k5 = set(aa_k5 + bb_k5)
                print(union_k5)
                intersection_k5 = set(aa_k5).intersection(bb_k5)
                print(intersection_k5)
                # Stability: 100%: exactly the same subgraph, 0% totally different edges selected
                resu_k5 = len(intersection_k5) / len(aa_k5)
                print(resu_k5)


                """
                bool_mask_top3 = torch.zeros(10556).bool()
                bool_mask_top5 = torch.zeros(10556).bool()

                for item in range(6):
                    bool_mask_top3[top_k_stab.indices[item].item()] = True
                for item in range(10):
                    bool_mask_top5[top_k_stab.indices[item].item()] = True
                import scipy
                hamming_k3_stab = scipy.spatial.distance.hamming(data_k3[3].tolist(), bool_mask_top3.tolist())
                hamming_k5_stab = scipy.spatial.distance.hamming(data_k5[3], bool_mask_top5)
                """
                hamming_list_k3_stab.append(resu_k3)
                hamming_list_k5_stab.append(resu_k5)

        # calculate mean of hamming distance of all instances
        stab_k3 = sum(hamming_list_k3_stab) / len(hamming_list_k3_stab)
        stab_k5 = sum(hamming_list_k5_stab) / len(hamming_list_k5_stab)
        print(stab_k3)
        print(stab_k5)

        return stab_k3, stab_k5

    stab_k3, stab_k5 = stability(i)





    """ Create Output File """

    colors = ["red", "orange", "lightblue", "green", "blue", "purple", "grey"]
    path = r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\GNNExplainer_experiments\{}_node{}_layers{}_gnnexp".format(
        args.dataset,
        i,
        args.n_layers)


    d = [edge_list[0], edge_list[1], top_k.values.tolist(), [spar_k3, spar_k5],[spar_neig_k3, spar_neig_k5],
         [fid_p_k3, fid_p_k5, fid_n_k3, fid_n_k5], [char_score_k3, char_score_k5], [cont_k3, cont_k5] , [stab_k3, stab_k5]]
    from itertools import zip_longest

    df = pd.DataFrame(zip_longest(*d, fillvalue=''),
                      columns=['Edge-source', 'Edge-target', 'Edge Importance','Sparsity', 'Sparsity-Neig', 'Fid 3+/5+/3-/5-', 'Char 3/5 ', 'Contrastivity nor/neig', 'Stability 3/5'])
    # df.rename(columns={'0':'Nodes', '1':'Edge source', '2':'Edge:target', '3':'Sparsity'}, inplace=True)
    print(df.head(15))
    df.to_csv(path + ".csv")

    """ Visualization of the Explanations """
    from visualization.subgraph_plotting import *

    def visualize_graph(k, nodes, edges, metrics):

        colors = ["red", "orange", "lightblue", "green", "blue", "purple", "grey"]
        path = r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\GNNExplainer_experiments\{}_node{}_layers{}_gnnexp_top{}".format(
            args.dataset,
            i,
            args.n_layers, k)

        prediction_labels = []
        prediction_prob = []
        for item in output.tolist():
            prediction_prob.append(max(item))
            prediction_labels.append(item.index(max(item)))
        draw_graph(node_index=nodes, edge_index=edges, y=dataset.data.y,
                   prediction=prediction_labels, colors=colors, path=path, label=metrics)

    visualize_graph(data_k3[0], data_k3[2], data_k3[1], ' Spar: ' + str(d[3][0]) + 'Spar-Neig: ' + str(d[4][0]) + ' Fid+: ' + str(d[5][0]) + ' Fid-: ' + str(d[5][2])
              + ' Char: ' + str(d[6][0]) + ' Contrast: ' + str(d[7][0]) + ' Stability: ' + str(d[8][0]))
    visualize_graph(data_k5[0], data_k5[2], data_k5[1], ' Spar: ' + str(d[3][1]) + 'Spar-Neig: ' + str(d[4][1])+ ' Fid+: ' + str(d[5][1]) + ' Fid-: ' + str(d[5][3])
              + ' Char: ' + str(d[6][1]) + ' Contrast: ' + str(d[7][1]) + ' Stability: ' + str(d[8][1]))

    instance_k3 = [i, spar_k3,spar_neig_k3,fid_p_k3, fid_n_k3,char_score_k3,cont_k3, stab_k3]
    instance_k5 = [i, spar_k5,spar_neig_k5,fid_p_k5, fid_n_k5,char_score_k5,cont_k5, stab_k5]
    results_table_k3.append(instance_k3)
    results_table_k5.append(instance_k5)

    df_results_k3 = pd.DataFrame(results_table_k3,
                      columns=[ 'Node','Sparsity', 'Sparsity-Neig','Fid3+' ,'Fid3-', 'Char3 ', 'Contrastivity', 'Stability'])
    # df.rename(columns={'0':'Nodes', '1':'Edge source', '2':'Edge:target', '3':'Sparsity'}, inplace=True)

    df_results_k3.to_csv(r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\GNNExplainer_experiments\aa_results_layers{}_gnnexp_top{}".format(
            args.n_layers, 3) + ".csv")

    df_results_k5 = pd.DataFrame(results_table_k5,
                                 columns=['Node', 'Sparsity', 'Sparsity-Neig', 'Fid5+', 'Fid5-', 'Char5 ', 'Contrastivity',
                                          'Stability'])
    # df.rename(columns={'0':'Nodes', '1':'Edge source', '2':'Edge:target', '3':'Sparsity'}, inplace=True)

    df_results_k5.to_csv(
        r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\GNNExplainer_experiments\aa_results_layers{}_gnnexp_top{}".format(
            args.n_layers, 5) + ".csv")

