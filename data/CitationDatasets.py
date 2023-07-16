import graphviz
from torch_geometric.datasets import Planetoid
#from torch_geometric.transforms import NormalizeFeatures, to_undirected
import torch_geometric.transforms as T
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

import numpy as np
import pandas as pd
import csv

"""
Data Import
"""

transform = T.Compose([T.ToUndirected(), T.NormalizeFeatures()])

def get_cora():
    cora = Planetoid(root='data/Planetoid', name='Cora', transform=transform)
    return cora

def get_pubmed():
    pubmed = Planetoid(root='data/Planetoid', name='Cora', transform=transform)
    return pubmed

def get_citeseer():
    citeseer = Planetoid(root='data/Planetoid', name='CiteSeer', transform=transform)
    return citeseer


"""
This Method creates a csv-file with all edges, composed of source and target node
"""
def create_csv2(data):
    edge_list = [data.edge_index[0].tolist(),data.edge_index[1].tolist()]
    #print(edge_list)
    edge_list_trans = tuple(zip(*edge_list))
    #print(edge_list_trans)
    with open('edges.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(edge_list_trans)


def plot_k_hop_subgraph(node, hops, dataset_edge_index, y, colors):
    nodes, edge_list, mapping, edge_mask = k_hop_subgraph(node, hops, dataset_edge_index)

    subg = graphviz.Digraph('node' + str(node) + '_' + str(hops) + 'hop', comment='Neighborhood-Subgraph')
    print(str(hops)+ "-Subgraph of Node " + str(node) + " has " + str(len(nodes)) + " nodes and " + str(len(edge_list[0])) + " edges.")
    for node in nodes.tolist():
        subg.node(str(node), color=colors[y[node]])
    strings0 = []
    for ele in edge_list[0].tolist():
        strings0.append(str(ele))
    strings1 = []
    for ele in edge_list[1].tolist():
        strings1.append(str(ele))

    edge_list = [strings0, strings1]
    edge_list_trans = tuple(zip(*edge_list))
    subg.edges(edge_list_trans)
    subg.render(directory=r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\Experiments")

    return nodes, edge_list
