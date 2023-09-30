import graphviz

def draw_graph(node_index, edge_index, y, prediction, colors, path, label):
    subg = graphviz.Digraph(path, comment='Neighborhood-Subgraph')
    subg.attr(label=label)
    for node in node_index:
        subg.node(str(node), fontcolor=colors[prediction[node]], color=colors[y[node]])
    strings0 = []
    for ele in edge_index[0]:
        strings0.append(str(ele))
    strings1 = []
    for ele in edge_index[1]:
        strings1.append(str(ele))

    edge_list = [strings0, strings1]
    edge_list_trans = tuple(zip(*edge_list))
    """
    edge_score = edge_score.tolist()
    print(len(edge_score))
    print(len(edge_list_trans))
    for ele in range(len(edge_list_trans)):
        print(round(edge_score[ele], 1))
        subg.edge(edge_list_trans[ele][0], edge_list_trans[ele][1],
                  color="0.000 0.000 0.000 " + str(edge_score[ele]))
    """
    subg.edges(edge_list_trans)
    subg.render(directory=r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\CF-GNN Experiments")


def draw_graph_without_imp_nodes(node_index, edge_index, y, prediction, prediction_index, colors, path):
    subg = graphviz.Digraph(path, comment='Neighborhood-Subgraph')

    for index in range(len(node_index)):
        subg.node(str(node_index[index]), fontcolor=colors[prediction[prediction_index[index]]],
                  color=colors[y[node_index[index]]])
    strings0 = []
    for ele in edge_index[0]:
        strings0.append(str(ele))
    strings1 = []
    for ele in edge_index[1]:
        strings1.append(str(ele))

    edge_list = [strings0, strings1]
    edge_list_trans = tuple(zip(*edge_list))
    subg.edges(edge_list_trans)
    subg.render(directory=r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\Experiments")

