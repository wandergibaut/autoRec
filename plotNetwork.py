import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def show_graph_with_labels(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    plt.show()


def show_graph_no_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    d = nx.degree(gr)
    d = [(d[node] + 1) * 20 for node in gr.nodes()]
    nx.draw(gr, with_labels=False, node_size=d, node_color=d, widths=0.01)

    plt.show()

users_friend_matrix = np.genfromtxt('../lastFM/fooData/friends_matrix.dat',
                                    dtype=None,
                                    delimiter=' ')


show_graph_no_labels(users_friend_matrix)
