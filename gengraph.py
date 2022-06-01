import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as colors
plt.switch_backend("agg")
import networkx as nx
import numpy as np
import synthetic_structsim
import featgen
import pdb

def perturb(graph_list, p):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list

""" Synthetic Graph #1:

    Start with Barabasi-Albert graph and attach house-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  number of edges to attach to existing node (for BA graph)

    Returns:
        G                 :  A networkx graph
        role_id           :  A list with length equal to number of nodes in the entire graph (basis
                          :  + shapes). role_id[i] is the ID of the role of node i. It is the label.
        name              :  A graph identifier
    """
def generate_graph(basis_type="ba", 
                   shape="house", 
                   nb_shapes=80, 
                   width_basis=300, 
                   feature_generator=None, 
                   m=5, 
                   random_edges=0.0):

    if shape == "house":
        list_shapes = [["house"]] * nb_shapes
    elif shape == "cycle":
        list_shapes = [["cycle", 6]] * nb_shapes
    elif shape == "diamond":
        list_shapes = [["diamond"]] * nb_shapes
    elif shape == "grid":
        list_shapes = [["grid"]] * nb_shapes
    else:
        assert False
    G, role_id, _ = synthetic_structsim.build_graph(width_basis, 
                                                    basis_type, 
                                                    list_shapes, 
                                                    rdm_basis_plugins=True, 
                                                    start=0, 
                                                    m=m)
                                            
    if random_edges != 0:
        G = perturb([G], random_edges)[0]
    feature_generator.gen_node_features(G)
    return G, role_id