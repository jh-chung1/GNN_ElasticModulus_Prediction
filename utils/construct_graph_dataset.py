import numpy as np
import networkx as nx
#from numba import jit  # Uncomment this line if you use @jit

# ----------- functions for graph dataset construction-------------------------

# Extract node features from a single cluster
#@jit(nopython=True)
def extract_nodes_from_cluster(cluster, is_solid=True):
    coords = np.array(cluster)  # Convert to numpy array   
    center_coords = np.mean(coords, axis=0)  # calculate center coordinate
    ellip_radii = np.max(coords[:,0])-np.min(coords[:,0]), np.max(coords[:,1])-np.min(coords[:,1]), np.max(coords[:,2])- np.min(coords[:,2])    
    number_point = len(coords)  # number of point in the cluster       
    node_type = np.array([1]) if is_solid else np.array([2])
    return np.concatenate([center_coords, ellip_radii, [number_point], node_type])

# Iterate through cluster_list and extract node features
def extract_all_node_features(cluster_list, is_solid=True):
    all_nodes = []
    for i in range(len(cluster_list)):
        for j in range(len(cluster_list[i])):
            node = extract_nodes_from_cluster(cluster_list[i][j], is_solid)
            all_nodes.append(node)
    return np.array(all_nodes)

# Extract edges from overlaps
def extract_edges_from_overlaps(overlaps, offset):
    return [(a+offset, b+offset) for a, b in overlaps]

def compute_graph_topological_features(G, nodes):
    """
    Calculate graph topological features and update node features.

    Parameters:
    - G: A networkx graph.
    - nodes: A numpy array of original node features.

    Returns:
    - nodes: A numpy array with the original and new features concatenated.
    """
    
    # calculate node degree and centrality measures
    degree = dict(nx.degree(G))   
    closeness_centrality = nx.closeness_centrality(G)

    # Handle PowerIterationFailedConvergence error for eigenvector_centrality
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G)
    except PowerIterationFailedConvergence:
        eigenvector_centrality = {i: -1 for i in range(len(nodes))}

    pagerank = nx.pagerank(G)

    # update nodes features with the new calculated features
    nodes_new_features = np.zeros((nodes.shape[0], 4))  # 4 new features
    for i in range(nodes.shape[0]):
        nodes_new_features[i, 0] = degree[i]
        nodes_new_features[i, 1] = closeness_centrality[i]
        nodes_new_features[i, 2] = eigenvector_centrality[i]
        nodes_new_features[i, 3] = pagerank[i]


    # concatenate old features with new features
    nodes = np.concatenate([nodes, nodes_new_features], axis=1)
    
    return nodes

# Load stiffness label
def load_stiffness_and_porosity(stiff_file, idx):
    """
    Load stiffness matrix and porosity from the file for the given index (idx).

    Parameters:
    - stiff_file (str): path to the file containing stiffness matrix and porosity data.
    - idx (int): index of the data entry to extract.

    Returns:
    - stiffness (np.ndarray): stiffness matrix of shape (6, 6).
    - porosity (float): porosity value.
    """
    with open(stiff_file, 'r') as f:
        lines = f.readlines()

    stiffness = np.array([list(map(float, line.split())) for line in lines[2 + 10 * (idx - 1):8 + 10 * (idx - 1)]])

    return stiffness