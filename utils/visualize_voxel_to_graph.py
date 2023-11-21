from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_voxels(data, ax):
    """
    Plot voxels in 3D space using an efficient method.

    Parameters:
    - data: 3D numpy array representing voxel data.
    - ax: Matplotlib axis for 3D plotting.
    """
    # Create a boolean array: True for cells with a voxel, False for empty cells
    voxels = data > 0

    # Set the colors for the voxels
    colors = np.empty(voxels.shape, dtype=object)
    colors[voxels] = 'lightgray'

    # Use the voxels method for efficient rendering
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Voxel Visualization')    
    
def plot_cluster(points, ax, color):
    # Convert list of tuples to a 2D list
    points = [list(p) for p in points]

    # Separate x, y, and z coordinates
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]

    # Repeat the color for all points 
    colors_for_points = [color for _ in points]

    # Scatter plot
    ax.scatter(x_coords, y_coords, z_coords, s=0.5, alpha=0.5, c=colors_for_points, marker='o', depthshade=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Clusters')      
    
def plot_3d_graph(graph, nodes, ax):
    """
    Plot a graph in 3D using the nodes' x, y, z coordinates.

    Parameters:
    - graph: NetworkX graph object.
    - nodes: Numpy array of node features. The first three columns should be x, y, z coordinates.
    - ax: Matplotlib axis for 3D plotting.
    """
    # Extract x, y, z coordinates from nodes
    x_coords = nodes[:, 0]
    y_coords = nodes[:, 1]
    z_coords = nodes[:, 2]

    # Determine which nodes are solid and which are pore
    # Assuming the last column indicates if a node is solid (1 for solid, 0 for pore)
    is_solid = nodes[:, 7] == 1 

    # Plot solid nodes (gray)
    ax.scatter(x_coords[is_solid], y_coords[is_solid], z_coords[is_solid], color='gray', s=50, alpha=0.5)

    # Plot pore nodes (black)
    ax.scatter(x_coords[~is_solid], y_coords[~is_solid], z_coords[~is_solid], color='black', s=50, alpha=0.5)

    # Plot edges with transparency
    for edge in graph.edges():
        x = np.array([x_coords[edge[0]], x_coords[edge[1]]])
        y = np.array([y_coords[edge[0]], y_coords[edge[1]]])
        z = np.array([z_coords[edge[0]], z_coords[edge[1]]])
        ax.plot(x, y, z, color='grey', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Graph Visualization')
    