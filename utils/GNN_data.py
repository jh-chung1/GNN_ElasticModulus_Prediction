import torch
import pickle
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, random_split
from itertools import permutations

def load_data(data_dir):
    with open(data_dir, 'rb') as f: 
        return pickle.load(f)
    
def Elastic_property_avg_scheme(data_list, scheme):
    new_dataset = []
    for data in data_list:
        new_data = data.clone()
        if scheme == 'Voigt':
            new_data.K = new_data.K_Voigt
            new_data.mu = new_data.mu_Voigt
            del new_data.K_Voigt, new_data.mu_Voigt
        elif scheme == 'Hill':
            new_data.K = new_data.K_Hill
            new_data.mu = new_data.mu_Hill
            del new_data.K_Hill, new_data.mu_Hill
        new_dataset.append(new_data)
    return new_dataset 

def is_undirected(data):
    # Extract the edge index
    edge_index = data.edge_index
    
    # Create a set to store unique edges
    edge_set = set()
    for i in range(edge_index.shape[1]):
        # Get nodes from edge index
        node_a = edge_index[0, i].item()
        node_b = edge_index[1, i].item()

        # Check if the reverse edge exists
        if (node_b, node_a) not in edge_set:
            edge_set.add((node_a, node_b))
        else:
            edge_set.remove((node_b, node_a))
            
    # If all edges had their reverse, the edge_set should be empty
    return len(edge_set) == 0

def make_undirected(data):
    # Extract edge index
    edge_index = data.edge_index

    # Convert edge_index to a set of tuples for O(1) look-up time
    edge_set = {(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.shape[1])}

    # Identify missing reverse edges
    missing_edges = [(b, a) for a, b in edge_set if (b, a) not in edge_set]

    # Convert lists of missing edges to tensor format and add them to the original edge_index
    missing_edges_tensor = torch.tensor(missing_edges, dtype=torch.long).t()
    data.edge_index = torch.cat([edge_index, missing_edges_tensor], dim=1)
    
    return data

def normalize_spatial_info(data_list):
    for data in data_list:
        max_val = torch.round(torch.max(data.x[:,:3])).item()+1  # Compute max value for current graph
        data.x[:,:6] /= max_val                   # Normalize the first 6 columns
        data.x[:,6] /= (max_val**2)               # Normalize the 7th column
    return data_list

def augment_dataset(data_list):
    all_augmented_data = []

    for data in data_list:
        augmented_data_list = [data.clone()]  # Clone the original data for each permutation

        # Get all permutations of indices [0, 1, 2] for spatial coordinates
        for perm in permutations([0, 1, 2]):
            new_data = data.clone()

            # Switch coordinates x, y, z
            new_data.x[:, :3] = data.x[:, list(perm)]

            # Switch dimensions for x, y, z
            new_data.x[:, 3:6] = data.x[:, [i + 3 for i in perm]]

            # Append augmented data to the list
            augmented_data_list.append(new_data)

        all_augmented_data.extend(augmented_data_list)

    return all_augmented_data

def prepare_dataset(data_list, batch_size, train_percentage=0.80, test_percentage=0.1):
    dataset_size = len(data_list)
    train_size = int(train_percentage * dataset_size)
    test_size = int(test_percentage * dataset_size)
    valid_size = dataset_size - train_size - test_size

    train_set, test_set, valid_set = random_split(data_list, [train_size, test_size, valid_size])
    
    print(f'Number of training graphs: {len(train_set)}')
    print(f'Number of test graphs: {len(test_set)}')
    print(f'Number of vali graphs: {len(valid_set)}')
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, valid_loader
