import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.GNN_data import *
from utils.GNN_model import GIN
from utils.train_model import train_model
from utils.evaluate_model import *

# --------------------------------------
# Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('--Rock', type=str, default='B1_B2_FB1_FB2', help='rock image in dataset')
parser.add_argument('--Unseen_Rock', type=str, default='CG', help='unseen rock image')
parser.add_argument('--batch_num', type=int, default=64, help='batch size') 
parser.add_argument('--epoch_num', type=int, default=200, help='Epoch number') 
parser.add_argument('--avg_scheme', choices=['Voigt', 'Hill'], default='Voigt')
parser.add_argument('--cover_interval', type=int, default=20, help='Cover interval')
parser.add_argument('--overlap', type=float, default=0.3, help='cover overlap')
parser.add_argument('--excluded_subcube', type=str, default='150', help='excluded subcube')
parser.add_argument('--unseen_subcube_size', type=str, default='90_100_180', help='excluded subcube size')
parser.add_argument('--save_model_dir', type=str, default='./examples/saved_GNN_model', help='saved GNN model directory')
args = parser.parse_args()

batch_size = args.batch_num 
epoch_num = args.epoch_num 
overlap=args.overlap
learning_rate = 0.005

# combined 4 rocks (B1, B2, FB1, FB2) with regular interval
data_dir = f'./examples/graph_data/{args.Rock}_DFS_CoverInterval_{args.cover_interval}_Overlap_{overlap}_except_subcube{args.excluded_subcube}.pkl'

# -------------------------------------------
# Load data and perform data pre-processing
# Load data
data_list = load_data(data_dir)

# Apply the selected material averaging scheme to the dataset
data_list = Elastic_property_avg_scheme(data_list, args.avg_scheme)

# Check if the graph data is directed or undirected. For this study, undirected edges are used.
all_undirected_before = all(is_undirected(data) for data in data_list)
print(f'Graphs are undirected (before conversion): {all_undirected_before}')  # True if all are undirected, False otherwise

# Convert all graphs in the dataset to undirected
data_list = [make_undirected(data) for data in data_list]

# Re-check if all graphs are now undirected
all_undirected_after = all(is_undirected(data) for data in data_list)
print(f'Graphs are undirected (after conversion): {all_undirected_after}')  # True if all are undirected, False otherwise

# Normalize the spatial features in nodes
data_list = normalize_spatial_info(data_list)

# Augment data
data_list = augment_dataset(data_list)

# Split dataset for train/validation/test
train_loader, test_loader, valid_loader = prepare_dataset(data_list, batch_size, train_percentage=0.80, test_percentage=0.1)

# ----------------------------------------------
# Define GNN architecture
# Define the model, optimizer, and loss function
no_node_feature = data_list[0].x.shape[1]
model = GIN(dim_h=16, node_feature=no_node_feature)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.MSELoss()
print(model)

# Define a learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# Use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters:", num_params)

# ------------------------------------------------
# train model
train_losses, test_losses, R2_trainings, R2_tests, train_acc_total, test_acc_total, best_state_dict = train_model(
    model, train_loader, test_loader, criterion, optimizer, scheduler, device=device, num_epochs=epoch_num)

# 1) Epoch vs R^2
plt.figure(dpi=300, figsize=(5,5)) 

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 15

plt.plot(np.array(train_losses), label='Train')
plt.plot(np.array(test_losses), label='Validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel(r'MSE Loss ($\mathcal{L})$')
plt.yscale("log") 
plt.xlim(-5,args.epoch_num)
plt.tight_layout() 
plt.savefig(f'{args.save_model_dir}/epoch_{args.epoch_num}.png')

# ------------------------------------------------
# save model
best_model_path = f'{args.save_model_dir}/epoch_{args.epoch_num}.pt'  
torch.save(best_state_dict, best_model_path)

# ------------------------------------------------
# Evaluate model for the test dataset
R2_K, R2_mu = evaluate_model(model, valid_loader, device, args.Rock, args.cover_interval, overlap, args.excluded_subcube, args.save_model_dir)

# ------------------------------------------------
# Evaluate model for the unseen rock dataset (CG sandstone)
# Load unseen rock
unseen_rock_data_dir = f'./examples/graph_data/{args.Unseen_Rock}_DFS_CoverInterval_{args.cover_interval}_Overlap_{overlap}_except_subcube{args.excluded_subcube}.pkl'
print(f'unseen rock data dir: {unseen_rock_data_dir}') 
# Load data & Apply the selected material averaging scheme to the dataset
unseen_rock_data_list = Elastic_property_avg_scheme(load_data(unseen_rock_data_dir), args.avg_scheme)
# Convert all graphs in the dataset to undirected
unseen_rock_data_list = [make_undirected(data) for data in unseen_rock_data_list]
# Normalize the spatial features in nodes
unseen_rock_data_list = normalize_spatial_info(unseen_rock_data_list)
# Split dataset for train/validation/test
unseen_rock_train_loader, unseen_rock_test_loader, unseen_rock_valid_loader = prepare_dataset(unseen_rock_data_list, batch_size, train_percentage=0.01, test_percentage=0.01)
R2_K, R2_mu = evaluate_model(model, unseen_rock_valid_loader, device, args.Unseen_Rock, args.cover_interval, args.overlap, args.excluded_subcube, args.save_model_dir)

# -------------------------------------------------
# Evaluate model for the unseen subcube size dataset (subcube size 150) 
# subcube size for the training dataset: 90, 100, 180
# Load unseen rock
unseen_subcube_data_dir = f'./examples/graph_data/{args.Rock}_DFS_CoverInterval_{args.cover_interval}_Overlap_{overlap}_except_subcube{args.unseen_subcube_size}.pkl'
print(f'unseen rock data dir: {unseen_subcube_data_dir}') 
# Load data & Apply the selected material averaging scheme to the dataset
unseen_subcube_data_list = Elastic_property_avg_scheme(load_data(unseen_subcube_data_dir), args.avg_scheme)
# Convert all graphs in the dataset to undirected
unseen_subcube_data_list = [make_undirected(data) for data in unseen_subcube_data_list]
# # Normalize the spatial features in nodes
unseen_subcube_data_list = normalize_spatial_info(unseen_subcube_data_list)
# # Split dataset for train/validation/test
unseen_subcube_train_loader, unseen_subcube_test_loader, unseen_subcube_valid_loader = prepare_dataset(unseen_subcube_data_list, batch_size, train_percentage=0.01, test_percentage=0.01)
R2_K, R2_mu = evaluate_model(model, unseen_subcube_valid_loader, device, args.Unseen_Rock, args.cover_interval, args.overlap, args.unseen_subcube_size, args.save_model_dir)

print('training_prediction_completed!')
