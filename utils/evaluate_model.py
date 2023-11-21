import matplotlib.pyplot as plt
import pickle
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score
import numpy as np
from .GNN_data import make_undirected

def visualize_performance(y_K, pred_K, R2_K, acc_K, y_mu, pred_mu, R2_mu, acc_mu, rock, cover_interval, overlap, excluded_subcube, save_dir):
    plt.rcParams['font.size'] = 15
    # Plotting for K
    fig, ax = plt.subplots(dpi=300, figsize=(5,5))   

    ax.plot(y_K, pred_K, 'bo', markersize=3, label=f'$R^2 = %.2f$' %R2_K)
    ax.axline((0, 0), slope=1, color='red', linestyle='--')
    ax.set_ylabel('Predicted $K$ [GPa]')
    ax.set_xlabel('Ground Truth $K$ [GPa]')

    ticks = [10, 20, 30, 40] 
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim([0, max(ticks)])
    ax.set_ylim([0, max(ticks)])
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/Test_{rock}_excluded_subcube_{excluded_subcube}_R2_K.png')
    
    print('Accuracy for K', np.mean(acc_K))

    # Plotting for mu
    fig, ax = plt.subplots(dpi=300, figsize=(5,5))

    ax.plot(y_mu, pred_mu, 'bo', markersize=3, label=f'$R^2 = %.2f$' %R2_mu)
    ax.axline((0, 0), slope=1, color='red', linestyle='--')
    ax.set_ylabel('Predicted $\mu$ [GPa]')
    ax.set_xlabel('Ground Truth $\mu$ [GPa]')

    ticks = [10, 20, 30, 40, 50] #, 60] #[10, 20, 30, 40, 50]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim([0, max(ticks)])
    ax.set_ylim([0, max(ticks)])
    ax.legend()
    plt.tight_layout()    
    plt.savefig(f'{save_dir}/Test_{rock}_excluded_subcube_{excluded_subcube}_R2_mu.png')
    
    print('Accuracy for mu', np.mean(acc_mu))
    
def load_graph_data(data_dir, avg_scheme, batch_size, val_percentage):
    """
    function for loading graph data
    """
    with open(data_dir, 'rb') as f: 
        data_list = pickle.load(f)

    new_dataset = []
    for data in data_list:
        new_data = data.clone()
        if avg_scheme == 'Voigt':
            new_data.K = new_data.K_Voigt
            new_data.mu = new_data.mu_Voigt
            del new_data.K_Voigt, new_data.mu_Voigt
        elif avg_scheme == 'Hill':
            new_data.K = new_data.K_Hill
            new_data.mu = new_data.mu_Hill
            del new_data.K_Hill, new_data.mu_Hill
        new_dataset.append(new_data)
    data_list = new_dataset 
    # Convert all graphs in data_list to undirected
    data_list = [make_undirected(data) for data in data_list]
    
    for data in data_list:
        max_val = torch.round(torch.max(data.x[:,:3])).item()+1  # Compute max value for current graph
        data.x[:,:6] /= max_val                   # Normalize the first 6 columns
        #data.x[:,6] /= (max_val**2)               # Normalize the 7th column
        #print(max_val)
    
    dataset_size = len(data_list)
    valid_percentage = val_percentage
    valid_size = int(valid_percentage*dataset_size)
    valid_set, remain_set = random_split(data_list, [valid_size, dataset_size - valid_size])
    dataset = valid_set
    print(f'Number of test graphs: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader    
    
def evaluate_model(model, loader, device, rock, cover_interval, overlap, excluded_subcube, save_dir):
    # Evaluate the model on the test set
    model.eval()

    K_pred_whole = []
    mu_pred_whole = []

    y_whole_K = []
    y_whole_mu = []
    test_acc_K = []
    test_acc_mu = []

    with torch.no_grad():
        for data in loader:
            K_pred, mu_pred = model(data.x.to(device), data.edge_index.long().to(device), data.batch.to(device))
            data.K = data.K.to(torch.float32)
            data.mu = data.mu.to(torch.float32)        

            acc_K = 100 - torch.abs((data.K.to(device) - K_pred)/data.K.to(device)) * 100
            acc_mu = 100 - torch.abs((data.mu.to(device) - mu_pred)/data.mu.to(device)) * 100

            K_pred_whole.append(K_pred.cpu().numpy())
            mu_pred_whole.append(mu_pred.cpu().numpy())
            y_whole_K.append(data.K.cpu().numpy())
            y_whole_mu.append(data.mu.cpu().numpy())
            test_acc_K.append(acc_K.detach().cpu().numpy())
            test_acc_mu.append(acc_mu.detach().cpu().numpy())

    K_pred_whole = np.concatenate(K_pred_whole, axis=0)
    mu_pred_whole = np.concatenate(mu_pred_whole, axis=0)
    y_whole_K = np.concatenate(y_whole_K, axis=0)  
    y_whole_mu = np.concatenate(y_whole_mu, axis=0)  
    R2_K = r2_score(y_whole_K, K_pred_whole)
    R2_mu = r2_score(y_whole_mu, mu_pred_whole)

    acc_K_total = np.concatenate(test_acc_K, axis=0)
    acc_mu_total = np.concatenate(test_acc_mu, axis=0)

    print(f'Test MSE for K: {R2_K} for mu: {R2_mu}, for avg: {(R2_K+ R2_mu)/2}')

    # Return values if needed later
    visualize_performance(y_whole_K, K_pred_whole, R2_K, acc_K_total, y_whole_mu, mu_pred_whole, R2_mu, acc_mu_total, rock, cover_interval, overlap, excluded_subcube, save_dir)
    return R2_K, R2_mu