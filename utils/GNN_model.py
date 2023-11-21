import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, global_add_pool

class GIN(torch.nn.Module):
    """GIN Model for Graph Neural Networks."""
    def __init__(self, dim_h, node_feature):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(node_feature, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))

        self.lin1 = Linear(dim_h*3, dim_h*2)
        self.lin2 = Linear(dim_h*2, dim_h*2)
        self.lin4 = Linear(dim_h*2, 2)   
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)
        
        # Classifier
        h = self.relu(self.lin1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.relu(self.lin2(h))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin4(h)
        K_prediction, mu_prediction = h[:, 0].squeeze(), h[:, 1].squeeze()
        return K_prediction, mu_prediction
