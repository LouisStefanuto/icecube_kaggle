import torch

from torch.nn import Linear, LeakyReLU
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class GNN_Sage(torch.nn.Module):
    def __init__(self, num_node_features, dim_output, dropout_rate=0.):
        super(GNN_Sage, self).__init__()
        
        torch.manual_seed(12345)
        self.num_node_features = num_node_features
        self.dim_output = dim_output
        self.dropout_rate = dropout_rate
        
        # feature extraction
        self.conv1 = SAGEConv(self.num_node_features, 256)
        self.conv2 = SAGEConv(256, 256)
        self.conv3 = SAGEConv(256, 256)
        
        # final regressor
        self.lin1 = Linear(256, 64)       
        self.lin2 = Linear(64, self.dim_output)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]        
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin1(x)
        x = x.tanh()
        x = self.lin2(x)
        
        return x