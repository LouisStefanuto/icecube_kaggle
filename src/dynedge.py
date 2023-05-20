import torch

from torch.nn import Linear, LeakyReLU
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class EdgeConvMLP(torch.nn.Module):
    """Basic convolutional block."""
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()

        self.sequential = torch.nn.Sequential(
            Linear(dim_in, dim_hidden),
            LeakyReLU(),
            Linear(dim_hidden, dim_out),
            LeakyReLU(),
        )

    def forward(self, x):
        return self.sequential(x)


class DynEdge(torch.nn.Module):
    """Dynedge model from https://iopscience.iop.org/article/10.1088/1748-0221/17/11/P11003)"""
    def __init__(self, num_node_features, dim_output, dropout_rate=0.):
        super(DynEdge, self).__init__()
        
        torch.manual_seed(12345)
        self.num_node_features = num_node_features
        self.dim_output = dim_output
        self.dropout_rate = dropout_rate
        self.K = 8
        self.aggrs_list = ['max']

        self.conv1 = DynamicEdgeConv(nn=EdgeConvMLP(2 * self.num_node_features, 336, 256), k=self.K)
        self.conv2 = DynamicEdgeConv(nn=EdgeConvMLP(512, 336, 256), k=self.K)
        self.conv3 = DynamicEdgeConv(nn=EdgeConvMLP(512, 336, 256), k=self.K)
        self.conv4 = DynamicEdgeConv(nn=EdgeConvMLP(512, 336, 256), k=self.K)
        
        # final regressor
        self.mlp1 = torch.nn.Sequential(
            Linear(256 * 4 + self.num_node_features, 336),
            LeakyReLU(),
            Linear(336, 256),
            LeakyReLU(),
        )

#         self.global_pool =  MultiAggregation(aggrs=self.aggrs_list)

        self.mlp2 =  torch.nn.Sequential(
            Linear(len(self.aggrs_list) * 256, 128), # input depends of number of aggregating fns
            LeakyReLU(),
            Linear(128, self.dim_output)
        )

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings at various embedding depths
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)

        x = torch.cat([x, x1, x2, x3, x4], dim=-1)
        
        x = self.mlp1(x)

        # 2. Pooling        
#         x = self.global_pool(x, batch)
        x = global_mean_pool(x, batch)

        # 3. Apply a final MLP regressor
        x = self.mlp2(x)
        
        return x