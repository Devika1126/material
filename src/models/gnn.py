import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool

class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network baseline.
    """
    def __init__(self, node_dim=3, edge_dim=32, hidden_dim=128, num_layers=3):
        super().__init__()
        self.conv1 = CGConv(node_dim, edge_dim, aggr='add')
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([
            CGConv(hidden_dim, edge_dim, aggr='add') for _ in range(num_layers - 1)
        ])
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # Multi-task output heads
        self.heads = nn.ModuleDict({
            "volume": nn.Linear(hidden_dim, 1),
            "band_gap": nn.Linear(hidden_dim, 1),
            "formation_energy": nn.Linear(hidden_dim, 1),
            "density": nn.Linear(hidden_dim, 1),
            "bulk_modulus": nn.Linear(hidden_dim, 1)
        })

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        # Log the range of values after conv1
        print(f"Range after conv1: min={x.min().item()}, max={x.max().item()}")
        if not torch.isfinite(x).all():
            print("WARNING: NaN or Inf detected after conv1")

        x = F.relu(self.node_proj(x))
        # Log the range of values after node_proj
        print(f"Range after node_proj: min={x.min().item()}, max={x.max().item()}")
        if not torch.isfinite(x).all():
            print("WARNING: NaN or Inf detected after node_proj")

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            # Log the range of values after each convolution layer
            print(f"Range after convolution layer: min={x.min().item()}, max={x.max().item()}")
            if not torch.isfinite(x).all():
                print("WARNING: NaN or Inf detected after a convolution layer")
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        # Log the range of values after global_mean_pool
        print(f"Range after global_mean_pool: min={x.min().item()}, max={x.max().item()}")
        if not torch.isfinite(x).all():
            print("WARNING: NaN or Inf detected after global_mean_pool")

        x = F.relu(self.fc1(x))
        # Log the range of values after fc1
        print(f"Range after fc1: min={x.min().item()}, max={x.max().item()}")
        if not torch.isfinite(x).all():
            print("WARNING: NaN or Inf detected after fc1")
        # Multi-task outputs
        out = {k: self.heads[k](x) for k in self.heads}
        for k, v in out.items():
            if not torch.isfinite(v).all():
                print(f"WARNING: NaN or Inf detected in output head {k}")
        return out
