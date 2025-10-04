import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool

class MultiTaskCGCNN(nn.Module):
    """
    CGCNN with multi-task heads for volume, band gap, formation energy, density, bulk modulus.
    """
    def __init__(self, node_dim=3, edge_dim=32, hidden_dim=128, num_layers=3, out_dims=None):
        super().__init__()
        if out_dims is None:
            out_dims = {
                "volume": 1,
                "band_gap": 1,
                "formation_energy": 1,
                "density": 1,
                "bulk_modulus": 1
            }
        self.conv1 = CGConv(node_dim, edge_dim, aggr='add')
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([
            CGConv(hidden_dim, edge_dim, aggr='add') for _ in range(num_layers - 1)
        ])
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # Multi-task heads
        self.heads = nn.ModuleDict({
            k: nn.Linear(hidden_dim, v) for k, v in out_dims.items()
        })

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(self.node_proj(x))
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        # Multi-task outputs
        out = {k: head(x) for k, head in self.heads.items()}
        return out
