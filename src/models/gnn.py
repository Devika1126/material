import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch_geometric.nn import CGConv, global_mean_pool

logger = logging.getLogger(__name__)
if not logger.handlers:
    # basic config for standalone runs
    logging.basicConfig(level=logging.INFO)

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
        # normalization for stability
        self.norm = nn.LayerNorm(hidden_dim)
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
        # defensive NaN handling
        if not torch.isfinite(x).all():
            logger.warning("NaN or Inf detected after conv1; applying nan_to_num")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        x = self.node_proj(x)
        x = self.norm(x)
        x = F.relu(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            if not torch.isfinite(x).all():
                logger.warning("NaN or Inf detected after a convolution layer; applying nan_to_num")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            # clamp to avoid extremely large activations
            x = torch.clamp(x, min=-1e6, max=1e6)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        if not torch.isfinite(x).all():
            logger.warning("NaN or Inf detected after global_mean_pool; applying nan_to_num")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        # final clamp
        x = torch.clamp(x, min=-1e6, max=1e6)
        # Multi-task outputs
        out = {k: self.heads[k](x) for k in self.heads}
        for k, v in out.items():
            if not torch.isfinite(v).all():
                logger.warning(f"NaN or Inf detected in output head {k}; applying nan_to_num")
                out[k] = torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
        return out
