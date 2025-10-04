import pytest
import torch
from torch_geometric.data import Data
from src.models import MODEL_ZOO


def _make_dummy_data():
    x = torch.rand(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.rand(4, 32)
    batch = torch.zeros(4, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


def test_forward_pass():
    dummy_data = _make_dummy_data()
    for name, model_cls in MODEL_ZOO.items():
        model = model_cls()
        out = model(dummy_data)
        assert isinstance(out, dict)
        for k in ["volume", "band_gap", "formation_energy", "density", "bulk_modulus"]:
            assert k in out
