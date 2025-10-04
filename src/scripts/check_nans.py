import pickle
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader

from src.models.gnn import CGCNN


def main():
    splits_dir = Path('data') / 'splits'
    with open(splits_dir / 'train.pkl', 'rb') as f:
        train = pickle.load(f)
    graphs = train['graphs']
    loader = DataLoader(graphs, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    print('Checking batch inputs for NaN/Inf...')
    x = batch.x
    edge_attr = getattr(batch, 'edge_attr', None)
    print('x shape:', None if x is None else tuple(x.shape))
    print('edge_attr shape:', None if edge_attr is None else tuple(edge_attr.shape))
    print('x has NaN:', torch.isnan(x).any().item())
    if edge_attr is not None:
        print('edge_attr has NaN:', torch.isnan(edge_attr).any().item())

    model = CGCNN()
    model.eval()
    with torch.no_grad():
        out = model(batch)

    print('Checking model outputs for NaN/Inf:')
    for k, v in out.items():
        print(f"{k}: shape={tuple(v.shape)}, has_nan={torch.isnan(v).any().item()}, has_inf={torch.isinf(v).any().item()}")


if __name__ == '__main__':
    main()
