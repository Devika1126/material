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
    print(f"Loaded {len(graphs)} graphs from train.pkl")
    loader = DataLoader(graphs, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    print("batch.y:", getattr(batch, 'y', None))
    y = batch.y
    print('y shape:', None if y is None else tuple(y.shape))
    if y is not None:
        print('y unique:', torch.unique(y))
        # show per-column stats
        by = y
        if by.dim() == 1:
            by = by.view(1, -1)
        print('per-col mins:', by.min(dim=0).values)
        print('per-col maxs:', by.max(dim=0).values)
    model = CGCNN()
    model.eval()
    with torch.no_grad():
        out = model(batch)
    for k, v in out.items():
        print(f"pred {k} shape:", tuple(v.shape), 'sample:', v.view(-1)[:5])


if __name__ == '__main__':
    main()
