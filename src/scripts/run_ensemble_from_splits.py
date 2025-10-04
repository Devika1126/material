import os
import pickle
import argparse
import torch
from pathlib import Path

from src.train import train_ensemble


def load_splits(splits_dir: Path):
    with open(splits_dir / 'train.pkl', 'rb') as f:
        train = pickle.load(f)
    with open(splits_dir / 'val.pkl', 'rb') as f:
        val = pickle.load(f)
    with open(splits_dir / 'test.pkl', 'rb') as f:
        test = pickle.load(f)
    # Create a unified 'dataset' as list of Data objects for now (train+val+test)
    dataset = []
    dataset.extend(train['graphs'])
    dataset.extend(val['graphs'])
    dataset.extend(test['graphs'])

    # compute per-target mean/std from train graphs (ignore NaNs)
    import numpy as np
    ys = []
    for g in train['graphs']:
        y = g.y.numpy()
        ys.append(y)
    ys = np.vstack(ys)
    means = np.nanmean(ys, axis=0)
    stds = np.nanstd(ys, axis=0)
    stds[stds == 0] = 1.0
    return dataset, (means, stds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits_dir', type=str, default=os.path.join('data', 'splits'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--ensemble', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits_dir = Path(args.splits_dir)
    dataset, (means, stds) = load_splits(splits_dir)

    # Build a simple args-like object for train_ensemble
    class A:
        pass

    a = A()
    a.epochs = args.epochs
    a.batch_size = args.batch_size
    a.lr = args.lr
    a.means = means
    a.stds = stds

    train_ensemble(a, dataset, device, n_ensemble=args.ensemble)


if __name__ == '__main__':
    main()
