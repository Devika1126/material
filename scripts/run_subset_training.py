"""Run a small training job on data/subset_materials.json and print logs.

Usage:
  python scripts/run_subset_training.py --subset data/subset_materials.json --epochs 1 --batch_size 4
"""
import argparse
from src.data_loader import prepare_dataset
from src.train import train_ensemble
import torch


def attach_y(graphs, targets):
    import torch as _t
    out = []
    for g, t in zip(graphs, targets):
        y = [t.get('volume', 0.0), t.get('band_gap', 0.0), t.get('formation_energy', 0.0), t.get('density', 0.0), t.get('bulk_modulus', 0.0)]
        g.y = _t.tensor(y, dtype=_t.float32)
        out.append(g)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='data/subset_materials.json')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    print(f"Preparing dataset from {args.subset}...")
    train_graphs, train_targets, val_graphs, val_targets, test_graphs, test_targets = prepare_dataset(
        json_path=args.subset, include_bandgap=True, split_ratios=(0.7, 0.15, 0.15), cutoff=5.0, random_seed=42
    )
    print(f"Prepared splits: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")

    train_list = attach_y(train_graphs, train_targets)
    val_list = attach_y(val_graphs, val_targets)
    test_list = attach_y(test_graphs, test_targets)

    class Args:
        pass

    args_run = Args()
    args_run.epochs = args.epochs
    args_run.batch_size = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training on device {device} for {args_run.epochs} epochs")
    train_ensemble(args_run, train_list + val_list + test_list, device, n_ensemble=1)


if __name__ == '__main__':
    main()
