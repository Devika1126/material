import argparse
import os
import pickle
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from src.models.gnn import CGCNN
from src.train import train_ensemble, eval_epoch

def load_splits(split_path):
    """Load train, val, and test splits from the specified path."""
    with open(os.path.join(split_path, 'train.pkl'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(split_path, 'val.pkl'), 'rb') as f:
        val = pickle.load(f)
    with open(os.path.join(split_path, 'test.pkl'), 'rb') as f:
        test = pickle.load(f)
    return train, val, test

def compute_scalers(train):
    """Compute per-target mean and std from the training set."""
    ys = [g.y.numpy() for g in train['graphs']]
    ys = np.vstack(ys)
    means = np.nanmean(ys, axis=0)
    stds = np.nanstd(ys, axis=0)
    stds[stds == 0] = 1.0  # Avoid division by zero
    return means, stds

def save_summary(metrics, output_path):
    """Save the final test metrics as a JSON file."""
    import json
    with open(os.path.join(output_path, 'summary.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

def main(args):
    # Load splits
    train, val, test = load_splits(args.split_path)

    # Compute scalers
    means, stds = compute_scalers(train)

    # Prepare data loaders
    train_loader = DataLoader(train['graphs'], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val['graphs'], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test['graphs'], batch_size=args.batch_size, shuffle=False)

    # Train ensemble
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    os.makedirs('results', exist_ok=True)
    with open('results/scalers.pkl', 'wb') as f:
        pickle.dump((means, stds), f)

    for i in range(args.ensemble):
        print(f"\n=== Training ensemble model {i+1}/{args.ensemble} ===")
        model = CGCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_losses = []
        val_metrics = []

        for epoch in range(args.epochs):
            train_loss, _ = train_ensemble(model, train_loader, optimizer, device, scalers=(means, stds))
            train_losses.append(train_loss)

            val_mae, val_rmse = eval_epoch(model, val_loader, device)
            val_metrics.append((val_mae, val_rmse))
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val MAE={val_mae:.4f}, Val RMSE={val_rmse:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f'results/cgcnn_member{i}.pt')

    # Evaluate ensemble on test set
    test_maes, test_rmses = [], []
    for i in range(args.ensemble):
        model = CGCNN().to(device)
        model.load_state_dict(torch.load(f'results/cgcnn_member{i}.pt'))
        mae, rmse = eval_epoch(model, test_loader, device)
        test_maes.append(mae)
        test_rmses.append(rmse)

    # Compute ensemble metrics
    ensemble_metrics = {
        'mae': np.mean(test_maes, axis=0).tolist(),
        'rmse': np.mean(test_rmses, axis=0).tolist(),
    }
    save_summary(ensemble_metrics, 'results')

    # Print final report
    print("\n=== Final Test Metrics ===")
    for prop, (mae, rmse) in enumerate(zip(ensemble_metrics['mae'], ensemble_metrics['rmse'])):
        print(f"Property {prop}: Test MAE={mae:.4f}, Test RMSE={rmse:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run stable training for CGCNN ensemble.")
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--ensemble', type=int, default=3, help='Number of ensemble members.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--split_path', type=str, required=True, help='Path to the directory containing train.pkl, val.pkl, and test.pkl.')
    args = parser.parse_args()
    main(args)
