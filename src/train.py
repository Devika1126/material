import os
import argparse
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_materials_json
from src.featurizers import MaterialsDataset
from src.models.gnn import CGCNN
from torch.nn import HuberLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------------------
# Training & Evaluation Utils
# -------------------------
def train_epoch(model, loader, optimizer, device, scalers=None):
    model.train()
    losses = []
    props = ["volume", "band_gap", "formation_energy", "density", "bulk_modulus"]
    per_target_loss = {k: [] for k in props}
    per_target_coverage = {k: [] for k in props}

    huber_loss = HuberLoss(delta=1.0)  # Use Huber loss for robustness

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)

        # Ensure batch.y is present
        if not hasattr(batch, 'y') or batch.y is None:
            # Nothing to train on in this batch
            continue

        # Normalize shape: ensure [batch_size, 5]
        by = batch.y
        if by.dim() == 1:
            if by.numel() % 5 == 0:
                by = by.view(-1, 5)
            else:
                # fallback
                by = by.unsqueeze(0)
        elif by.dim() == 0:
            by = by.view(1, -1)
        batch_y = by

        loss = torch.tensor(0.0, device=device)
        added = 0

        # prepare scalers if provided (means,stds)
        means_t = None
        stds_t = None
        if scalers is not None:
            try:
                means_np, stds_np = scalers
                means_t = torch.tensor(means_np, device=device, dtype=torch.float32)
                stds_t = torch.tensor(stds_np, device=device, dtype=torch.float32)
            except Exception:
                means_t = None
                stds_t = None

        for i, k in enumerate(props):
            # target vector for this property: shape [batch_size]
            target_vec = batch_y[:, i].float()
            pred = out[k]
            pred_vec = pred.view(-1).float()

            # Make sure sizes align
            if pred_vec.shape[0] != target_vec.shape[0]:
                # Skip mismatched batch
                continue

            # Use finite check: NaN marks missing values in saved splits
            mask = torch.isfinite(target_vec)
            coverage = int(mask.sum().item())
            per_target_coverage[k].append(coverage)
            if coverage > 0:
                # Compute loss on masked entries
                masked_pred = pred_vec[mask]
                masked_target = target_vec[mask]
                # If scalers provided, standardize both pred and target for stable loss
                if means_t is not None and stds_t is not None:
                    m = means_t[i]
                    s = stds_t[i]
                    # avoid division by zero
                    if s == 0:
                        s = 1.0
                    masked_pred_s = (masked_pred - m) / s
                    masked_target_s = (masked_target - m) / s
                    prop_loss = huber_loss(masked_pred_s, masked_target_s)
                else:
                    prop_loss = huber_loss(masked_pred, masked_target)
                # Defensive checks for NaN/Inf
                if not torch.isfinite(prop_loss):
                    print(f"WARNING: prop_loss is not finite for prop={k}. Skipping this prop for this batch.")
                    print(f"  masked_pred: {masked_pred[:5]}")
                    print(f"  masked_target: {masked_target[:5]}")
                    per_target_loss[k].append(0.0)
                    continue
                loss = loss + prop_loss
                per_target_loss[k].append(prop_loss.item())
                added += 1
            else:
                per_target_loss[k].append(0.0)

        # If no target had any coverage, skip backward to avoid zero tensor issues
        if added == 0:
            continue

        loss.backward()
        # gradient clipping to avoid exploding gradients
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        except Exception:
            pass
        optimizer.step()
        losses.append(loss.item())
    # After epoch, print per-target mean loss and coverage
    print("Per-target mean loss (masked):")
    for k in props:
        mean_loss = np.mean(per_target_loss[k]) if len(per_target_loss[k]) > 0 else 0.0
        mean_coverage = np.mean(per_target_coverage[k]) if len(per_target_coverage[k]) > 0 else 0.0
        print(f"  {k}: loss={mean_loss:.4f}, coverage={mean_coverage:.1f} samples per batch")
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    return mean_loss, losses

def eval_epoch(model, loader, device, return_preds=False):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            # Collect predictions and targets; ensure shapes align
            pred_cols = [out[k].cpu().numpy().reshape(-1, 1) for k in ["volume", "band_gap", "formation_energy", "density", "bulk_modulus"]]
            batch_preds = np.concatenate(pred_cols, axis=1)
            if not hasattr(batch, 'y') or batch.y is None:
                continue
            by = batch.y
            # reshape flattened y (e.g., shape [batch_size*5]) into [batch_size,5]
            if by.dim() == 1 and by.numel() % 5 == 0:
                by = by.view(-1, 5)
            elif by.dim() == 0:
                by = by.view(1, -1)
            batch_targets = by.cpu().numpy()
            # Only add if shapes match and batch is not empty
            if batch_preds.shape[0] == batch_targets.shape[0] and batch_preds.shape[0] > 0:
                preds.append(batch_preds)
                targets.append(batch_targets)
    if len(preds) == 0 or len(targets) == 0:
        # empty result
        if return_preds:
            return [0]*5, [0]*5, np.zeros((0,5)), np.zeros((0,5))
        return 0.0, 0.0
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    maes = []
    rmses = []
    # compute per-target metrics using only finite (non-NaN) pairs
    for i in range(5):
        pred_col = preds[:, i]
        targ_col = targets[:, i]
        mask = np.isfinite(pred_col) & np.isfinite(targ_col)
        if mask.sum() == 0:
            maes.append(float('nan'))
            rmses.append(float('nan'))
        else:
            maes.append(mean_absolute_error(targ_col[mask], pred_col[mask]))
            rmses.append(np.sqrt(mean_squared_error(targ_col[mask], pred_col[mask])))
    if return_preds:
        return maes, rmses, preds, targets
    # return averaged scalar metrics for progress logging
    return float(np.mean(maes)), float(np.mean(rmses))

# -------------------------
# Ensemble Training
# -------------------------
def train_ensemble(args, dataset, device, n_ensemble=5, scalers=None):
    """
    Train an ensemble of CGCNN models.
    """
    n = len(dataset)
    print(f"DEBUG: Total dataset size: {n}")
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    train_idx = idxs[:int(0.7*n)]
    val_idx = idxs[int(0.7*n):int(0.85*n)]
    test_idx = idxs[int(0.85*n):]
    print(f"DEBUG: Train split: {len(train_idx)}, Val split: {len(val_idx)}, Test split: {len(test_idx)}")
    # dataset may be a list of Data objects or an object with .get(i)
    def get_item(d, i):
        try:
            return d.get(i)
        except Exception:
            return d[i]

    train_set = [get_item(dataset, i) for i in train_idx]
    val_set = [get_item(dataset, i) for i in val_idx]
    test_set = [get_item(dataset, i) for i in test_idx]
    if len(train_set) > 0:
        t0 = getattr(train_set[0], 'y', 'No y')
        print(f"DEBUG: Sample train target: {t0}")
    if len(val_set) > 0:
        v0 = getattr(val_set[0], 'y', 'No y')
        print(f"DEBUG: Sample val target: {v0}")
    if len(test_set) > 0:
        te0 = getattr(test_set[0], 'y', 'No y')
        print(f"DEBUG: Sample test target: {te0}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    maes_all, rmses_all = [], []
    for i in range(n_ensemble):
        print(f"\n=== Training ensemble model {i+1}/{n_ensemble} ===")
        model = CGCNN().to(device)
        # initialize head biases to training means if provided
        if scalers is not None:
            means, stds = scalers
            for j, k in enumerate(["volume", "band_gap", "formation_energy", "density", "bulk_modulus"]):
                try:
                    model.heads[k].bias.data.fill_(means[j])
                except Exception:
                    pass
        optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)  # Reduced learning rate
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # Learning rate scheduler
        best_val_mae = float('inf')

        train_losses = []
        for epoch in range(args.epochs):
            train_loss, batch_losses = train_epoch(model, train_loader, optimizer, device, scalers=scalers)
            train_losses.append(train_loss)
            val_mae, val_rmse = eval_epoch(model, val_loader, device)
            scheduler.step(val_mae)  # Adjust learning rate based on validation MAE
            print(f"Model {i+1} Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val MAE={val_mae:.4f}, Val RMSE={val_rmse:.4f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), f"cgcnn_{i}.pt")

        # Gradient clipping adjustment
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Increased gradient clipping
        except Exception:
            pass

        # Plot training loss curve
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Curve (Model {i+1})')
        plt.legend()
        plt.savefig(f'train_loss_curve_{i+1}.png')
        plt.close()

        # Test set predictions vs actual
        test_maes, test_rmses, preds, targets = eval_epoch(model, test_loader, device, return_preds=True)
        maes_all.append(test_maes)
        rmses_all.append(test_rmses)
        print(f"Model {i+1} Test MAE: {[f'{m:.4f}' for m in test_maes]}")
        print(f"Model {i+1} Test RMSE: {[f'{r:.4f}' for r in test_rmses]}")
        os.makedirs("results", exist_ok=True)
        # save scalers for later un-scaling predictions
        if scalers is not None:
            import pickle
            with open(os.path.join('results', f'scalers_{i}.pkl'), 'wb') as f:
                pickle.dump(scalers, f)
        props = ["volume", "band_gap", "formation_energy", "density", "bulk_modulus"]
        for j, prop in enumerate(props):
            if preds.size == 0 or targets.size == 0:
                print(f"Skipping plot for {prop} because preds/targets empty")
                continue
            plt.figure()
            plt.scatter(targets[:, j], preds[:, j], alpha=0.6)
            mn, mx = targets[:, j].min(), targets[:, j].max()
            plt.plot([mn, mx], [mn, mx], 'r--', label='Ideal')
            plt.xlabel(f'Actual {prop}')
            plt.ylabel(f'Predicted {prop}')
            plt.title(f'Predicted vs Actual {prop} (Model {i+1})')
            plt.legend()
            plt.savefig(f'results/pred_vs_actual_{prop}_{i+1}.png')
            plt.close()

    print("\n=== Ensemble Results ===")
    props = ["volume", "band_gap", "formation_energy", "density", "bulk_modulus"]
    for j, prop in enumerate(props):
        maes = [maes_all[i][j] for i in range(n_ensemble)]
        rmses = [rmses_all[i][j] for i in range(n_ensemble)]
        print(f"{prop}: Ensemble Test MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
        print(f"{prop}: Ensemble Test RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

# -------------------------
# Main Script
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train CGCNN on materials data")
    parser.add_argument("--dataset_path", type=str, default="data/materials.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--checkpoint", type=str, default="cgcnn.pt")
    parser.add_argument("--ensemble", action="store_true", help="Train ensemble of models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    materials = load_materials_json(args.dataset_path)
    print(f"DEBUG: Loaded {len(materials)} materials from JSON.")
    if len(materials) > 0:
        print(f"DEBUG: Sample material: {materials[0]}")

    # Print and visualize target coverage stats
    import collections
    props = ["volume", "band_gap", "formation_energy", "density", "bulk_modulus"]
    stats = collections.defaultdict(lambda: [0, 0])  # {prop: [non-missing, non-zero]}
    for entry in materials:
        for i, prop in enumerate(props):
            val = entry.get(prop, None)
            if val is not None:
                stats[prop][0] += 1
                if isinstance(val, dict):
                    val = val.get('value', 0.0)
                if val != 0.0:
                    stats[prop][1] += 1
    print("\n==============================")
    print(" Target Coverage Summary ")
    print("==============================")
    print("| Property         | Non-missing | Non-zero |")
    print("|------------------|-------------|----------|")
    for prop in props:
        print(f"| {prop:<16} | {stats[prop][0]:>11} | {stats[prop][1]:>8} |")
    print("\nExplanation:")
    print("- Non-missing: Number of samples with a value for this property.")
    print("- Non-zero: Number of samples with a non-zero value (usable for training).\n")

    # Visualize distributions
    import matplotlib.pyplot as plt
    for prop in props:
        values = [entry.get(prop, 0.0) for entry in materials if entry.get(prop, None) is not None]
        values = [v.get('value', v) if isinstance(v, dict) else v for v in values]
        plt.figure()
        plt.hist(values, bins=50, alpha=0.7)
        plt.title(f"Distribution of {prop}")
        plt.xlabel(prop)
        plt.ylabel("Count")
        plt.savefig(f"results/{prop}_distribution.png")
        plt.close()

    # ...existing code for dataset creation and training...

if __name__ == "__main__":
    main()
