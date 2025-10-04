# Training script for multi-task and model zoo
import argparse
import torch
from torch_geometric.loader import DataLoader
from models import MODEL_ZOO
from models.ensemble import EnsembleModel

# ...existing data loading and featurization imports...

def train_epoch(model, loader, optimizer, device, loss_fns):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        # Multi-task loss
        loss = sum([loss_fns[k](outputs[k], batch.y[k].view(-1, 1)) for k in outputs])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

# ...validation, test, ensemble training functions...

def main():
    parser = argparse.ArgumentParser(description="Train GNN on materials data")
    parser.add_argument("--model", type=str, default="cgcnn", choices=list(MODEL_ZOO.keys()))
    parser.add_argument("--ensemble", action="store_true")
    # ...other args...
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ...data loading...
    model_cls = MODEL_ZOO[args.model]
    # ...prepare dataset, loaders...
    loss_fns = {k: torch.nn.L1Loss() for k in ["volume", "band_gap", "formation_energy", "density", "bulk_modulus"]}
    if args.ensemble:
        # ...ensemble training...
        pass
    else:
        model = model_cls().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(50):
            train_loss = train_epoch(model, train_loader, optimizer, device, loss_fns)
            # ...validation, checkpointing...

if __name__ == "__main__":
    main()
