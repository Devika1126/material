import os
import pickle
from pathlib import Path
from src.data_loader import prepare_dataset


def main():
    # save splits at repository-level data/splits for easy access
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / 'data' / 'splits'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving splits to {out_dir}")
    train_graphs, train_targets, val_graphs, val_targets, test_graphs, test_targets = prepare_dataset(
        json_path=None,
        include_bandgap=False,
        split_ratios=(0.7, 0.15, 0.15),
        cutoff=5.0,
        random_seed=42,
    )

    # Attach targets as .y tensors to graphs for PyG compatibility
    import torch

    import math

    def attach(graphs, targets):
        out = []
        for g, t in zip(graphs, targets):
            # targets dict -> ordered list [volume, band_gap, formation_energy, density, bulk_modulus]
            vol = t.get('volume', 0.0)
            bg = t.get('band_gap', 0.0)
            fe = t.get('formation_energy', 0.0)
            den = t.get('density', 0.0)
            bm = t.get('bulk_modulus', 0.0)
            # mark optional properties as NaN if zero (commonly missing in Materials Project)
            if bg == 0.0:
                bg_val = float('nan')
            else:
                bg_val = float(bg)
            if bm == 0.0:
                bm_val = float('nan')
            else:
                bm_val = float(bm)
            vals = [float(vol), bg_val, float(fe), float(den), bm_val]
            g.y = torch.tensor(vals, dtype=torch.float32)
            out.append(g)
        return out

    train_graphs = attach(train_graphs, train_targets)
    val_graphs = attach(val_graphs, val_targets)
    test_graphs = attach(test_graphs, test_targets)

    with open(out_dir / 'train.pkl', 'wb') as f:
        pickle.dump({'graphs': train_graphs}, f)
    with open(out_dir / 'val.pkl', 'wb') as f:
        pickle.dump({'graphs': val_graphs}, f)
    with open(out_dir / 'test.pkl', 'wb') as f:
        pickle.dump({'graphs': test_graphs}, f)

    print(f"Prepared and saved splits: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")


if __name__ == '__main__':
    main()
