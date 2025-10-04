import os
import json
import random
import math
from typing import List, Dict, Optional, Tuple
from pymatgen.core.structure import Structure
import numpy as np

from torch_geometric.data import Data
from src.featurizers import structure_to_graph

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_materials_json(path: Optional[str] = None) -> List[Dict]:
    """Load materials from JSON file. Defaults to data/materials.json.

    Args:
        path: Optional path to JSON file.

    Returns:
        List of material dicts.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "materials.json")
    with open(path, 'r') as f:
        return json.load(f)


def _get_value(entry: Dict, key: str) -> float:
    """Helper: extract numeric value, handle dict wrappers and None."""
    val = entry.get(key, None)
    if val is None:
        return 0.0
    if isinstance(val, dict):
        return float(val.get('value', 0.0))
    try:
        return float(val)
    except Exception:
        return 0.0


def prepare_dataset(json_path: str = None,
                    include_bandgap: bool = False,
                    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                    cutoff: float = 5.0,
                    random_seed: Optional[int] = 42):
    """Prepare PyG datasets and targets from materials JSON.

    Filtering rules:
    - Required targets: 'volume', 'formation_energy', 'density' (must be non-missing and numeric).
    - Optionally include 'band_gap' if include_bandgap is True (zero allowed but recorded).
    - Exclude 'bulk_modulus'.

    Returns:
        train_graphs, train_targets, val_graphs, val_targets, test_graphs, test_targets

    Each target is a dict: {"volume": ..., "formation_energy": ..., "density": ..., "band_gap": ...}
    """
    materials = load_materials_json(json_path)
    if random_seed is not None:
        random.seed(random_seed)

    props_required = ['volume', 'formation_energy', 'density']
    props_optional = ['band_gap'] if include_bandgap else []

    processed = []
    skipped_invalid_targets = 0
    skipped_structure_errors = 0
    for entry in materials:
        # Extract required targets and skip if missing/invalid (zero is allowed if numeric)
        vals = {}
        skip = False
        for k in props_required:
            v = _get_value(entry, k)
            if v == 0.0 or v is None or math.isnan(v) or math.isinf(v):
                # treat 0.0, NaN, or Inf as invalid for required targets (skip)
                skip = True
                break
            vals[k] = v
        if skip:
            skipped_invalid_targets += 1
            continue

        # Band gap handling
        if 'band_gap' in props_optional:
            bg = _get_value(entry, 'band_gap')
            if math.isnan(bg) or math.isinf(bg):
                bg = 0.0  # Replace invalid band gap values with 0.0
            vals['band_gap'] = bg
        else:
            bg = _get_value(entry, 'band_gap')
            if math.isnan(bg) or math.isinf(bg):
                bg = 0.0
            vals['band_gap'] = bg  # still include in dict (may be 0.0)

        # Convert structure to graph
        try:
            struct = Structure.from_dict(entry['structure'])
            graph: Data = structure_to_graph(struct, cutoff=cutoff)
            # Validate graph
            if graph.x is None or graph.edge_index is None or graph.edge_attr is None:
                raise ValueError("Invalid graph structure")
        except Exception as e:
            # Skip entries with malformed structure
            skipped_structure_errors += 1
            print(f"WARNING: skipping entry {entry.get('material_id', '')} due to structure error: {e}")
            continue

        processed.append({'graph': graph, 'targets': vals, 'material_id': entry.get('material_id')})

    # Log skipped entries
    print(f"Skipped {skipped_invalid_targets} entries due to invalid targets.")
    print(f"Skipped {skipped_structure_errors} entries due to structure errors.")

    # Log the range of values for each property in the dataset
    print("Logging target ranges before splitting:")
    for prop in ['volume', 'band_gap', 'formation_energy', 'density']:
        all_values = [t[prop] for t in processed if prop in t and np.isfinite(t[prop])]
        if all_values:
            print(f"{prop}: min={min(all_values)}, max={max(all_values)}, mean={np.mean(all_values):.4f}, std={np.std(all_values):.4f}")
        else:
            print(f"{prop}: No valid values found.")

    # Shuffle and split
    random.shuffle(processed)
    n = len(processed)
    if n == 0:
        raise ValueError('No valid materials after filtering. Check your JSON and filtering rules.')
    t_ratio, v_ratio, test_ratio = split_ratios
    if abs(t_ratio + v_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError('split_ratios must sum to 1.0')
    t_end = int(n * t_ratio)
    v_end = t_end + int(n * v_ratio)

    train = processed[:t_end]
    val = processed[t_end:v_end]
    test = processed[v_end:]

    def unpack(list_of):
        graphs = [it['graph'] for it in list_of]
        targets = [it['targets'] for it in list_of]
        return graphs, targets

    train_graphs, train_targets = unpack(train)
    val_graphs, val_targets = unpack(val)
    test_graphs, test_targets = unpack(test)

    # Compute per-property mean/std on valid targets
    def compute_scalers(targets):
        props = ['volume', 'band_gap', 'formation_energy', 'density']
        means = []
        stds = []
        for prop in props:
            values = [t[prop] for t in targets if np.isfinite(t[prop])]
            means.append(np.mean(values))
            stds.append(np.std(values))
        return means, stds

    # Exclude bulk_modulus and clip targets
    clip_ranges = {
        "volume": (None, 3000),
        "band_gap": (None, 20),
        "formation_energy": (-10, 10),
        "density": (None, 20),
    }
    for target in [train_targets, val_targets, test_targets]:
        for t in target:
            for prop, (min_val, max_val) in clip_ranges.items():
                if min_val is not None:
                    t[prop] = max(t[prop], min_val)
                if max_val is not None:
                    t[prop] = min(t[prop], max_val)

    return train_graphs, train_targets, val_graphs, val_targets, test_graphs, test_targets


def cli():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare dataset from materials.json')
    parser.add_argument('--json_path', type=str, default=None, help='Path to materials.json')
    parser.add_argument('--include_bandgap', action='store_true', help='Include band_gap in targets')
    parser.add_argument('--split_ratios', type=float, nargs=3, default=(0.7,0.15,0.15), help='Train/Val/Test ratios')
    parser.add_argument('--cutoff', type=float, default=5.0, help='Cutoff for structure_to_graph')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    train_graphs, train_targets, val_graphs, val_targets, test_graphs, test_targets = prepare_dataset(
        json_path=args.json_path,
        include_bandgap=args.include_bandgap,
        split_ratios=tuple(args.split_ratios),
        cutoff=args.cutoff,
        random_seed=args.seed
    )
    print(f"Prepared dataset: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")
    return train_graphs, train_targets, val_graphs, val_targets, test_graphs, test_targets


if __name__ == '__main__':
    cli()
