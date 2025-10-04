import numpy as np
from pymatgen.core.structure import Structure
from torch_geometric.data import Data, Dataset
from typing import List, Dict, Any
from pymatgen.core.periodic_table import Element

def get_atom_features(site) -> np.ndarray:
    """
    Returns atomic features: atomic number, electronegativity, atomic radius.
    """
    el = Element(site.specie.symbol)
    # Scale features to reasonable ranges to help model stability
    # atomic number up to ~100 -> scale to ~0-1
    z = float(el.Z) / 100.0
    # electronegativity (Pauling) roughly 0-4 -> scale to 0-1
    en = float(el.X) / 4.0 if el.X is not None else 0.0
    # atomic radius in angstroms -> scale by 5.0 as a conservative upper bound
    ar = float(el.atomic_radius) / 5.0 if el.atomic_radius is not None else 0.0
    arr = np.array([z, en, ar], dtype=np.float32)
    # defensive sanitization
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)
    return arr

def rbf_expand(dist, D_min=0, D_max=8, N=32, gamma=4):
    """
    Radial basis expansion for edge distances.
    """
    centers = np.linspace(D_min, D_max, N)
    out = np.exp(-gamma * (dist - centers) ** 2)
    # ensure dtype float32 and sanitize
    out = np.array(out, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)
    return out

def structure_to_graph(structure: Structure, cutoff: float = 5.0) -> Data:
    """
    Converts pymatgen Structure to torch_geometric Data object.
    """
    atom_features = [get_atom_features(site) for site in structure.sites]
    atom_features = np.stack(atom_features)
    edge_index = []
    edge_attr = []
    for i, site_i in enumerate(structure.sites):
        for j, site_j in enumerate(structure.sites):
            if i == j:
                continue
            dist = structure.get_distance(i, j)
            if dist <= cutoff:
                edge_index.append([i, j])
                edge_attr.append(rbf_expand(dist))
    import torch
    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 32), dtype=torch.float32)
    else:
        edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)
        # Ensure edge_attr is a well-shaped float32 array
        edge_attr = np.array(edge_attr, dtype=np.float32)
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.reshape(-1, 1)
        # If RBF produced unexpected size, pad or trim to 32
        if edge_attr.shape[1] != 32:
            desired = 32
            current = edge_attr.shape[1]
            if current < desired:
                pad = np.zeros((edge_attr.shape[0], desired - current), dtype=np.float32)
                edge_attr = np.concatenate([edge_attr, pad], axis=1)
            else:
                edge_attr = edge_attr[:, :desired]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    data = Data(
        x = torch.tensor(atom_features, dtype=torch.float32),
        edge_index = edge_index,
        edge_attr = edge_attr
    )
    return data

class MaterialsDataset(Dataset):
    """
    PyTorch Geometric Dataset for materials.
    """
    def __init__(self, materials: List[Dict], cutoff: float = 5.0):
        super().__init__()
        self.materials = materials
        self.cutoff = cutoff

    def len(self):
        return len(self.materials)

    def get(self, idx):
        import torch
        entry = self.materials[idx]
        structure = Structure.from_dict(entry['structure'])
        data = structure_to_graph(structure, cutoff=self.cutoff)
        # Multi-task targets: volume, band_gap, formation_energy, density, bulk_modulus
        # Extract bulk_modulus value if dict
        bm = entry.get('bulk_modulus', 0.0)
        if bm is None:
            bm = 0.0
        elif isinstance(bm, dict):
            bm = bm.get('value', 0.0)
        targets = [
            float(entry.get('volume', 0.0)),
            float(entry.get('band_gap', 0.0)),
            float(entry.get('formation_energy', 0.0)),
            float(entry.get('density', 0.0)),
            float(bm)
        ]
        data.y = torch.tensor(targets, dtype=torch.float32)
        data.material_id = entry['material_id']
        return data
