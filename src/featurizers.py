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
    return np.array([
        el.Z,
        el.X if el.X else 0.0,
        el.atomic_radius if el.atomic_radius else 0.0
    ], dtype=np.float32)

def rbf_expand(dist, D_min=0, D_max=8, N=32, gamma=4):
    """
    Radial basis expansion for edge distances.
    """
    centers = np.linspace(D_min, D_max, N)
    return np.exp(-gamma * (dist - centers) ** 2)

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
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float32)
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
