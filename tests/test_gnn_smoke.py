import torch
from pymatgen.core import Lattice, Structure
from src.featurizers import structure_to_graph
from src.models.gnn import CGCNN


def test_cgcnn_forward_minimal():
    lattice = Lattice.cubic(5.0)
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    species = ["Na", "Cl"]
    struct = Structure(lattice, species, coords)
    data = structure_to_graph(struct, cutoff=3.0)
    model = CGCNN()
    model.eval()
    with torch.no_grad():
        out = model(data)
    # Expect keys and shapes
    for k, v in out.items():
        assert isinstance(v, torch.Tensor)
        # batch size should be 1 for this single graph
        assert v.shape[0] == 1
        assert torch.isfinite(v).all()
