from pymatgen.core import Lattice, Structure
from src.featurizers import structure_to_graph


def test_structure_to_graph_minimal():
    lattice = Lattice.cubic(5.0)
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    species = ["Na", "Cl"]
    struct = Structure(lattice, species, coords)
    data = structure_to_graph(struct, cutoff=3.0)
    assert data.x.shape[0] == 2
    # edge_index may be empty for this small cutoff, but shapes should be consistent
    assert hasattr(data, 'edge_index')
    assert hasattr(data, 'edge_attr')
