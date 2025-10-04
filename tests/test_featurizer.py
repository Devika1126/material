from pymatgen.core import Structure, Lattice
from src.featurizers import structure_to_graph


def test_structure_to_graph():
    # Dummy cubic structure built from lattice + species + coords
    lattice = Lattice.cubic(5.64)
    species = ["Na", "Cl"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    structure = Structure(lattice, species, coords)
    data = structure_to_graph(structure)
    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert hasattr(data, "edge_attr")
    assert data.x.shape[1] == 3
