import math
from src.data_loader import prepare_dataset, load_materials_json


def test_prepare_dataset_basic():
    # Use default materials.json in data/
    train_graphs, train_targets, val_graphs, val_targets, test_graphs, test_targets = prepare_dataset(json_path=None, include_bandgap=False, split_ratios=(0.7,0.15,0.15), random_seed=123)

    total = len(train_graphs) + len(val_graphs) + len(test_graphs)
    data = load_materials_json()
    assert total <= len(data)

    # Check ratios roughly match (within 2 samples tolerance)
    n = total
    assert abs(len(train_graphs) - int(0.7 * n)) <= 2
    assert abs(len(val_graphs) - int(0.15 * n)) <= 2
    assert abs(len(test_graphs) - int(0.15 * n)) <= 2

    # Check alignment and non-missing numeric targets
    for graphs, targets in [(train_graphs, train_targets), (val_graphs, val_targets), (test_graphs, test_targets)]:
        assert len(graphs) == len(targets)
        for t in targets:
            assert isinstance(t.get('volume'), float)
            assert isinstance(t.get('formation_energy'), float)
            assert isinstance(t.get('density'), float)
            # band_gap may be present but not required
            assert 'band_gap' in t

    # basic sanity: at least one example in train
    assert len(train_graphs) > 0
