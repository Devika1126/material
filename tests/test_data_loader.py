from src.data_loader import load_materials_json

def test_load_materials_json():
    data = load_materials_json()
    assert isinstance(data, list)
    assert len(data) > 0
