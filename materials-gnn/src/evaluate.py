# Benchmarking script for all models and properties
import pandas as pd
from models import MODEL_ZOO

def benchmark(models, dataset, split, device):
    results = []
    for name, model_cls in models.items():
        model = model_cls().to(device)
        # TODO: Load checkpoint
        # TODO: Predict on split
        # TODO: Compute MAE/RMSE for each property
        for prop in ["volume", "band_gap", "formation_energy", "density", "bulk_modulus"]:
            mae, rmse = None, None  # TODO: Compute
            results.append({
                "model": name,
                "property": prop,
                "mae": mae,
                "rmse": rmse
            })
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    # TODO: Add plotting
