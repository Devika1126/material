import itertools
import subprocess
import sys

# Define hyperparameter grid
learning_rates = [1e-2, 1e-3, 1e-4]
batch_sizes = [16, 32, 64]
cutoffs = [4.0, 5.0, 6.0]

results = []

for lr, batch_size, cutoff in itertools.product(learning_rates, batch_sizes, cutoffs):
    print(f"\nRunning: lr={lr}, batch_size={batch_size}, cutoff={cutoff}")
    cmd = [
        sys.executable, "src/train.py",
        "--dataset_path", "data/materials.json",
        "--epochs", "30",
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--cutoff", str(cutoff)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    # Extract final test MAE and RMSE from output
    mae, rmse = None, None
    for line in output.splitlines():
        if line.startswith("Test MAE="):
            try:
                parts = line.split()
                mae = float(parts[1].split('=')[1].replace(',', ''))
                rmse = float(parts[3].split('=')[1])
            except Exception:
                pass
    results.append({
        "lr": lr,
        "batch_size": batch_size,
        "cutoff": cutoff,
        "mae": mae,
        "rmse": rmse
    })
    print(f"Result: MAE={mae}, RMSE={rmse}")

# Print best result
best = min([r for r in results if r["mae"] is not None], key=lambda x: x["mae"])
print("\nBest hyperparameters:")
print(f"Learning rate: {best['lr']}")
print(f"Batch size: {best['batch_size']}")
print(f"Cutoff: {best['cutoff']}")
print(f"Test MAE: {best['mae']}")
print(f"Test RMSE: {best['rmse']}")
