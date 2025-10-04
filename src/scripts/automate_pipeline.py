import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n[INFO] {description}...")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] {description} failed.")
        sys.exit(1)
    print(f"[INFO] {description} completed successfully.")

def prepare_dataset():
    """Prepare the dataset and save splits."""
    command = (
        "python -u -m src.scripts.prepare_and_save "
        "--input data/materials.json --targets volume formation_energy density "
        "--include_bandgap --output_dir data/splits --shuffle"
    )
    run_command(command, "Preparing dataset and saving splits")

def train_single_model():
    """Train, validate, and test a single model."""
    command = (
        "python -u -m src.scripts.run_ensemble_from_splits "
        "--epochs 100 --ensemble 1 --splits_dir data/splits"
    )
    run_command(command, "Training, validating, and testing a single model")

def summarize_results():
    """Summarize results and save artifacts."""
    command = (
        "python -u -m src.scripts.summarize_results "
        "--results_dir results --output_file results/summary_report.txt"
    )
    run_command(command, "Summarizing results and saving artifacts")

def main():
    prepare_dataset()
    train_single_model()
    summarize_results()

if __name__ == "__main__":
    main()