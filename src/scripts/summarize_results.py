import os
import json

def summarize_results(results_dir, output_file):
    """Summarize results from the training pipeline and save to a file."""
    summary = {}

    # Iterate through result files in the directory
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    try:
                        data = json.load(f)
                        summary[file] = data
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from {file_path}")

    # Write the summary to the output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Summary saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize results and save artifacts.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing result files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the summary file.")

    args = parser.parse_args()
    summarize_results(args.results_dir, args.output_file)