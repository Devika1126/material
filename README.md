# materials-discovery

AI-driven pipeline for discovering new nanoscale materials with tunable band gaps and stable formation energies.

## Features

- Fetches data from Materials Project API
- Converts crystal structures to graph representations
- Trains CGCNN baseline GNN
- Ensemble uncertainty estimation
- VAE-based generative scaffold
- Active learning loop
- FastAPI inference API

## Setup

1. Install requirements:
    ```sh
    pip install -r requirements.txt
    ```
2. Insert your Materials Project API key in `src/data_loader.py`.
3. Download data:
    ```sh
    python src/data_loader.py
    ```
4. Train model:
    ```sh
    python src/train.py --epochs 50 --lr 1e-3 --batch_size 32 --cutoff 5.0 --dataset_path data/materials.json
    ```
5. Run API server:
    ```sh
    uvicorn src.serve:app --host 0.0.0.0 --port 8000
    ```

## Docker

Build and run:
```sh
docker build -t materials-discovery .
docker run -p 8000:8000 materials-discovery
```

## Tests

Run tests:
```sh
pytest tests/
```
