# materials-gnn

A research-grade framework for multi-task prediction of materials properties using graph neural networks.

## Features
- Multi-task prediction (volume, band gap, formation energy, density, bulk modulus)
- Model zoo: CGCNN, MEGNet, SchNet, DimeNet++, ALIGNN
- Ensemble and uncertainty estimation
- Explainability (attention, integrated gradients, Grad-CAM)
- Benchmarking and cross-database validation
- FastAPI service for inference
- Reproducible Docker setup

## CLI Examples

### Data Download
```sh
python data/data_loader.py --api_key <YOUR_KEY>
```

### Training (single model)
```sh
python src/train.py --model cgcnn --epochs 50 --batch_size 32
```

### Ensemble Training
```sh
python src/train.py --model megnet --ensemble --epochs 50
```

### Evaluation
```sh
python src/evaluate.py --model cgcnn --split test
```

### Serving API
```sh
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

### Explainability
```sh
python src/explain.py --model cgcnn --input example.cif --method attention
```

## Reproducibility

- All commands above run in Docker:
```sh
docker build -t materials-gnn .
docker run -p 8000:8000 materials-gnn
```
- Pretrained models in `pretrained/`
