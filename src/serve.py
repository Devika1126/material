from fastapi import FastAPI, UploadFile, File
from pymatgen.core import Structure
import tempfile
import torch
from src.featurizers import structure_to_graph
from src.models.gnn import CGCNN
import numpy as np

app = FastAPI()

# Load ensemble models
N_ENSEMBLE = 5
models = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i in range(N_ENSEMBLE):
    model = CGCNN().to(device)
    checkpoint = f"cgcnn_{i}.pt"
    try:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        models.append(model)
    except Exception:
        pass

def predict_structure(structure: Structure):
    data = structure_to_graph(structure)
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(data.to(device))
            preds.append(out.cpu().numpy().flatten())
    preds = np.stack(preds)
    mean = np.mean(preds, axis=0)
    std = np.std(preds, axis=0)
    return mean, std

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict band gap, formation energy, and uncertainty from CIF or POSCAR file.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        try:
            structure = Structure.from_file(tmp.name)
        except Exception as e:
            return {"error": str(e)}
    mean, std = predict_structure(structure)
    return {
        "volume": float(mean[0]),
        "uncertainty": float(std[0])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
