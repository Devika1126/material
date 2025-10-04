from src.models.generator import propose_candidates
from src.featurizers import structure_to_graph
from src.models.gnn import CGCNN
import torch
import numpy as np

def predict_ensemble(models, data, device):
    """
    Predict with ensemble of models, return mean and stddev.
    """
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(data.to(device))
            preds.append(out.cpu().numpy())
    preds = np.stack(preds)
    mean = np.mean(preds, axis=0)
    std = np.std(preds, axis=0)
    return mean, std

def acquisition_function(pred, std):
    """
    Acquisition function: prioritize low volume and low uncertainty.
    """
    volume = pred[0]
    score = -volume - std.sum()  # prefer lower volume and uncertainty
    return score

def submit_dft_job(structure):
    """
    Stub for DFT job submission (to be implemented with atomate/ASE).
    """
    # TODO: Implement DFT job submission
    pass

def active_learning_loop(models, device, top_k=5):
    """
    Active learning loop: generate, predict, rank, select, submit.
    """
    candidates = propose_candidates(n=20)
    scored = []
    for struct in candidates:
        if struct is None:
            continue
        data = structure_to_graph(struct)
        return mean, std
    
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
        score = acquisition_function(mean, std)
        scored.append((score, struct, mean, std))
    scored.sort(reverse=True, key=lambda x: x[0])
    selected = scored[:top_k]
    for _, struct, mean, std in selected:
        submit_dft_job(struct)
    return selected
