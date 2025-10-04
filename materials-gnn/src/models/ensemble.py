import torch
import numpy as np

class EnsembleModel:
    """
    Ensemble wrapper for multi-task models.
    """
    def __init__(self, model_cls, checkpoints, device, **kwargs):
        self.models = []
        for ckpt in checkpoints:
            model = model_cls(**kwargs).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            self.models.append(model)
        self.device = device

    def predict(self, data):
        preds = []
        for model in self.models:
            with torch.no_grad():
                out = model(data.to(self.device))
                preds.append({k: v.cpu().numpy().flatten() for k, v in out.items()})
        keys = preds[0].keys()
        mean = {k: np.mean([p[k] for p in preds], axis=0) for k in keys}
        std = {k: np.std([p[k] for p in preds], axis=0) for k in keys}
        return mean, std
