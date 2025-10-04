from .gnn import CGCNN

MODEL_ZOO = {
    'cgcnn': CGCNN,
}

__all__ = ['CGCNN', 'MODEL_ZOO']
