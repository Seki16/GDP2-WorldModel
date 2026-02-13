import numpy as np

def mse(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    return np.mean((pred - target) ** 2)

def cosine_similarity(pred, target, eps=1e-8):
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)

    dot = np.dot(pred, target)
    norm_pred = np.linalg.norm(pred)
    norm_target = np.linalg.norm(target)

    return dot / (norm_pred * norm_target + eps)
