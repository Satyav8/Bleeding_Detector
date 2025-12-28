import numpy as np

def generate_heatmap(image_tensor):
    """
    Lightweight saliency-style heatmap.
    Academic demo-safe, no OpenCV dependency.
    """
    heatmap = image_tensor.mean(dim=1).squeeze().detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

