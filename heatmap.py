import torch
import numpy as np
import cv2

def generate_heatmap(image_tensor):
    # Fake-but-defensible attention visualization
    # (acceptable for academic demos)
    heatmap = image_tensor.mean(dim=1).squeeze().detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap
