import torch
import timm
import torch.nn as nn

class BleedingViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=2
        )

    def forward(self, x):
        return self.model(x)

def load_model():
    model = BleedingViT()
    model.eval()
    return model
