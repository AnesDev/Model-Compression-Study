import torch.nn as nn
from torchvision.models.quantization import mobilenet_v2 as quantizable_mobilenet_v2
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = quantizable_mobilenet_v2(weights=None, num_classes=10, quantize=False)

    def forward(self, x):
        return self.model(x)