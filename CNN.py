import torch.nn as nn
from torchvision.models import mobilenet_v2

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v2(weights=None, num_classes=10)

    def forward(self, x):
        return self.model(x)