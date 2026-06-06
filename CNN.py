import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.quant(x)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.reshape(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.dequant(x)
        return x
