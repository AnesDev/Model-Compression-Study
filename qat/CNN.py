import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq
from qat.fake_quantise import fake_quantise_ste

class CNN(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits

        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        w1 = fake_quantise_ste(self.conv1.weight, self.bits)
        x = F.conv2d(x, w1, self.conv1.bias, padding=1)
        x = self.pool(self.relu(x))

        w2 = fake_quantise_ste(self.conv2.weight, self.bits)
        x = F.conv2d(x, w2, self.conv2.bias, padding=1)
        x = self.pool(self.relu(x))

        x = x.reshape(x.size(0), -1)

        wfc1 = fake_quantise_ste(self.fc1.weight, self.bits)
        x = self.relu(F.linear(x, wfc1, self.fc1.bias))

        wfc2 = fake_quantise_ste(self.fc2.weight, self.bits)
        x = F.linear(x, wfc2, self.fc2.bias)

        return x