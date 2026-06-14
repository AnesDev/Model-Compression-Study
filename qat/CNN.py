import torch
import torch.nn as nn
import torch.nn.functional as F
from qat.fake_quantise import fake_quantise_ste

class CNN(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        w1 = fake_quantise_ste(self.conv1.weight, self.bits)
        x = self.pool(self.relu(self.bn1(F.conv2d(x, w1, self.conv1.bias, padding=1))))

        w2 = fake_quantise_ste(self.conv2.weight, self.bits)
        x = self.pool(self.relu(self.bn2(F.conv2d(x, w2, self.conv2.bias, padding=1))))

        w3 = fake_quantise_ste(self.conv3.weight, self.bits)
        x = self.pool(self.relu(self.bn3(F.conv2d(x, w3, self.conv3.bias, padding=1))))

        x = x.reshape(x.size(0), -1)

        wfc1 = fake_quantise_ste(self.fc1.weight, self.bits)
        x = self.relu(F.linear(x, wfc1, self.fc1.bias))

        wfc2 = fake_quantise_ste(self.fc2.weight, self.bits)
        x = F.linear(x, wfc2, self.fc2.bias)

        return x