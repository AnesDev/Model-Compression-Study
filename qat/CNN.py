import torch.nn as nn
import torch.ao.quantization as tq

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.relu_fc = nn.ReLU()
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 4 * 4, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))
        x = x.reshape(x.size(0), -1)
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        tq.fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                                ['conv2', 'bn2', 'relu2'],
                                ['conv3', 'bn3', 'relu3']], inplace=True)