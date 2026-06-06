import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

from CNN import CNN
from evaluate import evaluate
from quantise import quantise

from data import trainloader, testloader

import time

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    total_loss = 0

    for images, labels in trainloader:

        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, loss: {total_loss:.4f}")

torch.save(model.state_dict(), "fp32.pth")

model.eval()
start = time.time()
with torch.no_grad():
    for x, y in testloader:
        model(x)
fp32_time = time.time() - start
print(f"FP32 inference time: {fp32_time:.3f}s")

quantise(fp32_time)