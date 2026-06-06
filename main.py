import torch
import torch.nn as nn
import time
import os

from CNN import CNN
from evaluate import evaluate
from quantise import quantise
from data import trainloader, testloader

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

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

    # scheduler.step()
    print(f"Epoch {epoch+1}, loss: {total_loss:.4f}")

torch.save(model.state_dict(), "fp32.pth")

accuracy = evaluate(model, testloader)
print(f"FP32 Accuracy: {accuracy:.4f}")

model.eval()
start = time.time()
with torch.no_grad():
    for x, y in testloader:
        model(x)
fp32_time = time.time() - start
print(f"FP32 inference time: {fp32_time:.3f}s")

fp32_size = os.path.getsize("fp32.pth") / 1024
print(f"FP32 size: {fp32_size:.1f} KB")

quantise(fp32_time, fp32_size)