import torch
import torch.nn as nn
import os

from data import trainloader, testloader
from evaluate import evaluate
from qat.cnn import CNN

DEVICE = torch.device("cpu")
NUM_EPOCHS = 5

model = CNN(bits=8).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:>2}/{NUM_EPOCHS}  loss: {total_loss:.4f}")

save_path = os.path.join(os.path.dirname(__file__), "qat8_cnn.pth")
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

accuracy = evaluate(model, testloader, device=DEVICE)
print(f"QAT8 Accuracy: {accuracy:.4f}")