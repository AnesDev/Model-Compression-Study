import torch
import torch.nn as nn
import os

from qat.data import trainloader, testloader
from evaluate import evaluate
from qat.cnn import CNN

NUM_EPOCHS = 30

model = CNN(bits=8)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in trainloader:
        images, labels = images, labels
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

accuracy = evaluate(model, testloader)
print(f"QAT8 Accuracy: {accuracy:.4f}")