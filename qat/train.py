import torch
import torch.nn as nn
import torch.ao.quantization as tq
import os

from qat.data import trainloader, valloader, testloader
from evaluate import evaluate
from qat.cnn import CNN

NUM_EPOCHS = 40

model = CNN()
model.eval()
model.fuse_model()

torch.backends.quantized.engine = 'fbgemm'
model.qconfig = tq.get_default_qat_qconfig('fbgemm')

model.train()
tq.prepare_qat(model, inplace=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(trainloader)
    train_acc = correct / total
    val_acc = evaluate(model, valloader)

    print(
        f"Epoch {epoch+1:>2}/{NUM_EPOCHS} "
        f"loss={train_loss:.4f} "
        f"train_acc={train_acc:.4f} "
        f"val_acc={val_acc:.4f}"
    )

tq.convert(model.eval(), inplace=True)

save_path = os.path.join(os.path.dirname(__file__), "qat8_cnn.pth")
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

test_acc = evaluate(model, testloader)
print(f"QAT8 Test Accuracy: {test_acc:.4f}")