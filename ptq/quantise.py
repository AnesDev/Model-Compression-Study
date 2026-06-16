import torch
import torch.ao.quantization as tq
import os

from data import trainloader, testloader
from evaluate import evaluate
from cnn import CNN

model = CNN()
model.load_state_dict(torch.load("fp32_cnn.pth", map_location="cpu"))

model.eval()
model.fuse_model()

torch.backends.quantized.engine = 'fbgemm'
model.qconfig = tq.get_default_qconfig('fbgemm')
tq.prepare(model, inplace=True)

print("Calibrating...")
for i, (x, _) in enumerate(trainloader):
    model(x)
    if i >= 100:
        break

tq.convert(model.eval(), inplace=True)

save_path = "ptq/ptq8_cnn.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

test_acc = evaluate(model, testloader)
print(f"PTQ8 Test Accuracy: {test_acc:.4f}")