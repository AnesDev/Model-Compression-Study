import torch
import torch.nn as nn

from CNN import CNN
from fp32 import run_fp32
from fp16 import run_fp16
from int8 import run_int8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Uncomment to train from scratch
# criterion = nn.CrossEntropyLoss()
# model = CNN().to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# for epoch in range(20):
#     model.train()
#     total_loss = 0
#     for images, labels in trainloader:
#         images, labels = images.to(DEVICE), labels.to(DEVICE)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, loss: {total_loss:.4f}")
# torch.save(model.state_dict(), "fp32.pth")

print("Running FP32...")
fp32_accuracy, fp32_time, fp32_size = run_fp32()

print("Running FP16...")
fp16_accuracy, fp16_time, fp16_size = run_fp16()

print("Running INT8...")
int8_accuracy, int8_time, int8_size = run_int8()

# Comparison table 
print("\n" + "=" * 61)
print(f"{'Metric':<25} {'FP32':>8} {'FP16':>8} {'INT8':>8}")
print("-" * 61)
print(f"{'Accuracy':<25} {fp32_accuracy:>8.4f} {fp16_accuracy:>8.4f} {int8_accuracy:>8.4f}")
print(f"{'Inference time (s)':<25} {fp32_time:>8.3f} {fp16_time:>8.3f} {int8_time:>8.3f}")
print(f"{'Model size (KB)':<25} {fp32_size:>8.1f} {fp16_size:>8.1f} {int8_size:>8.1f}")
print("=" * 61)
print(f"{'Speedup':<25} {'1.00x':>8} {fp32_time/fp16_time:>7.2f}x {fp32_time/int8_time:>7.2f}x")
print(f"{'Size reduction':<25} {'1.00x':>8} {fp32_size/fp16_size:>7.2f}x {fp32_size/int8_size:>7.2f}x")
print(f"{'Accuracy drop':<25} {'0.00%':>8} {(fp32_accuracy-fp16_accuracy)*100:>7.2f}% {(fp32_accuracy-int8_accuracy)*100:>7.2f}%")
print("=" * 61)