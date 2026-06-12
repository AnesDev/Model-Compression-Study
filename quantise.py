import torch
import torch.ao.quantization as tq
import time
import os
from evaluate import evaluate
from data import trainloader, testloader
from CNN import CNN

def quantise(fp32_time, fp32_size):
    model = CNN()
    model.load_state_dict(torch.load("fp32.pth", map_location="cpu"))
    model.eval()

    torch.backends.quantized.engine = 'fbgemm'
    model.model.qconfig = tq.get_default_qconfig('fbgemm')

    model.model.fuse_model()               
    tq.prepare(model.model, inplace=True)

    print("Calibrating...")
    for i, (x, _) in enumerate(trainloader):
        model(x)
        if i > 100:
            break

    tq.convert(model.model, inplace=True)
    torch.save(model.state_dict(), "int8.pth")

    int8_accuracy = evaluate(model, testloader)
    print(f"INT8 accuracy: {int8_accuracy:.4f}")

    model.eval()
    start = time.time()
    with torch.no_grad():
        for x, _ in testloader:
            model(x)
    int8_time = time.time() - start

    int8_size = os.path.getsize("int8.pth") / 1024

    print(f"INT8 inference time : {int8_time:.3f}s")
    print(f"INT8 model size     : {int8_size:.1f} KB")

    print("=" * 45)
    print(f"{'Metric':<25} {'FP32':>8} {'INT8':>8}")
    print("-" * 45)
    print(f"{'Inference time (s)':<25} {fp32_time:>8.3f} {int8_time:>8.3f}")
    print(f"{'Model size (KB)':<25} {fp32_size:>8.1f} {int8_size:>8.1f}")
    print("=" * 45)
    print(f"Speedup        : {fp32_time / int8_time:.2f}x  (CPU inference)")
    print(f"Size reduction : {fp32_size / int8_size:.2f}x")