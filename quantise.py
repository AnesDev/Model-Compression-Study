import torch
import torch.ao.quantization as tq
import time
from evaluate import evaluate
from data import trainloader, testloader
from CNN import CNN

def quantise(fp32_time):
    model = CNN()
    model.load_state_dict(torch.load("fp32.pth", map_location="cpu"))
    model.eval()

    torch.backends.quantized.engine = 'fbgemm'
    model.qconfig = tq.get_default_qconfig('fbgemm')

    tq.prepare(model, inplace=True)

    for i, (x, _) in enumerate(trainloader):
        model(x)
        if i > 100:
            break

    quantized_model = tq.convert(model, inplace=False)
    torch.save(quantized_model.state_dict(), "int8.pth")

    accuracy = evaluate(quantized_model, testloader)
    print("INT8 accuracy:", accuracy)

    # time INT8
    quantized_model.eval()
    start = time.time()
    with torch.no_grad():
        for x, y in testloader:
            quantized_model(x)
    int8_time = time.time() - start

    print(f"INT8 inference time: {int8_time:.3f}s")
    print(f"Speedup: {fp32_time / int8_time:.2f}x")