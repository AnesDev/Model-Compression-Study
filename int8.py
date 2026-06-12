import torch
import torch.ao.quantization as tq
import time
import os
from evaluate import evaluate
from data import trainloader, testloader
from CNN import CNN


def run_int8():
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

    accuracy = evaluate(model, testloader)

    model.eval()
    start = time.time()
    with torch.no_grad():
        for x, _ in testloader:
            model(x)
    inference_time = time.time() - start

    size = os.path.getsize("int8.pth") / 1024

    return accuracy, inference_time, size