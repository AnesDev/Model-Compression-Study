import torch
import time
import os
from evaluate import evaluate
from data import testloader
from CNN import CNN


def run_fp16():
    model = CNN()
    model.load_state_dict(torch.load("fp32.pth", map_location="cpu"))
    model = model.half()
    model.eval()

    torch.save(model.state_dict(), "fp16.pth")

    accuracy = evaluate(model, testloader, half=True)

    start = time.time()
    with torch.no_grad():
        for x, _ in testloader:
            x = x.half()
            model(x)
    inference_time = time.time() - start

    size = os.path.getsize("fp16.pth") / 1024

    return accuracy, inference_time, size