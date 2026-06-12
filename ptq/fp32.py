import torch
import time
import os
from evaluate import evaluate
from data import testloader
from ptq.cnn import MobileNet


def run_fp32():
    model = MobileNet()
    model.load_state_dict(torch.load("ptq/fp32.pth", map_location="cpu"))
    model.eval()

    accuracy = evaluate(model, testloader)

    start = time.time()
    with torch.no_grad():
        for x, _ in testloader:
            model(x)
    inference_time = time.time() - start

    size = os.path.getsize("ptq/fp32.pth") / 1024

    return accuracy, inference_time, size