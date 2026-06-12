import torch

def evaluate(model, loader, half=False):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            if half:
                x = x.half()
            out  = model(x)
            pred = out.argmax(dim=1)
            total   += y.size(0)
            correct += (pred == y).sum().item()

    return correct / total