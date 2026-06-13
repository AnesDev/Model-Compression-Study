import torch

def fake_quantise(tensor, bits, symmetric=False):
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)

    if symmetric == False:
        q_min = 0
        q_max = 2**bits -1

        scale = (max_val - min_val) / (q_max - q_min)
        if scale == 0: scale = 1e-8 
        
        zero_point = torch.round(q_min - min_val/scale)

    else:
        q_min = -(2**(bits-1))
        q_max = 2**(bits-1) - 1
        
        scale = torch.max(abs(min_val), abs(max_val)) / q_max
        if scale == 0: scale = 1e-8 

        zero_point = 0

    q = torch.round(tensor/scale) + zero_point

    dequant = (q - zero_point) * scale

    return dequant

def fake_quantise_ste(tensor, bits, symmetric=False):
    q = fake_quantise(tensor, bits, symmetric)
    return tensor + (q - tensor).detach()

x = torch.tensor([0.3, 0.7], requires_grad=True)
y = fake_quantise_ste(x, 4)
y.sum().backward()
print(x.grad)  # expect [1., 1.]