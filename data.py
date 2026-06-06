from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    # before — wrong, single channel broadcast
    transforms.Normalize((0.5,), (0.5,)),

    # after — correct CIFAR-10 RGB mean and std
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

testset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)