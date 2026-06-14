from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])


train_full = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform_train
)

val_full = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform_eval
)

testset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform_eval
)


num_train = 50000
train_size = 45000
val_size = 5000

generator = torch.Generator().manual_seed(42)
perm = torch.randperm(num_train, generator=generator)

train_idx = perm[:train_size]
val_idx = perm[train_size:]



trainset = Subset(train_full, train_idx)
valset = Subset(val_full, val_idx)


trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader   = DataLoader(valset, batch_size=64, shuffle=False)
testloader  = DataLoader(testset, batch_size=64, shuffle=False)