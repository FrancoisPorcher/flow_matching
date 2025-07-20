import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

transform = transforms.ToTensor()

# Training set
train_dataset = CIFAR10(
    root="/datasets01/cifar-pytorch/11222017",
    train=True,
    download=False,
    transform=transform,
)

# Test set
test_dataset = CIFAR10(
    root="/datasets01/cifar-pytorch/11222017",
    train=False,
    download=False,
    transform=transform,
)

# construct dataloader

train_batch_size = 16
test_batch_size = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
)


# each element has shape [B, C, H, W]
breakpoint()
