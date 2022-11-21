from torchvision.datasets import CIFAR10
from torchvision import transforms

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_cifar_dataset(root="~/disk2/data/CIFAR"):
    train_set = CIFAR10(root=root, train=True, download=True, transform=transform)
    val_set = CIFAR10(root=root, train=False, download=True, transform=transform)
    return train_set, val_set