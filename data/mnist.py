from torchvision.datasets import MNIST
from torchvision import transforms

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

def get_mnist_dataset(root="~/disk2/data/MNIST"):
    train_set = MNIST(root=root, train=True, download=True, transform=transform)
    val_set = MNIST(root=root, train=False, download=True, transform=transform)
    return train_set, val_set