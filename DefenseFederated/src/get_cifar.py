from torchvision import datasets, transforms


data_dir = '../data/cifar/'
apply_transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                    transform=apply_transform)

test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=apply_transform)