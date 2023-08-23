from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CLEVRClassification


def LoadClevr(batch_size, image_size, root_dir, num_workers=2, download=True):
    transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((image_size, image_size), antialias=True),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=0., translate=(0.04, 0.04)),
                ])
    transform_valid = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size), antialias=True),
            ])

    # Initialize the CLEVRClassification class with download=True to download the dataset
    clevr_train = CLEVRClassification(root=root_dir, download=download, split='train', transform=transform_train)
    clevr_valid = CLEVRClassification(root=root_dir, download=download, split='val', transform=transform_valid)
    train_loader = DataLoader(dataset=clevr_train, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=clevr_valid, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader


