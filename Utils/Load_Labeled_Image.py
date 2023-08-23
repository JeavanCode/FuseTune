from PIL.Image import Resampling
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def Load_Labeled_Image(dataset_path, batch_size, image_size, num_workers=4, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225),
                       transform_train=None, transform_valid=None, if_test=False):
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])

    if transform_valid is None:
        transform_valid = transforms.Compose([
            transforms.Resize((image_size // 4 * 5, image_size // 4 * 5), interpolation=Resampling.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])

    train_dataset = datasets.ImageFolder(dataset_path + '/train', transform=transform_train)
    valid_dataset = datasets.ImageFolder(dataset_path + '/valid', transform=transform_valid)
    print(valid_dataset.class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    if if_test:
        test_dataset = datasets.ImageFolder(dataset_path + '/test', transform=transform_valid)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        return train_loader, valid_loader, test_loader
    return train_loader, valid_loader
