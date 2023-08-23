import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        self.label = list(h5py.File(label_path, 'r')['y'])
        self.data = None
        self.label_path = label_path
        self.data_path = data_path
        self.transform = transform

    def open_hdf5(self):
        self.data = list(h5py.File(self.data_path, 'r')['x'])
        self.label = list(h5py.File(self.label_path, 'r')['y'])

    def __getitem__(self, index):
        if self.data is None:
            self.open_hdf5()
        image = self.data[index]
        label = self.label[index]
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(np.squeeze(label), dtype=torch.int64)
        return image, label

    def __len__(self):
        return len(self.label)


def LoadCam(batch_size, image_size, root, num_workers=2):
    transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(image_size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=0., translate=(0.04, 0.04)),
            ])
    transform_valid = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(image_size, antialias=True),
            ])

    train_data = MyDataset(data_path=root + r'\camelyonpatch_level_2_split_train_x.h5', label_path=root + r'\camelyonpatch_level_2_split_train_y.h5', transform=transform_train)
    valid_data = MyDataset(data_path=root + r'\camelyonpatch_level_2_split_valid_x.h5', label_path=root + r'\camelyonpatch_level_2_split_valid_y.h5', transform=transform_valid)

    train_loader = DataLoader(dataset=train_data, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader

