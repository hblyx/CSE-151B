from numpy import genfromtxt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch


########## DO NOT change this function ##########
# If you change it to achieve better results, we will deduct points. 
def train_val_split(train_dataset):
    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(42))
    return train_subset, val_subset


#################################################

########## DO NOT change this variable ##########
# If you change it to achieve better results, we will deduct points. 
transform_test = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


#################################################

class FoodDataset(Dataset):
    def __init__(self, data_csv, transforms=None):
        # this read the csv to find the according jpg, and results in PIL objects
        self.data = genfromtxt(data_csv, delimiter=',', dtype=str)

        self.transforms = transforms

    def __getitem__(self, index):
        fp, _, idx = self.data[index]
        idx = int(idx)
        img = Image.open(fp)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, idx  # (img, class)

    def __len__(self):
        return len(self.data)


def get_dataset(csv_path, transform):
    return FoodDataset(csv_path, transform)


def create_dataloaders(dataset, batch_size=None):
    if batch_size is None:
        return dataset

    l_idx, r_idx = 0, batch_size
    while r_idx < len(dataset):
        yield dataset[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield dataset[l_idx:]


def get_dataloaders(train_csv, test_csv, transform_train=None, batch_size=64):
    train_dataset = get_dataset(train_csv, transform_train)

    ########## DO NOT change the following two lines ##########
    # If you change it to achieve better results, we will deduct points.
    test_dataset = get_dataset(test_csv, transform_test)
    train_set, val_set = train_val_split(train_dataset)
    ###########################################################

    # [train_loader, val_loader, test_loader]
    dataloaders = create_dataloaders(train_set, batch_size=batch_size), \
                  create_dataloaders(val_set), \
                  create_dataloaders(test_dataset)
    return dataloaders
