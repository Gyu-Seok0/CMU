import torch
import torch.nn as nn
import torchvision #This library is used for image-based operations (Augmentations)
import torchvision.transforms as transforms 
import os
from PIL import Image

class ClassificationTestDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, transforms):
        self.data_dir   = data_dir
        self.transforms = transforms

        # This one-liner basically generates a sorted list of full paths to each image in the test directory
        self.img_paths  = list(map(lambda fname: os.path.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        return self.transforms(Image.open(self.img_paths[idx]))


def make_dataset(DATA_DIR):
    TRAIN_DIR = os.path.join(DATA_DIR, "classification/train") 
    VAL_DIR = os.path.join(DATA_DIR, "classification/dev")
    TEST_DIR = os.path.join(DATA_DIR, "classification/test")
    train_transforms = transforms.Compose([ 
                        # Implementing the right transforms/augmentation methods is key to improving performance.
                        #transforms.GaussianBlur(kernel_size=(3, 3)),
                        transforms.RandomHorizontalFlip(0.5),
                        #transforms.RandomVerticalFlip(np.random.uniform(0,0.5)),
                        transforms.RandomRotation(degrees=5),
                        #transforms.RandomResizedCrop(size=(224, 224)),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5130, 0.4034, 0.3522),(0.3075, 0.2702, 0.2589))                    
                        ])

    val_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5130, 0.4034, 0.3522),(0.3075, 0.2702, 0.2589))
                                    ])
    
    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform = train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(VAL_DIR, transform = val_transforms)

    test_dataset = ClassificationTestDataset(TEST_DIR, transforms = val_transforms)
    return train_dataset, val_dataset, test_dataset