import os

from constants import DATA_PREPROCESS_MESSAGE
from utils import console
from rich.panel import Panel

import torch
from torchvision import datasets, transforms


class PreprocessData:
    """Class to preprocess data before training.
    It applies transformations to images and
    then loads them into DataLoaders."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_transforms = None
        self.datasets = None
        self.dataloaders = None

    @staticmethod
    def start(data_dir):
        preprocess_data = PreprocessData(data_dir)
        preprocess_data._preprocess_message
        preprocess_data._preprocess_data  # Access the property to trigger data processing
        return preprocess_data

    @property
    def _preprocess_data(self):
        console.print(f"[example][→][/example] Starting dataset preprocessing...")

        train_dir = os.path.join(self.data_dir, "train")
        test_dir = os.path.join(self.data_dir, "test")
        valid_dir = os.path.join(self.data_dir, "valid")

        self.data_transforms = {
            "train": transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
            ]),
            "test": transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
            ]),
            "valid": transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
            ])
        }

        self.datasets = {
            "train": datasets.ImageFolder(train_dir, transform=self.data_transforms['train']),
            "test": datasets.ImageFolder(test_dir, transform=self.data_transforms['test']),
            "valid": datasets.ImageFolder(valid_dir, transform=self.data_transforms['valid']),
        }

        self.dataloaders = {
            "train": torch.utils.data.DataLoader(self.datasets['train'], batch_size=64, pin_memory=True, shuffle=True),
            "test": torch.utils.data.DataLoader(self.datasets['test'], batch_size=64, pin_memory=True),
            "valid": torch.utils.data.DataLoader(self.datasets['valid'], batch_size=64, pin_memory=True),
        }

        console.print(f"[example][✓][/example] Data preprocessing completed successfully!\n")

    @property
    def _preprocess_message(self):
        console.print(Panel.fit(
            DATA_PREPROCESS_MESSAGE,
            title="Data Preprocessing",
            border_style="title"
        ))

        input("Press Enter to continue...")
