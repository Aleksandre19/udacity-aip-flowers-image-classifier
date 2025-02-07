# Standard library imports
import os

# Third-party imports
import torch
from rich.panel import Panel
from torchvision import datasets, transforms

# Local imports
from constants import DATA_PREPROCESS_MESSAGE
from utils import console


class PreprocessData:
    """Handles data preprocessing for the flower classifier training pipeline.
    
    This class manages the complete data preprocessing workflow including:
    - Applying appropriate transformations for train/test/validation sets
    - Loading and organizing image data using PyTorch DataLoaders
    - Implementing data augmentation for training set
    - Normalizing images for model compatibility
    """

    def __init__(self, data_dir):
        """Initialize the PreprocessData class.
        
        Args:
            data_dir (str): Path to the root directory containing train/test/valid subdirectories
        """
        self.data_dir = data_dir
        self.data_transforms = None
        self.datasets = None
        self.dataloaders = None

    @staticmethod
    def start(data_dir):
        """Create and initialize a PreprocessData instance.
        
        Args:
            data_dir (str): Path to the data directory
            
        Returns:
            PreprocessData: Initialized instance with processed datasets
        """
        preprocess_data = PreprocessData(data_dir)
        preprocess_data._preprocess_message
        preprocess_data._preprocess_data  # Access the property to trigger data processing
        return preprocess_data

    @property
    def _preprocess_data(self):
        """Process and prepare datasets for training.
        
        Performs the following steps:
        1. Sets up data transformations for each dataset split
        2. Creates dataset objects with appropriate transformations
        3. Initializes data loaders with batching and memory pinning
        
        The training set includes data augmentation (rotation, crop, flip),
        while test/validation sets only include resizing and normalization.
        """
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
        """Display preprocessing status message.
        
        Shows a panel with preprocessing information and waits for user confirmation
        before proceeding.
        """
        console.print(Panel.fit(
            DATA_PREPROCESS_MESSAGE,
            title="Data Preprocessing",
            border_style="title"
        ))

        input("Press Enter to continue...")
