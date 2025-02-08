"""Utility functions and classes for the flower image classifier project.

This module provides various utility functions and classes including:
- Argument parsing for training and prediction
- Custom logging configuration
- UI theme management
- Dataset downloading and extraction
- Model architecture utilities
- File selection dialogs

The module uses Rich for enhanced terminal output and Questionary for interactive prompts.
"""

# Standard library imports
import argparse
import logging
import sys
import tarfile
import urllib.request
from pathlib import Path
from tkinter import Tk, filedialog

# Third-party imports
import questionary
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.theme import Theme
from torch import nn
from torchvision import models

# Local imports
from constants import CHOOSE_MODEL_ERROR_MESSAGE, PROVIDE_DATA_RICH_MESSAGE


def setup_logger():
    """Set up and configure the logger with Rich formatting.
    
    Returns:
        logging.Logger: Configured logger instance with Rich formatting
        and tracebacks enabled.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger("rich")

# Set up logger
log = setup_logger()

# Define color schemes for terminal light and dark themes
light_theme = Theme({
    'arg': 'blue',
    'desc': 'green',
    'example': 'blue',
    'title': 'cyan',
    'error': 'red',
    'info': 'yellow',
    'purple': 'magenta'
})

dark_theme = Theme({
    'arg': 'bright_blue',
    'desc': 'bright_green',
    'example': 'bright_cyan',
    'title': 'bright_cyan',
    'error': 'bright_red',
    'info': 'bright_yellow',
    'purple': 'bright_magenta'
})

# Initialize rich console
console = Console()

# Detect terminal theme color
is_light_theme = console.color_system == 'standard' or \
                 (console.color_system == '256' and console.is_terminal)

# Set appropriate theme to the rich console
console = Console(theme=light_theme if is_light_theme else dark_theme)


class CustomArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser for training script with enhanced error messages.
    
    This class overrides the default ArgumentParser to provide more user-friendly
    error messages with Rich formatting and helpful usage examples.
    """
    
    def error(self, message):
        """Override error handling with custom formatted messages.
        
        Args:
            message (str): The error message from argparse
        """

        console.print(f"[error]error: Data directory is required in order to train a model![/error]")
        console.print(f"[example]- Example: python3 train.py data/directory[/example]")
        console.print("------------------------------------------")
        console.print(f"[info]For more information: python3 train.py --info[/info]")
        self.exit(2)
        


class InfoAction(argparse.Action):
    """Custom action for handling the --info flag with Rich formatting.
    
    This class overrides the default Action class to display detailed information
    about the script usage with enhanced formatting using Rich panels.
    """
    
    def __call__(self, parser, namespace, values, option_string=None):
        """Execute the info action when --info flag is used.
        
        Args:
            parser: The argument parser instance
            namespace: Parsed arguments namespace
            values: Command-line supplied values
            option_string: Option string from command-line
        """

        console.print(Panel.fit(
            f"{PROVIDE_DATA_RICH_MESSAGE}",
            title="Flowers Image Classifier Information",
            border_style="title"
        ))
        parser.exit()


def get_train_terminal_args():
    """Parse and validate command-line arguments for model training.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments including:
            - data_dir: Path to training data directory
            - save_dir: Directory for saving checkpoints
            - arch: Model architecture (vgg11, vgg13, vgg16, vgg19)
            - learning_rate: Learning rate for training
            - hidden_units: Number of units in hidden layers
            - epochs: Number of training epochs
            - gpu: Whether to use GPU for training
    """

    # Create parser first to handle info flag
    parser = CustomArgumentParser(
        description='Train a neural network on a dataset of images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage='example: python3 train.py "data/directory"'
    )

    # Add info argument first to ensure it's checked before other arguments
    parser.add_argument('--info',
                    action=InfoAction,
                    nargs=0,
                    help='show detailed information about the script')

    # Required argument
    parser.add_argument('data_dir',
                        type=str,
                        help='path to the data directory containing datasets')

    # Optional arguments
    parser.add_argument('--save_dir', 
                        type=str,
                        default='checkpoints',
                        help='directory to save checkpoints')

    parser.add_argument('--arch', 
                        type=str,
                        default='vgg16',
                        choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'],
                        help='Used model architecture')
                            
    parser.add_argument('--learning_rate', 
                        type=float,
                        default=0.0001,
                        help='learning rate for training')

    parser.add_argument('--input_size', 
                        type=int,
                        default=25088,
                        help='input size')

    parser.add_argument('--hidden_units', 
                        type=int,
                        default=[4096, 1024],
                        help='number of units in hidden layers')

    parser.add_argument('--output_size', 
                        type=int,
                        default=102,
                        help='output size')

    parser.add_argument('--drop_p', 
                        type=float,
                        default=0.2,
                        help='dropout probability')

    parser.add_argument('--epochs', 
                        type=int,
                        default=17,
                        help='number of epochs for training')

    parser.add_argument('--valid_interval', 
                        type=int,
                        default=100,
                        help='steps of validation interval')

    parser.add_argument('--gpu', 
                        action='store_true',
                        help='use GPU for training if available')

    return parser.parse_args()


class CustomPredictArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser for prediction script with enhanced error messages.
    
    This class overrides the default ArgumentParser to provide more user-friendly
    error messages specifically for model prediction requirements.
    """
    
    def error(self, message):
        """Override error handling with custom formatted messages.
        
        Args:
            message (str): The error message from argparse
        """

        console.print(Panel.fit(
            CHOOSE_MODEL_ERROR_MESSAGE,
            title="Specify Model",
            border_style="title"
        ))
        self.exit(2)  
        

def get_predict_terminal_args():
    """Parse and validate command-line arguments for model prediction.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments including:
            - model: Path to the trained model file
            - info: Flag for displaying detailed information
    """

    # Create parser first to handle info flag
    parser = CustomPredictArgumentParser(
        description='Predict images using a trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage='example: python3 predict.py "data/directory"'
    )

    # Add info argument first to ensure it's checked before other arguments
    parser.add_argument('--info',
                    action=InfoAction,
                    nargs=0,
                    help='show detailed information about the script')

    # Required argument
    parser.add_argument('--model',
                        type=str,
                        help='path to the model directory')

    return parser.parse_args()


def download_dataset(url, tar_file, data_path):
    """Download and extract a dataset from a URL with progress tracking.
    
    Args:
        url (str): URL to download the dataset from
        tar_file (Path): Path to save the downloaded tar file
        data_path (Path): Directory to extract the dataset to
    
    The function shows progress bars for both download and extraction
    processes using Rich progress indicators.
    """
    # Download dataset
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        transient=False
    ) as progress:
        # Download with progress tracking
        download_task = progress.add_task("[yellow]Downloading dataset...", total=None)
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('content-length', 0))
        progress.update(download_task, total=total_size)
        
        with open(tar_file, 'wb') as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                progress.update(download_task, advance=len(chunk))
    
    # Extract with progress tracking in a separate progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        transient=False
    ) as progress:
        extract_task = progress.add_task("[yellow]Extracting dataset...", total=None)
        with tarfile.open(tar_file, 'r:gz') as tar:
            members = tar.getmembers()
            progress.update(extract_task, total=len(members))
            for member in members:
                tar.extract(member, path=data_path)
                progress.advance(extract_task)
    
    # Clean up
    tar_file.unlink()
    console.print("[green]✓[/green] Dataset downloaded and extracted successfully!")

def start_data_process_questionary():
    """Display an interactive menu for dataset setup options.
    
    Returns:
        str: Selected option from the menu:
            - 'Download sample dataset (recommended)'
            - 'I have it'
            - 'Exit'
    
    Displays information about the dataset and provides options for
    setting up the training data.
    """
    console.print(Panel.fit(
        "[info]The project uses the [arg]102 Category Flower Dataset.[/info]\n" 
        "[info]More information can be found [example](ctr + click)[/example] [arg][link=https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html]here[cyan]↗️[/cyan][/link][/arg]\n\n"
        "[info]Use the following menu to choose[/info]\n"
        "[info]how would you like to set up your dataset[/info]\n",
        title="Dataset Information",
        border_style="title"
    ))
    return questionary.select(
        "Choose an option",
        choices=[
            "Download sample dataset (recommended)",
            "I have it",
            "Exit"
        ],
        style=questionary.Style([
            ('qmark', 'fg:yellow bold'),
            ('question', 'bold'),
            ('answer', 'fg:green bold'),
            ('pointer', 'fg:yellow bold'),
            ('highlighted', 'fg:yellow'),
            ('selected', 'fg:green'),
        ])
    ).ask()

def questionary_default_style():
    """Create a consistent style configuration for questionary prompts.
    
    Returns:
        questionary.Style: Configured style with custom colors and formatting
        for interactive prompts.
    """
    return questionary.Style([
            ('qmark', 'fg:yellow bold'),
            ('question', 'bold'),
            ('answer', 'fg:green bold'),
            ('pointer', 'fg:yellow bold'),
            ('highlighted', 'fg:yellow'),
            ('selected', 'fg:green')
        ])

class CustomClassifier(nn.Module):
    """Custom neural network classifier for the flower image classifier.
    
    This classifier is designed to be attached to pretrained models and includes:
    - Configurable input and output sizes
    - Multiple hidden layers with customizable units
    - Dropout for regularization
    - ReLU activation functions
    """
    
    def __init__(self, args):
        """Initialize the custom classifier.
        
        Args:
            args: Arguments containing:
                - input_size: Size of input features
                - hidden_units: List of hidden layer sizes
                - output_size: Number of output classes
                - drop_p: Dropout probability
        """
        super().__init__()
        self.args = args
        self.drop_p = float(self.args.drop_p)
        self.input_size = int(self.args.input_size)
        self.output_size = int(self.args.output_size)
        # Initialize network layers
        self.all_layers = nn.ModuleList()
        
        # Define input layer
        self.all_layers.append(nn.Linear(self.input_size, self.args.hidden_units[0])) 
        self.all_layers.append(nn.ReLU())
        self.all_layers.append(nn.Dropout(p=self.drop_p))
        
        # Define hidden layers
        for i in range(len(self.args.hidden_units) - 1):  # Note the -1 here
            self.all_layers.append(nn.Linear(self.args.hidden_units[i], self.args.hidden_units[i + 1]))
            self.all_layers.append(nn.ReLU())
            self.all_layers.append(nn.Dropout(p=self.drop_p))
            
        # Define output layer
        self.all_layers.append(nn.Linear(self.args.hidden_units[-1], self.output_size))
        self.all_layers.append(nn.LogSoftmax(dim=1))
        
    def forward(self, x):
        """Forward pass through the classifier.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output probabilities (log-softmax)
        """
        for layer in self.all_layers:
            x = layer(x)
        return x


def print_model_classifier(classifier, message=""):
    """Display the model classifier architecture with Rich formatting.
    
    Args:
        classifier (nn.Module): The classifier to display
        message (str, optional): Additional message to display before
            the architecture. Defaults to "".
    
    Formats and displays the classifier architecture with color coding
    and proper indentation using Rich panels.
    """
    # Display model classifier architecture with custom formatting
    classifier_str = str(classifier)
    
    # Parse and format the classifier string
    lines = classifier_str.split('\n')
    formatted_lines = []
    
    # Format the Sequential header
    formatted_lines.append("[purple]Sequential([/purple]")
    
    # Process each layer
    for line in lines[1:-1]:  # Skip first and last lines (Sequential( and ))
        if not line.strip():
            continue
            
        # Extract the layer number, type and parameters
        parts = line.strip().split(':', 1)
        if len(parts) == 2:
            layer_num = parts[0].strip('() ')
            layer_info = parts[1].strip()
            
            # Split layer type and parameters
            layer_type = layer_info.split('(', 1)[0]
            layer_params = '(' + layer_info.split('(', 1)[1] if '(' in layer_info else ''
            
            # Format with colors
            formatted_line = f"  [example]({layer_num}):[/example] [green]{layer_type}[/green][info]{layer_params}[/info]"
            formatted_lines.append(formatted_line)
    
    # Add closing bracket
    formatted_lines.append("[purple])[/purple]")
    
    # Join all lines
    formatted_str = message + '\n'.join(formatted_lines)
    
    # Print the formatted string with a title
    console.print(Panel(
        formatted_str, 
        border_style="title"
    ))

    console.print("Press Enter to continue...")
    input("")

def get_model(model_name, for_training=False):
    """Get a pretrained model by name.
    
    Args:
        model_name (str): Name of the model to load ('vgg11', 'vgg13',
            'vgg16', 'vgg19')
        for_training (bool): If True, freeze parameters for transfer learning.
            If False (prediction mode), no need to freeze parameters.
    
    Returns:
        torchvision.models: Pretrained model instance
    
    Raises:
        SystemExit: If the model name is not supported
    """
    model_mapping = {
        "vgg11": (models.vgg11, models.VGG11_Weights.DEFAULT),
        "vgg13": (models.vgg13, models.VGG13_Weights.DEFAULT),
        "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
        "vgg19": (models.vgg19, models.VGG19_Weights.DEFAULT)
    }
    
    if model_name not in model_mapping:
        console.print(f"[error]Error:[/error] Unsupported model: [arg]{model_name}[/arg]")
        console.print(f"[info]Available models:[/info] [desc]{', '.join(model_mapping.keys())}[/desc]")
        sys.exit(1)
    
    model_func, weights = model_mapping[model_name]
    model = model_func(weights=weights)
    
    # Freeze parameters only if we're training
    if for_training:
        for param in model.parameters():
            param.requires_grad = False
    
    console.print(f"[example][✓][/example] The model [arg]'{model_name}'[/arg] was successfully loaded")
    console.print(f"[example][✓][/example] The model parameters were successfully frozen")
    return model

def define_device(gpu):
    """Determine the appropriate device for model training.
    
    Args:
        gpu (bool): Whether to use GPU if available
    
    Returns:
        torch.device: Device to use for training ('cuda' or 'cpu')
    """
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    if gpu and str(device) == "cpu":
        console.print(f"[error][❌] GPU is not available[/error]\n")

    console.print(f"[example][✓][/example] Set the device to:[arg]'{device}'[/arg]")
    return device


def select_file(title, filetypes):
    """Open a file selection dialog.
    
    Args:
        title (str): Title for the file dialog window
        filetypes (list): List of tuples containing file type descriptions
            and patterns
    
    Returns:
        str: Selected file path or None if cancelled
    """
    root = Tk()
    root.withdraw()  # Hide the main window
    file = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    root.destroy()

    return file