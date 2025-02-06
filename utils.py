import argparse
import logging
import sys
import tarfile
import urllib.request
from pathlib import Path

from torch import nn
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
import questionary
from constants import PROVIDE_DATA_RICH_MESSAGE, CHOOSE_MODEL_ERROR_MESSAGE


def setup_logger():
    """Set up and configure the logger with Rich formatting."""
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
    """
    Custom ArgumentParser to override the default error message and format.
    """
    def error(self, message):
        console.print(f"[error]error: Data directory is required in order to train a model![/error]")
        console.print(f"[example]- Example: python3 train.py data/directory[/example]")
        console.print("------------------------------------------")
        console.print(f"[info]For more information: python3 train.py --info[/info]")
        self.exit(2)
        


class InfoAction(argparse.Action):
    """
    Override the default argparse.Action class to apply custom formatting to the --info action.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        console.print(Panel.fit(
            f"{PROVIDE_DATA_RICH_MESSAGE}",
            title="Flowers Image Classifier Information",
            border_style="title"
        ))
        parser.exit()


def get_train_terminal_args():
    """
    Define terminal arguments for training a model.
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
                        help='number of validation interval')

    parser.add_argument('--gpu', 
                        action='store_true',
                        help='use GPU for training if available')

    return parser.parse_args()


class CustomPredictArgumentParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser to override the default error message and format.
    """
    def error(self, message):
        console.print(Panel.fit(
            CHOOSE_MODEL_ERROR_MESSAGE,
            title="Specify Model",
            border_style="title"
        ))
        self.exit(2)  
        

def get_predict_terminal_args():
    """
    Define terminal arguments for prediction.
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
    return questionary.Style([
            ('qmark', 'fg:yellow bold'),
            ('question', 'bold'),
            ('answer', 'fg:green bold'),
            ('pointer', 'fg:yellow bold'),
            ('highlighted', 'fg:yellow'),
            ('selected', 'fg:green')
        ])

class CustomClassifier(nn.Module):
    def __init__(self, args):
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
        for layer in self.all_layers:
            x = layer(x)
        return x