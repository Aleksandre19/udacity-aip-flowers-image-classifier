import argparse
import logging
import sys
import tarfile
import urllib.request
from pathlib import Path

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
    'info': 'yellow'
})

dark_theme = Theme({
    'arg': 'bright_blue',
    'desc': 'bright_green',
    'example': 'bright_cyan',
    'title': 'bright_cyan',
    'error': 'bright_red',
    'info': 'bright_yellow'
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
            "[info]Command Line Arguments Information[/info]\n\n"
            "• [arg]data_dir[/arg]: [desc]Directory containing your training images[/desc] (Required)\n"
            "• [arg]--save_dir[/arg]: [desc]Save trained model checkpoints to this directory[/desc] (default: checkpoints)\n"
            "• [arg]--arch[/arg]: [desc]Neural network architecture to use[/desc] (default: vgg16)(avaliables: vgg11, vgg13, vgg19)\n"
            "• [arg]--learning_rate[/arg]: [desc]How fast the model learns during training[/desc] (default: 0.0001)\n"
            "• [arg]--hidden_units[/arg]: [desc]Number of neurons in hidden layers[/desc] (default: [4096, 1024])\n"
            "• [arg]--epochs[/arg]: [desc]Number of complete training cycles[/desc] (default: 17)\n"
            "• [arg]--gpu[/arg]: [desc]Use GPU for faster training if available[/desc] (default: False)\n\n"
            "[info]Example:[/info]\n"
            "[example]python train.py flowers/dataset --arch vgg16 --gpu[/example]",
            title="Training Model",
            border_style="title"
        ))
        parser.exit()


def get_train_terminal_args():
    """
    Define terminal arguments for training a model.
    """

    # Check if we have any arguments before creating parser
    if len(sys.argv) == 1:
        return None

    parser = CustomArgumentParser(
        description='Train a neural network on a dataset of images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage='example: python3 train.py "data/directory"'
    )

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

    parser.add_argument('--info',
                        action=InfoAction,
                        nargs=0,
                        help='show detailed information about the script')
                            
    parser.add_argument('--learning_rate', 
                        type=float,
                        default=0.0001,
                        help='learning rate for training')

    parser.add_argument('--hidden_units', 
                        type=int,
                        default=[4096, 1024],
                        help='number of units in hidden layers')

    parser.add_argument('--epochs', 
                        type=int,
                        default=17,
                        help='number of epochs for training')

    parser.add_argument('--gpu', 
                        action='store_true',
                        help='use GPU for training if available')

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
    return questionary.select(
                    "How would you like to set up your dataset?",
        choices=[
            "Download sample dataset (recommended)",
            "I'll provide my own dataset",
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

