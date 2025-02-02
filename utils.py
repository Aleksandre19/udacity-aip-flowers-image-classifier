import argparse
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme
import sys

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



class WelcomeMessage:
    """
    Display a welcome message before starting the training process 
    and ask the user if they want to continue.
    If the user does not want to continue, exit.
    """
    def __init__(self):
        self.display_welcome_message()
        self.start_training()

    def display_welcome_message(self):
        console.print(Panel.fit(
            "[info]Welcome to Flowers Image Classifier Training![/info]\n\n"
            "You are about to start training a neural network to classify different types of flowers.\n\n"
            "• [arg]Dataset[/arg]: [desc]Make sure your dataset is properly organized in the data directory[/desc]\n"
            "• [arg]Model[/arg]: [desc]You can choose from various VGG architectures (vgg11, vgg13, vgg16, vgg19)[/desc]\n"
            "• [arg]GPU[/arg]: [desc]Training can be accelerated using GPU if available[/desc]\n"
            "• [arg]Checkpoints[/arg]: [desc]Your trained model will be saved in the checkpoints directory[/desc]\n\n"
            "[info]Getting Started:[/info]\n"
            "• Use [example]python train.py --info[/example] to see all available options\n"
            "• Start training with [example]python train.py data/flowers[/example]",
            title="Flowers Image Classifier",
            border_style="title"
        ))

    def start_training(self):
        # Ask the user if they want to continue
        console.print("[info]Start Training? [desc](y/N)[/desc][/info]\n")
        welcome = input().lower()

        # If the user does not want to continue, exit
        if welcome != 'y' and welcome != 'yes':
            sys.exit("Exiting...")
            return False

        return True

    def set_data_directory(self):
        # Ask for data directory
        console.print("[info]Please specify the data directory:[/info] [example](e.g., data/flowers)[/example]", end=" ")
        data_dir = input().strip()
        
        # If the user does not provide a data directory, exit
        if not data_dir:
            return
            
        # Append data directory to terminal arguments
        sys.argv.append(data_dir)