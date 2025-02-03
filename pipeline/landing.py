import sys
import os
from utils import log, console, questionary_default_style, get_train_terminal_args
from rich.panel import Panel
import questionary

class WelcomeMessage:
    """
    Welcome message with the training parameters menu for specifying the training parameters.
    """
    def __init__(self):
        self.display_welcome_message()
        self.continue_to_data_directory_setup()

    def display_welcome_message(self):
        """Interactive menu for selecting training parameters."""
    
        # Style for questionary
        style = questionary_default_style()

        # Training parameters menu
        console.print(Panel.fit(
            "[desc]Welcome to Flowers Image Classifier Training![/desc]\n\n"
            "[desc]You are about to start training a neural network to classify different types of flowers.\n"
            "Please answer the following questions to specify the training parameters.[/desc]\n\n"
            "[info]Press Enter to use default values[/info]",
            title="Flowers Image Classifier",
            border_style="title"
        ))

        args = get_train_terminal_args()
        data_dir = args.data_dir if args else None
        # Required data directory input
        # Skip if a user provided it in the command line
        if not data_dir:
            while True:
                data_dir = questionary.text(
                    "(Required) Where do you want your dataset to live? (data/flowers)|(exit to quit):",
                    style=style
                ).ask()

                if data_dir == 'exit':
                    sys.exit("Exiting...")

                if data_dir:
                    # Add data directory to args
                    sys.argv.append(data_dir)
                    break
                else:
                    console.print("[error]Dataset directory is required to continue[/error]")

        # Save directory
        save_dir = questionary.text(
            "Save Directory: Where to save model checkpoints (default: checkpoints):",
            default="checkpoints",
            style=style
        ).ask()

        # Architecture selection
        arch = questionary.select(
            "Model Architecture: Neural network backbone (default: vgg16):",
            choices=[
                "vgg16 (default)",
                "vgg11",
                "vgg13",
                "vgg19"
            ],
            default="vgg16 (default)",
            style=style
        ).ask()

        # Hidden units input
        hidden_units = questionary.text(
            "Hidden Units: Neurons in hidden layers (default: 4096,1024):",
            validate=lambda x: all(u.isdigit() and int(u) > 0 for u in x.split(',')) if x else True,
            style=style
        ).ask()

        # Learning rate input with better validation
        def validate_float(text):
            if not text:  # Allow empty for default
                return True
            try:
                value = float(text)
                return 0 < value <= 1
            except ValueError:
                return False

        learning_rate = questionary.text(
            "Learning Rate: How fast model learns (default: 0.0001):",
            validate=validate_float,
            style=style
        ).ask()

        # Epochs input
        epochs = questionary.text(
            "Epochs: Training cycles (default: 17):",
            validate=lambda x: x.isdigit() and 0 < int(x) <= 100 if x else True,
            style=style
        ).ask()

        # GPU option
        gpu = questionary.confirm(
            "GPU: Use GPU acceleration (default: False):\n",
            default=False,
            style=style
        ).ask()

        # Update sys.argv for argument parsing
        sys.argv = [sys.argv[0]]  # Keep the script name
        sys.argv.append(data_dir)  # Add data directory as positional arg
        
        if save_dir and save_dir != "checkpoints":
            sys.argv.extend(['--save_dir', save_dir])
        if arch and 'default' not in arch:
            sys.argv.extend(['--arch', arch.split()[0]])
        if hidden_units:
            units = [int(u) for u in hidden_units.split(',')]
            sys.argv.extend(['--hidden_units', str(units)])
        if learning_rate:
            sys.argv.extend(['--learning_rate', learning_rate])
        if epochs:
            sys.argv.extend(['--epochs', epochs])
        if gpu:
            sys.argv.append('--gpu')


    def continue_to_data_directory_setup(self):
        # Continue with dataset direcotry setup
        console.print(Panel.fit(
            "[desc]The following steps will setup the dataset directory.[/desc]\n"
            "[desc]To continue type [arg](y|yes)[/arg][/desc]\n"
            "[info]Press Enter to use default value [arg](n|no)[/arg][/info]",
            title="Dataset Directory Setup",
            border_style="title"
        ))

        # Style for questionary
        style = questionary_default_style()

        # Ask if user wants to continue
        if not questionary.confirm(
            "Continue?",
            default=False,
            style=style
        ).ask():
            console.print(f"Exiting...")
            sys.exit()

        return True

