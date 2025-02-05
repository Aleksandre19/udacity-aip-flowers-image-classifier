from ast import arg
import sys
import os
import turtle
from utils import log, console, questionary_default_style, get_train_terminal_args
from rich.panel import Panel
import questionary

class WelcomeMessage:
    """
    Welcome message with the training parameters menu for specifying the training parameters.
    """
    def __init__(self):
        self._args = None
        self.display_welcome_message()
        self.continue_to_data_directory_setup()

    def display_welcome_message(self):
        """Interactive menu for selecting training parameters."""
    
        # Style for questionary
        style = questionary_default_style()

        # Training parameters menu
        console.print(Panel.fit(
            "[info]Welcome to Flowers Image Classifier Training![/info]\n\n"
            "[desc]You are about to start training a neural network to classify different types of flowers.\n"
            "Please answer the following questions to specify the training parameters.[/desc]\n\n"
            "[info]Press Enter to use default values[/info]",
            title="Flowers Image Classifier",
            border_style="title"
        ))

        self._args = get_train_terminal_args()

        # Save directory
        save_dir = questionary.text(
            f"Save Directory: Where to save model checkpoints (current: {self._args.save_dir}):",
            default=f"{self._args.save_dir}",
            style=style
        ).ask()
        self._args.save_dir = save_dir

        # Architecture selection
        arch = questionary.select(
            f"Model Architecture: Neural network backbone (current: {self._args.arch}):",
            choices=[
                "vgg16 (default)",
                "vgg11",
                "vgg13",
                "vgg19"
            ],
            default="vgg16 (default)",
            style=style
        ).ask()
        self._args.arch = [a for a in arch.split() if len(a) > 0][0]

        # Input size
        input_size   = questionary.text(
            f"Input Size: Number of input units (current: {self._args.input_size}):",
            validate=lambda x: x.isdigit() if x else False,
            default=f"{self._args.input_size}",
            style=style
        ).ask()
        self._args.input_size = int(input_size)

        # Hidden units input
        hidden_units = questionary.text(
            f"Hidden Units: Neurons in hidden layers (current: {','.join(map(str, self._args.hidden_units))}):",
            validate=lambda x: bool(x.strip() and all(u.strip().isdigit() and int(u.strip()) > 0 for u in x.split(','))),
            default=f"{','.join(map(str, self._args.hidden_units))}",
            style=style
        ).ask()
        self._args.hidden_units = list(map(int, hidden_units.split(',')))

        # Output size
        output_size   = questionary.text(
            f"Output Size: Number of output units (current: {self._args.output_size}):",
            validate=lambda x: x.isdigit() if x else False,
            default=f"{self._args.output_size}",
            style=style
        ).ask()
        self._args.output_size = int(output_size)

         # Learning rate input with better validation
        def validate_float(text):
            if not text:  # Allow empty for default
                return True
            try:
                value = float(text)
                return 0 < value <= 1
            except ValueError:
                return False
        
        drop_p = questionary.text(
            f"Dropout Probability: How much to drop out probability? (current: {self._args.drop_p}):",
            validate=validate_float,
            default=f"{self._args.drop_p}",
            style=style
        ).ask()
        self._args.drop_p = drop_p

        learning_rate = questionary.text(
            f"Learning Rate: How fast model learns (current: {self._args.learning_rate}):",
            validate=validate_float,
            default=f"{self._args.learning_rate}",
            style=style
        ).ask()
        self._args.learning_rate = float(learning_rate)

        # Epochs input
        epochs = questionary.text(
            f"Epochs: Training cycles (current: {self._args.epochs}):",
            validate=lambda x: x.isdigit() and 0 < int(x) <= 100 if x else False,
            default=f"{self._args.epochs}",
            style=style
        ).ask()
        self._args.epochs = int(epochs)

        # GPU option
        gpu = questionary.confirm(
            f"GPU: Use GPU acceleration (current: {self._args.gpu}):",
            default=self._args.gpu,
            style=style
        ).ask()
        self._args.gpu = gpu

        print("")

    @property
    def args(self):
        return self._args



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
