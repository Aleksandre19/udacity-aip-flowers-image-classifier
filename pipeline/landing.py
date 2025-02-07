# Standard library imports
import sys

# Third-party imports
import questionary
from rich.panel import Panel

# Local imports
from utils import (
    console,
    get_train_terminal_args,
    questionary_default_style
)

class WelcomeMessage:
    """Handles the welcome interface and training parameter configuration.
    
    This class manages the initial setup process for the flower classifier,
    including parameter configuration through an interactive menu and
    data directory setup.
    """
    def __init__(self, retrain=False):
        """Initialize the WelcomeMessage class.
        
        Args:
            retrain (bool): If True, indicates this is a retraining session
                           and skips data directory setup.
        """
        self.retrain = retrain
        self._args = None
        self.display_welcome_message()
        if not self.retrain: 
            self.continue_to_data_directory_setup()

    def display_welcome_message(self):
        """Display interactive menu for configuring training parameters.
        
        Presents a series of prompts for the user to configure:
        - Save directory for model checkpoints
        - Model architecture selection
        - Network architecture parameters (input, hidden, output sizes)
        - Training parameters (dropout, learning rate, epochs)
        - Hardware acceleration options (GPU)
        
        All inputs have default values that can be accepted by pressing Enter.
        """
    
        # Style for questionary
        style = questionary_default_style()

        if not self.retrain:
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

        # Validation interval
        validation_interval = questionary.text(
            f"Validation Interval: How often to validate (current: {self._args.valid_interval}):",
            validate=lambda x: x.isdigit() if x else False,
            default=f"{self._args.valid_interval}",
            style=style
        ).ask()
        self._args.valid_interval = int(validation_interval)

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
        """Get the configured training arguments.
        
        Returns:
            argparse.Namespace: Object containing all training parameters
        """
        return self._args



    def continue_to_data_directory_setup(self):
        """Prompt for data directory setup continuation.
        
        Displays a prompt asking if the user wants to proceed with
        dataset directory setup. Exits the program if user declines.
        
        Returns:
            bool: True if user wants to continue, exits otherwise
        """
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
