import sys
import os
from utils import log, console
from rich.panel import Panel

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
        while True:
            console.print(f"[info]Please specify the data directory[/info]" 
                          f"[example](e.g., data/flowers)[/example] or type" 
                          f"[error]exit[/error] to quit: ", end=" ")
            data_dir = input().strip().lower()

            if data_dir == 'exit':
                sys.exit("Exiting...")

            if data_dir:       
                break
            
        # Append data directory to terminal arguments
        sys.argv.append(data_dir)


    def handle_data_directory(self, data_dir):
        """Handle data directory creation if it doesn't exist."""
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            log.info(f"Created data directory:'{data_dir}'")
        else:
            log.info(f"Using existing data directory:'{data_dir}'")