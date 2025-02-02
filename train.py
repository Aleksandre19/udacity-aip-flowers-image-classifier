from utils import get_train_terminal_args, log
from pipeline.landing import WelcomeMessage
from pipeline.process_data import ProcessDataStructure
from rich import console

console = console.Console()

def main():
    try:
        # Welcome message
        start_training = WelcomeMessage()

        # If the user does not want to continue, exit
        if not start_training:
            return

        # Check if we need to get data directory interactively
        args = get_train_terminal_args()
        if args is None:
            start_training.set_data_directory()
            args = get_train_terminal_args()

        # Handle data directory
        start_training.handle_data_directory(args.data_dir)

        # Check and validate data directory and dataset structure
        ProcessDataStructure.start(args.data_dir)

    except Exception as e:
        log.error(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()