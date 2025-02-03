import sys
from utils import get_train_terminal_args, log
from pipeline.landing import WelcomeMessage
from pipeline.process_data import ProcessDataStructure
from rich import console

console = console.Console()

def main():
    try:
        # Parse command line args to check for --info
        args = get_train_terminal_args()
        if args and args.info:
            return
        # Welcome message
        start_training = WelcomeMessage()

        # If the user does not want to continue, exit
        if not start_training:
            return

        # Check and validate data directory and dataset structure
        ProcessDataStructure.start(args.data_dir)

    except Exception as e:
        log.error(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()