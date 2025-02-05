import sys
from utils import get_train_terminal_args, log
from pipeline import WelcomeMessage, ProcessDataStructure, PreprocessData, TrainModel
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

        # Transforming dataset and creating DataLoaders for 
        # train, validation and test datasets
        PreprocessData.start(args.data_dir)

        # Start training
        TrainModel.start(start_training.args)

    except Exception as e:
        log.error(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()