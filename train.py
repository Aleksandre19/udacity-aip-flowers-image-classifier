"""Main training script for the flower image classifier.

This script orchestrates the complete training workflow including:
- Parameter configuration through command line or interactive menu
- Dataset validation and preprocessing
- Model training and evaluation
- Checkpoint saving

The script can be run with command line arguments or in interactive mode.
Use --info flag to see available command line options.
"""

# Standard library imports
import sys

# Third-party imports
from rich import console

# Local imports
from pipeline import (
    ProcessDataStructure,
    PreprocessData,
    TrainModel,
    WelcomeMessage
)
from utils import get_train_terminal_args, log

# Initialize rich console
console = console.Console()

def main():
    """Execute the main training workflow.
    
    The workflow consists of the following steps:
    1. Parse command line arguments
    2. Display welcome message and get training parameters
    3. Validate dataset structure
    4. Preprocess data and create DataLoaders
    5. Train the model
    
    Raises:
        Exception: Any unexpected errors during training
    """
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
        processed_data = PreprocessData.start(args.data_dir)

        # Start training
        TrainModel.start(start_training.args, processed_data)

    except Exception as e:
        log.error(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()