"""Main prediction script for the flower image classifier.

This script handles the prediction workflow including:
- Loading a trained model from a checkpoint
- Processing input images
- Making and displaying predictions

The script can be run with command line arguments or in interactive mode.
"""

# Local imports
from pipeline import LoadModel, MakePrediction


def main():
    """Execute the main prediction workflow.
    
    The workflow consists of two main steps:
    1. Load a trained model from a checkpoint file
    2. Make predictions on new images
    
    The user will be prompted to:
    - Select a model checkpoint if not specified via command line
    - Choose images for prediction
    - Select category-to-name mapping file
    """
    loaded_model = LoadModel()

    prediction = MakePrediction.start(loaded_model.model)


if __name__ == '__main__':
    main()