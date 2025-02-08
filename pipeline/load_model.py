# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import torch
from rich.panel import Panel
from tkinter import Tk, filedialog
from torchvision import models

# Local imports
from constants import CHOOSE_MODEL_ERROR_MESSAGE
from utils import (
  console,
  CustomClassifier,
  get_model,
  get_predict_terminal_args,
  print_model_classifier,
  questionary_default_style,
  select_file
)

class LoadModel:
  """Handles model loading and initialization for the flower classifier.
  
  This class manages the complete model loading workflow including:
  - Loading saved model checkpoints
  - Reconstructing model architecture
  - Setting up custom classifier layers
  - Restoring model state and class mappings
  """
  
  def __init__(self):
    """Initialize the LoadModel class.
    
    Sets up command line arguments, prompts for model selection,
    and loads the specified model checkpoint.
    """
    self.args = get_predict_terminal_args()  # Get command line arguments
    self.model = None  # Will store the loaded model
    self.setup_questionary()  # Prompt for model selection
    self.load_model()  # Load and initialize the model

  def setup_questionary(self):
    """Handle model file selection and validation.
    
    Provides a file dialog for model selection if not specified via command line.
    Validates the model file path and converts to relative path if possible.
    Exits with appropriate error messages if model selection fails.
    """
    style = questionary_default_style()
    
    if not self.args.model:
      # Choose model file
      console.print(f"[example][→] Please select the model file from the file dialog...[/example]\n")
      model = select_file(
        title="Choose the model to make predictions with",
        filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")]
      )
      
      if not model:  # If user cancels the dialog
          console.print(Panel.fit(
              CHOOSE_MODEL_ERROR_MESSAGE,
              title="Specify Model",
              border_style="title"
          ))
          sys.exit("Exiting...")

      # Convert absolute path to relative path
      try:
          model_path = Path(model)
          relative_path = model_path.relative_to(Path.cwd())
          self.args.model = str(relative_path)
      except ValueError:
          print("Warning: Selected file is outside the current working directory")
          self.args.model = model
    
    if not os.path.exists(self.args.model):
        console.print(Panel.fit(
                f"[error][❌] The model file [info]`{self.args.model}`[/info] does not exist[/error]\n\n"
                f"[example]Please specify the correct model file again with one of the following options:[/example]\n"
                f"• [desc] Run command: [info]`python3 predict.py --model 'model/folder/model_name.pth`[/info] [/desc]\n"
                f"• [desc] Or you can run only: [info]`python3 predict.py`[/info] and choose a model from the file dialog window![/desc]\n",
                title="Model Does Not Exist",
                border_style="title"
        ))
        sys.exit(1)
    
  
  def load_model(self):
    """Load and initialize the model from a checkpoint file.
    
    Performs the following steps:
    1. Loads the model checkpoint
    2. Reconstructs the model architecture
    3. Updates model parameters from checkpoint
    4. Creates and sets up the custom classifier
    5. Restores class index mappings
    
    Exits with error message if loading fails.
    """
    try:
      # Load the checkpoint
      checkpoint = torch.load(self.args.model)
      
      # Get the base model architecture
      self.model = get_model(checkpoint['arch'])
      
      # Update args with the loaded parameters
      self.args.input_size = checkpoint['input']
      self.args.output_size = checkpoint['output']
      self.args.hidden_units = checkpoint['hidden_layers']
      self.args.drop_p = checkpoint['dropout']
      
      # Create and set the custom classifier
      classifier = CustomClassifier(self.args)
      self.model.classifier = classifier
      
      self.model.class_to_idx = checkpoint['class_to_idx']

      # Reverse class_to_index to idx_to_class
      self.model.idx_to_class = {v:k for k,v in self.model.class_to_idx.items()}
      
      # Load the state dict
      self.model.load_state_dict(checkpoint['state_dict'])

      print_model_classifier(self.model.classifier)
      
    except Exception as e:
      console.print(f"[error][❌] Failed to load model: {str(e)}[/error]")
      sys.exit(1)
