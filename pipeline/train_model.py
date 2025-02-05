import sys
import torch
import torch.hub
from constants import MODEL_TRAIN_MESSAGE, CURRENT_MODEL_ARCHITECTURE_MESSAGE
from torchvision import models
from utils import console
from rich.panel import Panel

class TrainModel:
  def __init__(self, args):
    self.args = args
    self.pre_train_message
    self.device = self._define_device()
    self.model = self._get_model(self.args.arch)

  @staticmethod
  def start(args):
    train = TrainModel(args)
    train._print_model_classifier()

  @property
  def pre_train_message(self):
    console.print(Panel.fit(
        MODEL_TRAIN_MESSAGE,
        title="Model Training",
        border_style="title"
    ))

    input("Press Enter to start...\n")

    console.print(f"[example][→][/example] Starting Model Training...")

  def _define_device(self):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[example][✓][/example] Set the device to:[arg]'{device}'[/arg]")
    return device

  def _get_model(self, model_name):
    model_mapping = {
      "vgg11": (models.vgg11, models.VGG11_Weights.DEFAULT),
      "vgg13": (models.vgg13, models.VGG13_Weights.DEFAULT),
      "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
      "vgg19": (models.vgg19, models.VGG19_Weights.DEFAULT)
    }
    
    if model_name not in model_mapping:
        console.print(f"[error]Error:[/error] Unsupported model: [arg]{model_name}[/arg]")
        console.print(f"[info]Available models:[/info] [desc]{', '.join(model_mapping.keys())}[/desc]")
        sys.exit(1)
    
    model_func, weights = model_mapping[model_name]
    model = model_func(weights=weights) 

    console.print(f"[example][✓][/example] The model [arg]'{self.args.arch}'[/arg] was successfully loaded")
    return model


  def _print_model_classifier(self):
    """
    Print the current model classifier architecture with custom formatting.
    """
    # Display model classifier architecture with custom formatting
    classifier_str = str(self.model.classifier)
    
    # Parse and format the classifier string
    lines = classifier_str.split('\n')
    formatted_lines = []
    
    # Format the Sequential header
    formatted_lines.append("[purple]Sequential([/purple]")
    
    # Process each layer
    for line in lines[1:-1]:  # Skip first and last lines (Sequential( and ))
        if not line.strip():
            continue
            
        # Extract the layer number, type and parameters
        parts = line.strip().split(':', 1)
        if len(parts) == 2:
            layer_num = parts[0].strip('() ')
            layer_info = parts[1].strip()
            
            # Split layer type and parameters
            layer_type = layer_info.split('(', 1)[0]
            layer_params = '(' + layer_info.split('(', 1)[1] if '(' in layer_info else ''
            
            # Format with colors
            formatted_line = f"  [example]({layer_num}):[/example] [green]{layer_type}[/green][info]{layer_params}[/info]"
            formatted_lines.append(formatted_line)
    
    # Add closing bracket
    formatted_lines.append("[purple])[/purple]")
    
    # Join all lines
    formatted_str = CURRENT_MODEL_ARCHITECTURE_MESSAGE + '\n'.join(formatted_lines)
    
    # Print the formatted string with a title
    console.print(Panel(
      formatted_str, 
      border_style="title"
    ))

    console.print("Press Enter to continue...")
    input("")