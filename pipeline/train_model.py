import sys
import torch
import torch.hub
from constants import MODEL_TRAIN_MESSAGE
from torchvision import models
from utils import console
from rich.panel import Panel

class TrainModel:
  def __init__(self, args):
    self.args = args
    print(self.args.arch)
    self.pre_train_message
    self.device = self._define_device()
    console.print(f"[example][✓][/example] Set the device to:[arg]'{self.device}'[/arg]")
    self.model = self._get_model(self.args.arch)

  @staticmethod
  def start(args):
    train = TrainModel(args)

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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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