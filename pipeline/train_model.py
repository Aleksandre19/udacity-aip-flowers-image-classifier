import os
import sys
import torch
import torch.hub
import urllib.request
from constants import MODEL_TRAIN_MESSAGE
from torchvision import models
from utils import console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

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
    
    # Get model URL and create progress bar
    model_url = weights.url
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        transient=False
    ) as progress:
        # Get total file size first
        response = urllib.request.urlopen(model_url)
        total_size = int(response.headers.get('content-length', 0))
        response.close()
        
        # Start download task with known total
        download_task = progress.add_task(
            f"[yellow]Downloading {model_name.upper()} model...",
            total=total_size
        )
        
        # Save original download function
        original_load_state_dict = torch.hub.load_state_dict_from_url
        
        def download_progress_hook(count, block_size, total_size):
            progress.update(download_task, completed=count * block_size)
        
        try:
            # Download the model file directly using urllib
            filename = model_url.split('/')[-1]
            cache_dir = torch.hub._get_torch_home()
            cached_file = os.path.join(cache_dir, self.args.save_dir, filename)
            os.makedirs(os.path.dirname(cached_file), exist_ok=True)
            
            urllib.request.urlretrieve(model_url, cached_file, reporthook=download_progress_hook)
            
            # Load the downloaded model
            model = model_func(weights=weights)
        finally:
            torch.hub.load_state_dict_from_url = original_load_state_dict
    
    console.print(f"[info]Model architecture:[/info] [desc]{model.classifier}[/desc]\n")
    return model