import os
import sys
import plotext as plt

from pathlib import Path
from tkinter import Tk, filedialog

import json
from pathlib import Path
from constants import CHOOSE_IMAGE_ERROR_MESSAGE
from rich.panel import Panel
from utils import console, questionary_default_style, define_device

import torch
import questionary
from PIL import Image
from torchvision import transforms

class MakePrediction:
    def __init__(self, model):
        self.model = model
        self.image = None
        self.image_path = None
        self.device = None
        self.topk = 5
        self.cat_to_name = None
        
    
    @staticmethod
    def start(model):
        prediction = MakePrediction(model)
        prediction.predict_questionary()

    def predict_questionary(self):
        self.choose_image()
        self.transform_image()
        self.preidct_questionary()
        probs, classes = self.predict()
        self.map_cat_to_name()
        self.print_predictions(probs, classes)
        self.predict_again()

    def choose_image(self):
        console.print(f"[example][→] Please select the image[/example]")
        root = Tk()
        root.withdraw()  # Hide the main window
        image = filedialog.askopenfilename(
            title="Choose the image to make predictions with",
            filetypes=[
                ("JPEG files", "*.jpg *.jpeg"),
            ]
        )
        root.destroy()
        
        if not image:  # If user cancels the dialog
            console.print(Panel.fit(
                CHOOSE_IMAGE_ERROR_MESSAGE,
                title="Specify Image",
                border_style="title"
            ))
            sys.exit("Exiting...")
        
        image_path = Path(image)
        self.image_path = image_path.relative_to(Path.cwd())
        self.image = self.image_path

        console.print(f"[example][✓][/example] Image is selected:[arg]'{self.image_path}'[/arg]")

    def transform_image(self):
        # Load the image using PIL
        with Image.open(self.image) as pil_image:
            transform = transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
            ])
            
            # Transform the PIL image
            self.image = transform(pil_image)
        
        console.print(f"[example][✓][/example] The image was transformed successfully")
    
    def preidct_questionary(self):
        topk   = questionary.text(
            f"Top K: Number of top predictions to display (current: {self.topk}):",
            validate=lambda x: x.isdigit() if x else False,
            default=f"{self.topk}",
            style=questionary_default_style()
        ).ask()
        self.topk = int(topk)

        console.print(f"[example][✓][/example] Top K is selected:[arg]'{self.topk}'[/arg]")

        # GPU option
        gpu = questionary.confirm(
            f"GPU: Use GPU acceleration (current: {False if not self.device else True}):",
            default=False,
            style=questionary_default_style()
        ).ask()

        self.device = define_device(gpu)


    def predict(self):
        # Move model to the device
        self.model.to(self.device)
        
        # Verify model device
        model_device = next(self.model.parameters()).device
        console.print(f"[example][→][/example] Model is on device: [info]`{model_device}`[/info]")

        # Add batch dimension and move to device
        self.image = self.image.unsqueeze(0).to(self.device)
        console.print(f"[example][→][/example] Image is on device: [info]`{self.image.device}`[/info]")

        # Switch off dropouts
        self.model.eval()

        # Freeze weights update 
        with torch.no_grad():

            # Model logits
            logps = self.model(self.image)

            # Calculate probabilities
            ps = torch.exp(logps)

            # Grab top k probabilities
            top_p, top_class = ps.topk(self.topk, dim=1)
          
            # Convert and format         
            probs = top_p[0].cpu().numpy()

            # Map model predicted indices to class labels (folder names)
            classes = [self.model.idx_to_class[c] for c in top_class[0].cpu().numpy()]

            # Switch on dropouts
            self.model.train()       
            return probs, classes


    def map_cat_to_name(self):
       # Load category names
        cat_to_name_path = Path(__file__).parent.parent / 'cat_to_name.json'
        with open(cat_to_name_path, 'r') as f:
            self.cat_to_name = json.load(f)


    def print_predictions(self, probs, classes):
        print()
        console.print(f"[purple][↓][/purple] [purple]Predictions Result: [/purple]")
        console.print(f"[example][→][/example] [info]Top Prediction[/info] is" 
                      f"[desc]`{self.cat_to_name[classes[0]]}`[/desc] with probability [arg]{probs[0]:.4f}[/arg]\n")

        console.print(f"[purple][↓][/purple] [purple]Probability Distribution: [/purple]")
        for p, c in zip(probs, classes):
            console.print(
                f"[example][→][/example] [info]Flower:[/info] [desc]{self.cat_to_name[c]:25}[/desc] | " 
                f"[info]Category:[/info] [desc]{c:3}[/desc] | " 
                f"[info]Probability:[/info] [arg]{p:.4f}[/arg]")
        
        # Display the image in terminal
        console.print("\n[purple][↓][/purple] [purple]Input Image:[/purple]")
        from urllib.parse import quote
        abs_path = Path(self.image_path).absolute()
        file_url = f"file://{quote(str(abs_path))}"
        console.print(f"[blue underline][link={file_url}]{self.image_path}[/link][/blue underline]\n")


    def predict_again(self):
        answer = questionary.confirm(
            f"Would you like to make another prediction?",
            style=questionary_default_style()
            ).ask()

        # If user wants to predict again, run the questionary
        if answer:
            self.predict_questionary()
            return
            
        sys.exit("Welcome back...")
        