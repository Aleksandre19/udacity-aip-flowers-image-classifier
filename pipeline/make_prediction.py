import sys

from pathlib import Path
from tkinter import Tk, filedialog

from constants import CHOOSE_IMAGE_ERROR_MESSAGE
from rich.panel import Panel
from utils import console, questionary_default_style

import questionary
from PIL import Image
from torchvision import transforms

class MakePrediction:
    def __init__(self, model):
        self.model = model
        self.image = None
        self.topk = 5
    
    @staticmethod
    def start(model):
        prediction = MakePrediction(model)
        prediction.predict_questionary()

    def predict_questionary(self):
        self.choose_image()
        self.transform_image()
        self.choose_topk()

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
        self.image = image_path.relative_to(Path.cwd())

        console.print(f"[example][✓][/example] Image is selected:[arg]'{self.image}'[/arg]")

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
    
    def choose_topk(self):
        topk   = questionary.text(
            f"Top K: Number of top predictions to display (current: {self.topk}):",
            validate=lambda x: x.isdigit() if x else False,
            default=f"{self.topk}",
            style=questionary_default_style()
        ).ask()
        self.topk = int(topk)

        console.print(f"[example][✓][/example] Top K is selected:[arg]'{self.topk}'[/arg]")