import sys

import json
import questionary
import torch
from PIL import Image
from pathlib import Path


from constants import CHOOSE_IMAGE_ERROR_MESSAGE
from rich.panel import Panel
from torchvision import transforms
from utils import (
    console, 
    define_device, 
    questionary_default_style, 
    select_file
)

class MakePrediction:
    """A class for making flower predictions using a trained deep learning model.
    
    This class handles the complete prediction pipeline including image selection,
    transformation, and model inference.
    """
    
    def __init__(self, model):
        """Initialize the MakePrediction class.
        
        Args:
            model: The trained PyTorch model for making predictions
        """
        self.model = model  # Store the trained model
        self.image = None  # Will store the transformed image tensor
        self.image_path = None  # Path to the input image
        self.device = None  # CPU/GPU device for computation
        self.topk = 5  # Number of top predictions to return
        self.cat_to_name = None  # Dictionary mapping categories to flower names
        
    
    @staticmethod
    def start(model):
        """Static method to create and start the prediction process.
        
        Args:
            model: The trained model to use for predictions
        """
        prediction = MakePrediction(model)
        prediction.predict_questionary()

    def predict_questionary(self):
        """Main method that orchestrates the prediction workflow.
        
        Handles the complete prediction pipeline including image selection,
        transformation, model inference, and result display.
        """
        self.choose_image()
        if not self.cat_to_name:
            self.select_cat_to_name()
        self.transform_image()
        self.preidct_questionary()
        probs, classes = self.predict()
        self.print_predictions(probs, classes)
        self.predict_again()
    
    def choose_image(self):
        """Prompt user to select an image file for prediction.
        
        Opens a file dialog for image selection and validates the choice.
        Exits if no image is selected.
        """
        console.print(f"[example][→] Please select the image[/example]")
        image = select_file(
            title="Choose the image to make predictions with",
            filetypes=[
                ("Image files", "*.jpg *.jpeg"),
            ]
        )
        
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

    def select_cat_to_name(self):
        """Prompt user to select a JSON file containing category to name mappings.
        
        Opens a file dialog for JSON selection and loads the mapping.
        Exits if no file is selected.
        """
        console.print(f"[example][→] Please select the category-to-name mapping file from the file dialog (*.json)[/example]")
        cat_to_name = select_file(
            title="Choose the category mapping to names (*.json):",
            filetypes=[("JSON files", "*.json"),]
        )

        if not cat_to_name:
            print("no cat_to_name")
            sys.exit("Exiting...")
            
        self.cat_to_name = json.load(open(cat_to_name))

        console.print(f"[example][✓][/example] Category-to-name mapping was selected successfully")

    def transform_image(self):
        """Transform the input image for model inference.
        
        Applies a series of transformations including resize, crop, and normalization
        to prepare the image for the model.
        """
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
        """Prompt user for prediction parameters.
        
        Gets user input for:
        - Number of top predictions to display (topk)
        - Whether to use GPU acceleration
        """
        topk   = questionary.text(
            f"Top K: Number of top predictions to display (current: {self.topk}):",
            validate=lambda x: x.isdigit() if x else False,
            default=f"{self.topk}",
            style=questionary_default_style()
        ).ask()
        self.topk = int(topk)

        console.print(f"[example][✓][/example] Top K is selected:[arg]'{self.topk}'[/arg]")
        
          # If user cancels the dialog
        # GPU option
        gpu = questionary.confirm(
            f"GPU: Use GPU acceleration (current: {False if not self.device else True}):",
            default=False,
            style=questionary_default_style()
        ).ask()

        self.device = define_device(gpu)


    def predict(self):
        """Perform model inference on the transformed image.
        
        Returns:
            tuple: (probabilities, class_indices) for top k predictions
        """
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
    
    def _select_cat_name(self, class_index):
        """Get the flower name for a given class index.
        
        Args:
            class_index: The class index to look up
            
        Returns:
            str or None: The flower name if found, None otherwise
        """
        cat_name = self.cat_to_name.get(class_index, None)
        if not cat_name:
            cat_name = None
        return cat_name

    def print_predictions(self, probs, classes):
        """Display prediction results in a formatted way.
        
        Args:
            probs: List of prediction probabilities
            classes: List of predicted class indices
        """
        print()
        cat_name = self._select_cat_name(classes[0])
            
        console.print(f"[purple][↓][/purple] [purple]Predictions Result: [/purple]")

        if cat_name:
            console.print(f"[example][→][/example] [info]Top Prediction[/info] is" 
                          f"[desc]`{cat_name}`[/desc] with probability [arg]{probs[0]:.4f}[/arg]\n")
        else:
            console.print(f"[error][❌] No name found for the predicted class {classes[0]}."
                          f" Please check the category-to-name mapping file. [/error]\n")
                          
        console.print(f"[purple][↓][/purple] [purple]Probability Distribution: [/purple]")

        print_message = ""
        for p, c in zip(probs, classes):
            cat_name = self._select_cat_name(c)
            if not cat_name:
                print_message = f"[error][❌] No name found for the predicted class {c}. Please check the category-to-name mapping file. [/error]"
            else:
                print_message = (f"[example][→][/example] [info]Flower:[/info] [desc]{cat_name:25}[/desc] | "
                                f"[info]Category:[/info] [desc]{c:3}[/desc] | "
                                f"[info]Probability:[/info] [arg]{p:.4f}[/arg]")
                
            console.print(f"{print_message}")
        
        # Display the image in terminal
        console.print("\n[purple][↓][/purple] [purple]Input Image:[/purple]")
        from urllib.parse import quote
        abs_path = Path(self.image_path).absolute()
        file_url = f"file://{quote(str(abs_path))}"
        console.print(f"[blue underline][link={file_url}]{self.image_path}[/link][/blue underline]\n")


    def predict_again(self):
        """Prompt user if they want to make another prediction.
        
        Returns:
            bool: True if user wants to predict again, False otherwise
        """
        answer = questionary.confirm(
            f"Would you like to make another prediction?",
            style=questionary_default_style()
            ).ask()

        # If user wants to predict again, run the questionary
        if answer:
            self.predict_questionary()
            return
            
        sys.exit("Welcome back...")
        