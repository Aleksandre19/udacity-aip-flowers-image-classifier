import os
import sys
from utils import questionary_default_style, get_predict_terminal_args, console
from tkinter import Tk, filedialog
from pathlib import Path
from rich.panel import Panel
from constants import CHOOSE_MODEL_ERROR_MESSAGE

class SetupModel:
  def __init__(self):
    self.args = get_predict_terminal_args()
    self.setup_questionary()

  def setup_questionary(self):
    style = questionary_default_style()
    
    if not self.args.model:
      # Choose model file
      console.print(f"[example][→] Please select the model file from the file dialog...[/example]\n")
      root = Tk()
      root.withdraw()  # Hide the main window
      model = filedialog.askopenfilename(
          title="Choose the model to make predictions with",
          filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")]
      )
      root.destroy()
      
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
    
    console.print(f"[example][✓][/example] The model file '{self.args.model}' was successfully loaded")