import os
import sys
from pathlib import Path
import subprocess

from rich.panel import Panel
from utils import log, console, download_dataset
from utils import PROVIDE_DATA_RICH_MESSAGE
import questionary

class ProcessData:
    """
    Class to process data before training.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    @staticmethod
    def start(data_dir):
        ProcessData(data_dir)

        if not os.listdir(data_dir):
            while True:
                log.warning(f"Data directory '{data_dir}' is empty. Please ensure there are images in the directory.")
                
                choice = questionary.select(
                    "How would you like to set up your dataset?",
                    choices=[
                        "Download sample dataset (recommended)",
                        "I'll provide my own dataset",
                        "Exit"
                    ],
                    style=questionary.Style([
                        ('qmark', 'fg:yellow bold'),
                        ('question', 'bold'),
                        ('answer', 'fg:green bold'),
                        ('pointer', 'fg:yellow bold'),
                        ('highlighted', 'fg:yellow'),
                        ('selected', 'fg:green'),
                    ])
                ).ask()

                if choice == "Exit":
                    sys.exit("Exiting...")
                    
                elif choice == "I'll provide my own dataset":
                    # Show dataset structure information
                    console.print(Panel.fit(
                        PROVIDE_DATA_RICH_MESSAGE,
                        title="Dataset Organization Guide",
                        border_style="title")
                    )
                    
                    input()
                    sys.exit("Exiting... Please run the script again once your dataset is ready.")
                    
                elif choice == "Download sample dataset (recommended)":
                    try:
                        # Create data directory if it doesn't exist
                        data_path = Path(data_dir)
                        data_path.mkdir(parents=True, exist_ok=True)
                        
                        url = "https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"
                        tar_file = data_path / "flower_data.tar.gz"

                        download_dataset(url, tar_file, data_path)
                        
                        return
                    except subprocess.CalledProcessError as e:
                        log.error(f"Failed to download or extract dataset: {str(e)}")
                        sys.exit(1)
                    except Exception as e:
                        log.error(f"An error occurred: {str(e)}")
                        sys.exit(1)

        log.info(f"Data directory: {data_dir}")