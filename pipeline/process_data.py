import os
import sys
from pathlib import Path
import subprocess

from rich.panel import Panel
from constants import DATASET_URL, PROVIDE_DATA_RICH_MESSAGE
from utils import log, console, download_dataset, start_data_process_questionary
import questionary

class ProcessData:
    """
    Class to process data before training.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    @staticmethod
    def start(data_dir):
        process_data = ProcessData(data_dir)
        process_data.handle_data_directory
        log.info(f"Data directory: {data_dir}")

    @property
    def handle_data_directory(self):
        """Handle data directory creation if it doesn't exist."""
        if not os.listdir(self.data_dir):
            while True:
                log.warning(f"Data directory '{self.data_dir}' is empty." 
                            f"Please ensure there are images in the directory.")
                
                choice = start_data_process_questionary()

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
                        data_path = Path(self.data_dir)
                        data_path.mkdir(parents=True, exist_ok=True)
                        
                        url = DATASET_URL
                        tar_file = data_path / "flower_data.tar.gz"

                        download_dataset(url, tar_file, data_path)
                        
                        return
                    except subprocess.CalledProcessError as e:
                        log.error(f"Failed to download or extract dataset: {str(e)}")
                        sys.exit(1)
                    except Exception as e:
                        log.error(f"An error occurred: {str(e)}")
                        sys.exit(1)