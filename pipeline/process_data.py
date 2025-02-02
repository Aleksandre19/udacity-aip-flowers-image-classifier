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
        process_data._handle_data_directory
        process_data._check_dataset_structure
        log.info(f"Data directory: {data_dir}")

    @property
    def _handle_data_directory(self):
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
                    self._dataset_organization_guide_message
                    
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


    @property
    def _check_dataset_structure(self):
        """Check if the data directory has the expected structure including category folders."""
        expected_structure = {"train", "valid", "test"}
        data_path = Path(self.data_dir)
        
        # Check top-level directories
        if not set(os.listdir(self.data_dir)) == expected_structure:
            log.warning(f"Data directory '{self.data_dir}' missing required directories (train/valid/test).")
            self._dataset_organization_guide_message
            return False
        
        # Check each subdirectory for category folders and image files
        for subdir in expected_structure:
            subdir_path = data_path / subdir
            category_dirs = [d for d in subdir_path.iterdir() if d.is_dir()]
            
            # Check if there are any category directories
            if not category_dirs:
                log.warning(f"No category directories found in {subdir}/.")
                self._dataset_organization_guide_message
                return False
            
            # Check each category directory
            for category_dir in category_dirs:
                try:
                    # Verify category name is a number
                    int(category_dir.name)
                    
                    # Get all files in the category directory
                    all_files = list(category_dir.iterdir())
                    if not all_files:
                        log.warning(f"No files found in {subdir}/{category_dir.name}/")
                        self._dataset_organization_guide_message
                        return False
                    
                    # Check if all files are .jpg
                    non_jpg_files = [f.name for f in all_files if not f.name.lower().endswith('.jpg')]
                    if non_jpg_files:
                        log.warning(f"Non-jpg files found in {subdir}/{category_dir.name}/: {', '.join(non_jpg_files)}")
                        self._dataset_organization_guide_message
                        return False
                    
                    # Check if there are any jpg files
                    if not all_files:
                        log.warning(f"No image files found in {subdir}/{category_dir.name}/")
                        self._dataset_organization_guide_message
                        return False
                        
                except ValueError:
                    log.warning(f"Invalid category directory name: {category_dir.name} (must be a number)")
                    self._dataset_organization_guide_message
                    return False
        
        log.info(f"Data directory '{self.data_dir}' has the correct structure with valid category folders and images.")
        return True

    @property
    def _dataset_organization_guide_message(self):
        """Validate the dataset structure"""
        console.print(Panel.fit(
            PROVIDE_DATA_RICH_MESSAGE,
            title="Dataset Organization Guide",
            border_style="title")
        )
        
        input()
        sys.exit("Exiting... Please run the script again once your dataset is ready.")