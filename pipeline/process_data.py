import os
import sys
from pathlib import Path
import subprocess
from time import sleep

from rich.panel import Panel
from constants import DATA_STRUCTURE_MESSAGE, DATASET_URL, PROVIDE_DATA_RICH_MESSAGE
from utils import log, console, download_dataset, start_data_process_questionary

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.live import Live

class ProcessDataStructure:
    """
    Class to process data before training.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    @staticmethod
    def start(data_dir):
        console.print(f"[example][→][/example] Starting dataset validation for [arg]'{data_dir}'[/arg]")
        process_data = ProcessDataStructure(data_dir)
        process_data._create_data_directory
        process_data._handle_data_directory
        process_data._check_dataset_structure

    
    @property
    def _create_data_directory(self):
        """Handle data directory creation if it doesn't exist."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            console.print(f"[example][✓][/example] Created data directory:[arg]'{self.data_dir}'[/arg]")
        else:
            console.print(f"[example][✓][/example] Using existing data directory:[arg]'{self.data_dir}'[/arg]")

    @property
    def _handle_data_directory(self):
        """Handle data directory creation if it doesn't exist."""

        # If directory exists and is not empty, return
        if not os.path.exists(self.data_dir):
            console.print(f"[error][→][/error] Data directory must be provided to continue.\n"
                          f"Please start the training again and provide a valid data directory. \n\n")
            console.print("Press Enter to exit...")
            sys.exit("Exiting...")
       
        while True:
            console.print(f"[example][→][/example] Data directory [arg]'{self.data_dir}'[/arg] is empty.\n")
            
            choice = start_data_process_questionary()

            if choice == "Exit":
                sys.exit("Exiting...")
                
            elif choice == "I have it":
                # Show dataset structure information
                console.print("[example][→][/example] Validating dataset...")
                break
                
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
        # Step 1: Check main directories exist
        def check_main_dir(path, dir_name):
            return (path / dir_name).is_dir(), dir_name
        
        if not self._validation_steps(
            start_message="[yellow]Step 1/4: Checking main directories (train/valid/test)...[/yellow]",
            error_message="Missing required directory",
            end_message="[green]✓[/green] Main directories check passed",
            validation_func=check_main_dir
        ):
            return False

        # Step 2: Check category directories exist
        def check_category_dirs(path, dir_name):
            subdir_path = path / dir_name
            category_dirs = [d for d in subdir_path.iterdir() if d.is_dir()]
            return bool(category_dirs), f"No category directories found in {dir_name}/"

        if not self._validation_steps(
            start_message="[yellow]Step 2/4: Checking for category directories...[/yellow]",
            error_message="Invalid directory structure",
            end_message="[green]✓[/green] Category directories check passed",
            validation_func=check_category_dirs
        ):
            return False

        # Step 3: Validate category names
        def check_category_names(path, dir_name):
            subdir_path = path / dir_name
            for category_dir in [d for d in subdir_path.iterdir() if d.is_dir()]:
                try:
                    int(category_dir.name)
                except ValueError:
                    return False, f"Invalid category name in {dir_name}/: {category_dir.name} (must be a number)"
            return True, None

        if not self._validation_steps(
            start_message="[yellow]Step 3/4: Validating category names...[/yellow]",
            error_message="Invalid category",
            end_message="[green]✓[/green] Category names validation passed",
            validation_func=check_category_names
        ):
            return False

        # Step 4: Check image files
        def check_image_files(path, dir_name):
            subdir_path = path / dir_name
            for category_dir in [d for d in subdir_path.iterdir() if d.is_dir()]:
                files = list(category_dir.iterdir())
                if not files:
                    return False, f"No files found in {dir_name}/{category_dir.name}/"
                for f in files:
                    if not f.name.lower().endswith('.jpg'):
                        return False, f"Non-jpg file found: {dir_name}/{category_dir.name}/{f.name}"
            return True, None

        if not self._validation_steps(
            start_message="[yellow]Step 4/4: Validating image files...[/yellow]",
            error_message="Invalid files",
            end_message="[green]✓[/green] Image files validation passed",
            validation_func=check_image_files
        ):
            return False

        console.print("[example][✓][/example] Dataset validation completed successfully!")
        return True

    def _validation_steps(self, start_message, error_message, end_message, validation_func):
        """Generic validation step handler with progress bar.
        
        Args:
            start_message: Initial progress bar message
            error_message: Message template for errors
            end_message: Success message after completion
            validation_func: Function that takes (path, name) and returns (is_valid, error_details)
        """
        expected_structure = {"train", "valid", "test"}
        data_path = Path(self.data_dir)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            transient=False
        )
        with Live(progress, refresh_per_second=10) as live:
            task = progress.add_task(
                start_message,
                total=len(expected_structure)
            )
            
            for dir_name in expected_structure:
                progress.advance(task)
                is_valid, error_details = validation_func(data_path, dir_name)
                if not is_valid:
                    log.error(f"{error_message}: {error_details}")
                    live.stop()
                    return self._dataset_organization_guide_message
            progress.update(task, description=end_message)
            return True

    @property
    def _dataset_organization_guide_message(self):
        """Show dataset organization guide and exit"""
        # Clear any active progress displays
        console.clear_live()
        # Show the guide message
        console.print(Panel.fit(
            DATA_STRUCTURE_MESSAGE,
            title="[bold]Dataset Organization Guide[/bold]",
            border_style="blue")
        )
        
        input("\nPress Enter to exit...")
        sys.exit("Exiting...")