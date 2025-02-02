import os
import sys
from pathlib import Path
import subprocess
from time import sleep

from rich.panel import Panel
from constants import DATASET_URL, PROVIDE_DATA_RICH_MESSAGE
from utils import log, console, download_dataset, start_data_process_questionary
import questionary
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

        # Step 1: Check main directories
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
            structure_task = progress.add_task(
                "[yellow]Step 1/4: Checking main directories (train/valid/test)...",
                total=len(expected_structure)
            )
            
            for dir_name in expected_structure:
                progress.advance(structure_task)
                if not (data_path / dir_name).is_dir():
                    log.error(f"Missing required directory: {dir_name}")
                    live.stop()
                    return self._dataset_organization_guide_message
            progress.update(structure_task, description="[green]✓[/green] Main directories check passed")

        # Step 2: Check for category directories
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
            # Count total directories to check
            total_dirs = len(expected_structure)
            dir_task = progress.add_task(
                "[yellow]Step 2/4: Checking for category directories...",
                total=total_dirs
            )

            for subdir in expected_structure:
                progress.advance(dir_task)
                subdir_path = data_path / subdir
                category_dirs = [d for d in subdir_path.iterdir() if d.is_dir()]
                
                if not category_dirs:
                    log.error(f"No category directories found in {subdir}/")
                    live.stop()
                    return self._dataset_organization_guide_message
            progress.update(dir_task, description="[green]✓[/green] Category directories check passed")

        # Step 3: Validate category names
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
            total_categories = sum(
                len([d for d in (data_path / subdir).iterdir() if d.is_dir()])
                for subdir in expected_structure
            )
            
            name_task = progress.add_task(
                "[yellow]Step 3/4: Validating category names...",
                total=total_categories
            )

            for subdir in expected_structure:
                subdir_path = data_path / subdir
                category_dirs = [d for d in subdir_path.iterdir() if d.is_dir()]
                
                for category_dir in category_dirs:
                    progress.advance(name_task)
                    try:
                        int(category_dir.name)
                    except ValueError:
                        log.error(f"Invalid category directory name in {subdir_path}/: {category_dir.name} (must be a number)")
                        live.stop()
                        return self._dataset_organization_guide_message
            progress.update(name_task, description="[green]✓[/green] Category names validation passed")

        # Step 4: Check image files
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
            # Calculate total files across all categories
            total_files = 0
            for subdir in expected_structure:
                subdir_path = data_path / subdir
                for cat_dir in [d for d in subdir_path.iterdir() if d.is_dir()]:
                    total_files += len([f for f in cat_dir.iterdir() if f.is_file()])
            
            files_task = progress.add_task(
                "[yellow]Step 4/4: Validating image files...",
                total=total_files
            )

            for subdir in expected_structure:
                subdir_path = data_path / subdir
                for category_dir in [d for d in subdir_path.iterdir() if d.is_dir()]:
                    all_files = list(category_dir.iterdir())
                    
                    if not all_files:
                        log.error(f"No files found in {subdir}/{category_dir.name}/")
                        progress.update(files_task, completed=total_files)
                        live.stop()
                        return self._dataset_organization_guide_message
                    
                    # Advance progress for each file we're about to check
                    for f in all_files:
                        progress.advance(files_task)
                        if not f.name.lower().endswith('.jpg'):
                            log.error(f"Non-jpg file found: {subdir}/{category_dir.name}/{f.name}")
                            live.stop()
                            return self._dataset_organization_guide_message
            progress.update(files_task, description="[green]✓[/green] Image files validation passed")

        console.print("\n[green]✓[/green] Dataset structure validation completed successfully!")
        return True

    @property
    def _dataset_organization_guide_message(self):
        """Show dataset organization guide and exit"""
        # Clear any active progress displays
        console.clear_live()
        # Show the guide message
        console.print(Panel.fit(
            PROVIDE_DATA_RICH_MESSAGE,
            title="Dataset Organization Guide",
            border_style="title")
        )
        
        input("\nPress Enter to exit...")
        sys.exit("Exiting...")