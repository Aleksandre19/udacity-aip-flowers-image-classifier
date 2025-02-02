import os
import sys
from pathlib import Path
import subprocess
import tarfile
import urllib.request

from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from utils import log, console
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
                "[bold yellow]Dataset Structure Requirements[/bold yellow]\n\n"
                "Your dataset should be organized as follows:\n"
                "[blue]data_directory/[/blue]\n"
                "├── [green]train/[/green]\n"
                "│   ├── [yellow]1/[/yellow] (category number)\n"
                "│   │   └── image1.jpg, image2.jpg, ...\n"
                "│   ├── [yellow]2/[/yellow]\n"
                "│   └── ...\n"
                "├── [green]valid/[/green]\n"
                "│   ├── [yellow]1/[/yellow]\n"
                "│   └── ...\n"
                "└── [green]test/[/green]\n"
                "    ├── [yellow]1/[/yellow]\n"
                "    └── ...\n\n"
                "[bold]Important Notes:[/bold]\n"
                "• Category numbers should match cat_to_name.json located in the root directory\n"
                "• Each category folder should contain only image files\n"
                "• Supported format: .jpg\n\n"
                "[red]Press Enter to exit and organize your dataset...[/red]",
                title="Dataset Organization Guide",
                border_style="bold white"
            ))
            input()
            sys.exit("Exiting... Please run the script again once your dataset is ready.")
            
        elif choice == "Download sample dataset (recommended)":
            try:
               

                # Create data directory if it doesn't exist
                data_path = Path(data_dir)
                data_path.mkdir(parents=True, exist_ok=True)
                
                url = "https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"
                tar_file = data_path / "flower_data.tar.gz"

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                ) as progress:
                    # Download with progress tracking
                    download_task = progress.add_task("[yellow]Downloading dataset...", total=None)
                    response = urllib.request.urlopen(url)
                    total_size = int(response.headers.get('content-length', 0))
                    progress.update(download_task, total=total_size)
                    
                    with open(tar_file, 'wb') as f:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            progress.update(download_task, advance=len(chunk))
                    
                    # Extract with progress tracking
                    extract_task = progress.add_task("[yellow]Extracting dataset...", total=None)
                    with tarfile.open(tar_file, 'r:gz') as tar:
                        members = tar.getmembers()
                        progress.update(extract_task, total=len(members))
                        for member in members:
                            tar.extract(member, path=data_path)
                            progress.advance(extract_task)
                    
                    # Clean up
                    tar_file.unlink()
                
                console.print("[green]✓[/green] Dataset downloaded and extracted successfully!")
                return
                
            except subprocess.CalledProcessError as e:
                log.error(f"Failed to download or extract dataset: {str(e)}")
                sys.exit(1)
            except Exception as e:
                log.error(f"An error occurred: {str(e)}")
                sys.exit(1)

    log.info(f"Data directory: {data_dir}")