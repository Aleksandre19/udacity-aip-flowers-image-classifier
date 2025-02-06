import os
import sys
import torch
import torch.hub
from datetime import datetime
from constants import MODEL_TRAIN_MESSAGE, CURRENT_MODEL_ARCHITECTURE_MESSAGE, START_MODEL_TRAIN_MESSAGE, RETRAIN_MODEL_MESSAGE
from torch import nn
from torchvision import models
from utils import console, questionary_default_style, CustomClassifier, print_model_classifier, get_model, define_device
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import questionary
from pipeline import WelcomeMessage


class TrainModel:
    def __init__(self, args, processed_data, retrain=False):
        self.args = args
        self.retrain = retrain
        self.processed_data = processed_data
        self.lr = float(self.args.learning_rate)
        self.pre_train_message
        self.device = define_device()
        self.model = get_model(self.args.arch)
        self.optimizer = None
        self.criterion = None
        self.steps = 0
        self.running_loss = 0
        self.train_losses = []
        self.train_steps = []
        self.valid_losses = []
        self.valid_steps = []
        self.progress = None
        self.train_task = None

    @staticmethod
    def start(args, processed_data, retrain=False):
        train = TrainModel(args, processed_data, retrain)
        print_model_classifier(train.model.classifier, CURRENT_MODEL_ARCHITECTURE_MESSAGE)
        custom_classifier = CustomClassifier(args)
        train.replace_classifier(custom_classifier)
        train.initialize_optimizer_and_criterion()
        train.train_model()

    @property
    def pre_train_message(self):

        if not self.retrain:
            console.print(Panel.fit(
                MODEL_TRAIN_MESSAGE,
                title="Model Training",
                border_style="title"
            ))
        else:
            console.print(Panel.fit(
                RETRAIN_MODEL_MESSAGE,
                title="Model Retraining",
                border_style="title"
            ))

        input("Press Enter to start...\n")

        console.print(f"[example][→][/example] Starting Model Training...")

    def replace_classifier(self, new_classifier, message=""):
        """
        Replace the current model classifier with the new one.
        """
        self.model.classifier = new_classifier

        console.print(f"[example][✓][/example] The model classifier was successfully replaced")
        
        message = "Next you will set up the Criterion, Optimizer and Hyperparameters\n\nArchitecture of new classifier.\n"
        print_model_classifier(self.model.classifier, message)


    def initialize_optimizer_and_criterion(self):
        """
        Initialize the parameters of the new classifier.
        """
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.lr)
        console.print(f"[example][✓][/example] Optimizer and Criterion were successfully initialized")
        self.model.to(self.device)
        console.print(f"[example][✓][/example] Model was moved to the device")


    def train_model(self):
        """
        Start the training process.
        """
        console.print(f"[example][→][/example] Starting Model Training Loop...\n")
        console.print(Panel.fit(
            START_MODEL_TRAIN_MESSAGE,
            title="Model Training Loop",
            border_style="title"
        ))

        input("Press Enter to start...\n")

        self.train_loop()


    def train_loop(self):
        """
        Implement the training loop.
        """
        # Calculate total number of batches for progress tracking
        total_batches = len(self.processed_data.dataloaders['train'])
        total_steps = int(total_batches * self.args.epochs)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            transient=False
        ) as progress:
            # Store progress and task for validation
            self.progress = progress
            self.train_task = progress.add_task(
                f"[yellow]Training model...", 
                total=total_steps
            )
            
            for epoch in range(self.args.epochs):
                for images, labels in self.processed_data.dataloaders['train']:
              # Increment step
                  self.steps += 1
                  
                  # Move images and labels to GPU
                  images, labels = images.to(self.device), labels.to(self.device)

                  # Clear gradients
                  self.optimizer.zero_grad()

                  # Grab logits
                  logps = self.model(images)

                  # Calculate loss
                  loss = self.criterion(logps, labels)

                  # Back propagate
                  loss.backward()

                  # Update weights
                  self.optimizer.step()

                  # Update loss counter
                  self.running_loss += loss.item()

                  self.train_losses.append(loss.item())
                  self.train_steps.append(self.steps)

                  self.training_validation(epoch)

                  # Update progress
                  self.progress.update(self.train_task, advance=1)

        self.model_evaluation()
    
    def training_validation(self, epoch):
        if self.steps % self.args.valid_interval == 0:
            # Create validation task
            self.progress.update(self.train_task, description="[cyan]Running validation...[/cyan]")
            
            self.model.eval()
            
            val_loss = 0
            total_correct = 0
            total_samples = 0 

            with torch.no_grad():
                for inputs, labels in self.processed_data.dataloaders['valid']:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    logps = self.model.forward(inputs)
                    batch_loss = self.criterion(logps, labels)
                    val_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)

                    # This approach calculates exact average since
                    # it is not deppended if the examples are or not perfectly
                    # divisible by batch size
                    total_correct += equals.sum().item()
                    total_samples += equals.shape[0]
                    
                    # # Update validation progress
                    # self.progress.update(self.valid_task, advance=1)

            # Calculate average losses and accuracy
            avg_train_loss = self.running_loss/self.args.valid_interval
            avg_val_loss = val_loss/len(self.processed_data.dataloaders['valid'])
            avg_accuracy = total_correct / total_samples

            # Track validation loss
            self.valid_losses.append(avg_val_loss)
            self.valid_steps.append(self.steps)

            # Reset progress description
            self.progress.update(self.train_task, description=f"[yellow]Training model {epoch+1}/{self.args.epochs}")
        
            # Print metrics
            console.print(
                f"\n[bold]Epoch[/bold] [yellow]{epoch+1}[/yellow]/[yellow]{self.args.epochs}[/yellow] | "
                f"[bold]Step[/bold] [blue]{self.steps}[/blue] | "
                f"[bold]Train loss[/bold] [red]{avg_train_loss:.3f}[/red] | "
                f"[bold]Val loss[/bold] [green]{avg_val_loss:.3f}[/green] | "
                f"[bold]Accuracy[/bold] [magenta]{avg_accuracy:.3f}[/magenta]"
            )
            
            # Reset running loss and return to training mode
            self.running_loss = 0
            self.model.train()


    def model_evaluation (self):
        console.print(f"[example][✓][/example] Model training was successfully completed")

        # Switch off dropouts 
        self.model.eval()   
        
        # Calculate total test batches
        total_test_batches = len(self.processed_data.dataloaders['test'])

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            transient=False
        ) as progress:
            # Add task for evaluation tracking
            eval_task = progress.add_task(
                f"[yellow]Evaluating model...", 
                total=total_test_batches
            )

            with torch.no_grad():
                # accuracy = 0
                test_loss = 0
                total_correct = 0
                total_samples = 0
                
                for images,labels in self.processed_data.dataloaders['test']:
                    # Move to device
                    images,labels = images.to(self.device), labels.to(self.device)

                    # Model logits
                    logps = self.model(images)

                    # Negative Log Likelihood Loss
                    loss = self.criterion(logps, labels)

                    # Accumulate loss
                    test_loss += loss.item()
                    
                    # Calculate probabilities
                    ps = torch.exp(logps)

                    # Grab top k probabilities
                    top_p, top_class = ps.topk(1, dim=1)

                    # Calculate predictions and labels matching
                    equals = top_class == labels.view(*top_class.shape)

                    # Accumulate total correct samples
                    total_correct += equals.sum().item()

                    # Track total samples for average calculation
                    total_samples += equals.shape[0]

                    # Update progress
                    progress.update(eval_task, advance=1)

            # Switch on dropouts
            self.model.train()

        
        accuracy = total_correct / total_samples
        confidence = 1 - (test_loss / len(self.processed_data.dataloaders['test']))
        self.training_complete(accuracy, confidence)

    def training_complete(self, accuracy, confidence):
        print("")
        console.print(Panel.fit(
            f"[info]Congratulations! Training completed successfully![/info]\n"
            f"\n[example]Final Evaluation Results:[/example]\n"
            f"Model Accuracy: [green]{accuracy:.3f}[/green] | "
            f"Model Confidence: [blue]{confidence:.3f}[/blue]\n\n"
            f"[desc]Please choose one of the following options\n"
            f"from the menu below to proceed further[/desc]\n",
            title="Training Completed",
            border_style="green"
        ))

        choice = questionary.select(
            "Choose an option",
            choices=[
                "Save model",
                "Retrain model",
                "Exit"
            ],
            style=questionary_default_style()
        ).ask()

        if choice == "Save model":
            while True:
                model_name = questionary.text(
                    "Give a name to the saved model (without .pth):",
                    default=f"checkpoint_{self.args.arch}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    style=questionary_default_style()
                    ).ask()

                # If folder doesn't exists, create it
                if not os.path.exists(self.args.save_dir):
                    os.makedirs(self.args.save_dir)
                
                # Check if file already exists
                file_path = os.path.join(self.args.save_dir, f"{model_name}.pth")

                if os.path.exists(file_path):
                    overwrite = questionary.confirm(
                        f"Model '{model_name}' already exists. Do you want to overwrite it?",
                        default=False
                    ).ask()
                    
                    if overwrite:
                        break  # User wants to overwrite, proceed with saving
                    else:
                        # User doesn't want to overwrite, restart loop to ask for a new name
                        # This will keep asking until they either:
                        # 1. Enter a name that doesn't exist
                        # 2. Choose to overwrite an existing file
                        continue
                else:
                    break  # File doesn't exist, proceed with saving

            self.save_model(model_name, file_path)
        elif choice == "Retrain model":
            self.retrain_model()
        elif choice == "Exit":
            sys.exit("Exiting...")

    def save_model(self, model_name, checkpoint_path):
        """
        This function saves a checkpoint
        """

        # Define structure of the checkpoint
        checkpoint = {
            'input': self.args.input_size,
            'output': self.args.output_size,
            'hidden_layers': self.args.hidden_units,
            'learning_rate': self.lr,
            'dropout': self.args.drop_p,
            'epochs': self.args.epochs,
            'batch_size': self.processed_data.dataloaders['train'].batch_size,
            'activation': 'LogSoftmax',
            'criterion': self.criterion,
            'optimizer': self.optimizer,
            'arch': self.args.arch,
            'class_to_idx': self.processed_data.datasets['train'].class_to_idx,
            'state_dict': self.model.state_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)
        console.print(f"[example][✓][/example] The Model '{model_name}'  was successfully saved to '{checkpoint_path}'")

    def retrain_model(self):
        reconfigure = WelcomeMessage(retrain=True)
        TrainModel.start(reconfigure.args, self.processed_data, retrain=True)
    

