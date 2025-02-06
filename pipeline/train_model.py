import sys
import torch
import torch.hub
from constants import MODEL_TRAIN_MESSAGE, CURRENT_MODEL_ARCHITECTURE_MESSAGE, START_MODEL_TRAIN_MESSAGE
from torch import nn
from torchvision import models
from utils import console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import questionary


class TrainModel:
    def __init__(self, args, processed_data):
        self.args = args
        self.processed_data = processed_data
        self.lr = float(self.args.learning_rate)
        self.pre_train_message
        self.device = self.define_device()
        self.model = self.get_model(self.args.arch)
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
    def start(args, processed_data):
        train = TrainModel(args, processed_data)
        train.print_model_classifier(CURRENT_MODEL_ARCHITECTURE_MESSAGE)
        custom_classifier = CustomClassifier(args)
        train.replace_classifier(custom_classifier)
        train.initialize_optimizer_and_criterion()
        train.train_model()

    @property
    def pre_train_message(self):
        console.print(Panel.fit(
            MODEL_TRAIN_MESSAGE,
            title="Model Training",
            border_style="title"
        ))

        input("Press Enter to start...\n")

        console.print(f"[example][→][/example] Starting Model Training...")

    def define_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[example][✓][/example] Set the device to:[arg]'{device}'[/arg]")
        return device

    def get_model(self, model_name):
        model_mapping = {
            "vgg11": (models.vgg11, models.VGG11_Weights.DEFAULT),
            "vgg13": (models.vgg13, models.VGG13_Weights.DEFAULT),
            "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
            "vgg19": (models.vgg19, models.VGG19_Weights.DEFAULT)
        }
        
        if model_name not in model_mapping:
            console.print(f"[error]Error:[/error] Unsupported model: [arg]{model_name}[/arg]")
            console.print(f"[info]Available models:[/info] [desc]{', '.join(model_mapping.keys())}[/desc]")
            sys.exit(1)
        
        model_func, weights = model_mapping[model_name]
        model = model_func(weights=weights) 

        console.print(f"[example][✓][/example] The model [arg]'{self.args.arch}'[/arg] was successfully loaded")
        return model

    def replace_classifier(self, new_classifier, message=""):
        """
        Replace the current model classifier with the new one.
        """
        self.model.classifier = new_classifier

        console.print(f"[example][✓][/example] The model classifier was successfully replaced")
        
        message = "Next you will set up the Criterion, Optimizer and Hyperparameters\n\nArchitecture of new classifier.\n"
        self.print_model_classifier(message)


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
            # Update progress description to show validation is running
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
            style=questionary.Style([
                ('qmark', 'fg:yellow bold'),
                ('question', 'bold'),
                ('answer', 'fg:green bold'),
                ('pointer', 'fg:yellow bold'),
                ('highlighted', 'fg:yellow'),
                ('selected', 'fg:green'),
            ])
        ).ask()

        if choice == "Save model":
            print("Saving model...")
        elif choice == "Retrain model":
            print("Retraining model...")
        elif choice == "Exit":
            sys.exit("Exiting...")


    def print_model_classifier(self, message=""):
        """
        Print the current model classifier architecture with custom formatting.
        """
        # Display model classifier architecture with custom formatting
        classifier_str = str(self.model.classifier)
        
        # Parse and format the classifier string
        lines = classifier_str.split('\n')
        formatted_lines = []
        
        # Format the Sequential header
        formatted_lines.append("[purple]Sequential([/purple]")
        
        # Process each layer
        for line in lines[1:-1]:  # Skip first and last lines (Sequential( and ))
            if not line.strip():
                continue
                
            # Extract the layer number, type and parameters
            parts = line.strip().split(':', 1)
            if len(parts) == 2:
                layer_num = parts[0].strip('() ')
                layer_info = parts[1].strip()
                
                # Split layer type and parameters
                layer_type = layer_info.split('(', 1)[0]
                layer_params = '(' + layer_info.split('(', 1)[1] if '(' in layer_info else ''
                
                # Format with colors
                formatted_line = f"  [example]({layer_num}):[/example] [green]{layer_type}[/green][info]{layer_params}[/info]"
                formatted_lines.append(formatted_line)
        
        # Add closing bracket
        formatted_lines.append("[purple])[/purple]")
        
        # Join all lines
        formatted_str = message + '\n'.join(formatted_lines)
        
        # Print the formatted string with a title
        console.print(Panel(
            formatted_str, 
            border_style="title"
        ))

        console.print("Press Enter to continue...")
        input("")


class CustomClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.drop_p = float(self.args.drop_p)
        self.input_size = int(self.args.input_size)
        self.output_size = int(self.args.output_size)
        # Initialize network layers
        self.all_layers = nn.ModuleList()
        
        # Define input layer
        self.all_layers.append(nn.Linear(self.input_size, self.args.hidden_units[0])) 
        self.all_layers.append(nn.ReLU())
        self.all_layers.append(nn.Dropout(p=self.drop_p))
        
        # Define hidden layers
        for i in range(len(self.args.hidden_units) - 1):  # Note the -1 here
            self.all_layers.append(nn.Linear(self.args.hidden_units[i], self.args.hidden_units[i + 1]))
            self.all_layers.append(nn.ReLU())
            self.all_layers.append(nn.Dropout(p=self.drop_p))
            
        # Define output layer
        self.all_layers.append(nn.Linear(self.args.hidden_units[-1], self.output_size))
        self.all_layers.append(nn.LogSoftmax(dim=1))
        
    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)
        return x