DATASET_URL="https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"

PROVIDE_DATA_RICH_MESSAGE = (
   "[info]Command Line Arguments Information[/info]\n\n"
      "• [arg]data_dir[/arg]: [desc]Directory containing your training images[/desc] (Required)\n"
      "• [arg]--save_dir[/arg]: [desc]Save trained model checkpoints to this directory[/desc] (default: checkpoints)\n"
      "• [arg]--arch[/arg]: [desc]Neural network architecture to use[/desc] (default: vgg16)(avaliables: vgg11, vgg13, vgg19)\n"
      "• [arg]--learning_rate[/arg]: [desc]How fast the model learns during training[/desc] (default: 0.0001)\n"
      "• [arg]--hidden_units[/arg]: [desc]Number of neurons in hidden layers[/desc] (default: [4096, 1024])\n"
      "• [arg]--epochs[/arg]: [desc]Number of complete training cycles[/desc] (default: 17)\n"
      "• [arg]--gpu[/arg]: [desc]Use GPU for faster training if available[/desc] (default: False)\n\n"
    "[info]Example:[/info]\n"
    "[example]python train.py flowers/dataset --arch vgg16 --gpu[/example]\n\n"
    
    "[bold info]Dataset Structure Requirements[/bold info]\n"
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
    "[bold yellow]Important Notes:[/bold yellow]\n"
    "• Category numbers should match cat_to_name.json located in the root directory\n"
    "• Each category folder should contain only image files\n"
    "• Supported format: .jpg\n"
)