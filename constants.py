DATASET_URL="https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"

PROVIDE_DATA_RICH_MESSAGE = (
   "[info]Command Line Arguments Information[/info]\n\n"
      "• [arg]data_dir[/arg]: [desc]Directory containing your training images[/desc] (Required)\n"
      "• [arg]--save_dir[/arg]: [desc]Save trained model checkpoints to this directory[/desc] (default: checkpoints)\n"
      "• [arg]--arch[/arg]: [desc]Neural network architecture to use[/desc] (default: vgg16)(avaliables: vgg11, vgg13, vgg19)\n"
      "• [arg]--learning_rate[/arg]: [desc]How fast the model learns during training[/desc] (default: 0.0001)\n"
      "• [arg]--input_size[/arg]: [desc]Number of input units to the first linear layer[/desc] (default: 25088)\n"
      "• [arg]--hidden_units[/arg]: [desc]Number of neurons in hidden layers[/desc] (default: [4096, 1024])\n"
      "• [arg]--output_size[/arg]: [desc]Number of neurons in output layer[/desc] (default: 102)\n"
      "• [arg]--drop_p[/arg]: [desc]Dropout probability[/desc] (default: 0.2)\n"
      "• [arg]--epochs[/arg]: [desc]Number of complete training cycles[/desc] (default: 17)\n"
      "• [arg]--valid_interval[/arg]: [desc]Steps of validation interval[/desc] (default: 100)\n"
      "• [arg]--gpu[/arg]: [desc]Use GPU for faster training if available[/desc] (default: False)\n\n"
    "[info]Example:[/info]\n"
    "[example]python train.py flowers/dataset --arch vgg16 --gpu[/example]\n\n"
    
    "[bold info]Dataset Structure Requirements[/bold info]\n"
    "Your dataset should be organized as follows:\n"
    "[blue]data_directory/[/blue]\n"
    "├── [desc]train/[/desc]\n"
    "│   ├── [yellow]1/[/yellow] (category number)\n"
    "│   │   └── image1.jpg, image2.jpg, ...\n"
    "│   ├── [yellow]2/[/yellow]\n"
    "│   └── ...\n"
    "├── [desc]valid/[/desc]\n"
    "│   ├── [yellow]1/[/yellow]\n"
    "│   └── ...\n"
    "└── [desc]test/[/desc]\n"
    "    ├── [yellow]1/[/yellow]\n"
    "    └── ...\n\n"
    "[bold yellow]Important Notes:[/bold yellow]\n"
    "• Category numbers should match cat_to_name.json located in the root directory\n"
    "• Each category folder should contain only image files\n"
    "• Supported format: .jpg\n"
)


DATA_STRUCTURE_MESSAGE = (
   "[info]Dataset should have the following structure:[/info]\n"
   "[blue]data_directory/[/blue]\n"
   "[desc]├── train/[/desc] ── [yellow]1/[/yellow] ── [white]image_67823.jpg, image_23456.jpg, ...[/white]\n"
   "[desc]├── valid/[/desc] ── [yellow]1/[/yellow] ── [white]image_89123.jpg, image_45678.jpg, ...[/white]\n"
   "[desc]└── test/ [/desc] ── [yellow]1/[/yellow] ── [white]image_12345.jpg, image_78901.jpg, ...[/white]\n\n"
   "[info]For more information - [/info][desc]'python3 train.py --info'[/desc]\n\n"
)


DATA_PREPROCESS_MESSAGE = (
   "[info]Your are about to preprocess your data:[/info]\n\n"
    "• [desc] Data will be split into train, valid, and test sets.[/desc]\n"
    "• [desc] Transformations will be applied to images[/desc]\n"
    "• [desc] It will be loaded into DataLoaders[/desc]\n\n"
)

MODEL_TRAIN_MESSAGE = (
   "[info]Your are about to train your model:[/info]\n\n"
    "• [desc] Device will be set to [arg]gpu[/arg] if available.[/desc]\n"
    "• [desc] Pre-trained VGG model will be loaded [/desc]\n"
    "• [desc] Models all parameters will be frozen[/desc]\n"
    "• [desc] New Classifier will be created by the specified architecture[/desc]\n"
    "• [desc] Criterion, Optimizer, and Hyperparameters will be initialized[/desc]\n"
    "• [desc] Finally model will be trained [/desc]\n"
)

CURRENT_MODEL_ARCHITECTURE_MESSAGE = (
   "This is the current model classifier architecture.\n"
   "in the next step it will be modified according to the\n"
   "structure you specified in the hidden layers.\n\n"
)


START_MODEL_TRAIN_MESSAGE = (
   "[info]Now it's time to train the model[/info]\n\n"
   "[example]During training:[/example]\n"
   "• [desc] You can track training process with progress bar [/desc]\n"
   "• [desc] You will track train loss, validation loss and accuracy values[/desc]\n"
)

RETRAIN_MODEL_MESSAGE = (
   "[info]You are about to retrain the model[/info]\n"
)

CHOOSE_MODEL_ERROR_MESSAGE = (
   "[error][❌] The Model is required to continue[/error]\n\n"
   "[example]You can specify the model with the following options:[/example]\n"
   "• [desc] Run command: [info]`python3 predict.py --model 'model/folder/model_name.pth`[/info] [/desc]\n"
   "• [desc] Or you can run only: [info]`python3 predict.py`[/info] and choose a model from the file dialog window![/desc]\n"
)

CHOOSE_IMAGE_ERROR_MESSAGE = (
   "[error][❌] The Image is required to make predictions [/error]\n\n"
   "[example]Please press [info][Enter][/info] to choose an image[/example]\n"
)

CONTINUE_WITH_PREDICTION_MESSAGE = (
   "[info]Would you like to make predictions?[/info]\n"
   "\n[example]You have the following options:[/example]\n"
   "• [desc]Manually: [info]`python3 predict.py`[/info][/desc]\n"
   "• [desc]Automatically: choose [info]'Continue with prediction'[/info] from the menu below[/desc]"
)