from utils import get_train_terminal_args, WelcomeMessage
from rich import console

console = console.Console()

def main():

  # Welcome message
  start_training = WelcomeMessage()
  
  # If the user does not want to continue, exit
  if not start_training:
    return
  
  # Set data directory
  start_training.set_data_directory()

  # Get terminal arguments
  args = get_train_terminal_args()

  print(args)

if __name__ == '__main__':
  main()