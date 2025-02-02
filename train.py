from utils import get_train_terminal_args, WelcomeMessage
from rich import console

console = console.Console()

def main():

  # Welcome message
  start_training = WelcomeMessage()
  
  # If the user does not want to continue, exit
  if not start_training:
    return
  
  # Check if we need to get data directory interactively
  args = get_train_terminal_args()
  if args is None:
    start_training.set_data_directory()
    args = get_train_terminal_args()

  print(args)

if __name__ == '__main__':
  main()