from utils import get_train_terminal_args
import sys
import argparse

def main():
  try:
    args = get_train_terminal_args()
  except argparse.ArgumentError as e:
    print('eeeeeeeeeeee')
    sys.exit(1)

if __name__ == '__main__':
  main()