from pipeline import LoadModel, MakePrediction

def main():
  loaded_model = LoadModel()

  prediction = MakePrediction.start(loaded_model.model)


if __name__ == '__main__':
  main()