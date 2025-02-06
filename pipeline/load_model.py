from pipeline import SetupModel

class LoadModel:
  def __init__(self):
    # Initialize the model
    self._setup = SetupModel()
    
  @property
  def classifier(self):
    return self._setup.classifier

  @staticmethod
  def start():
    return LoadModel()