from pipeline import SetupModel

class LoadModel:
  def __init__(self):
    self.pre = SetupModel()

  @staticmethod
  def start():
    LoadModel()