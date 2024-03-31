import os
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from keras.models import load_model
from google.colab import drive

drive.mount('/content/drive')
BASE_DIR = "/content/drive/MyDrive/Colab/"

## save model
def save_model(model,accuracy):
  # Get the current date and time
  current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
  accuracy = "{:.3f}".format(accuracy).replace('.', '')
  # Save the model with current date and time in the filename
  filename = BASE_DIR + f'lstm_model_{current_datetime}_{accuracy}'
  print("Model is saved to " + filename)
  model.save(filename)
