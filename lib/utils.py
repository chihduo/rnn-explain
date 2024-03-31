import os
import math
import random
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix

def save_model(model, accuracy):
  # Get the current date and time
  current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
  accuracy = "{:.3f}".format(accuracy).replace('.', '')
  # Save the model with current date and time in the filename
  filename = BASE_DIR + f'lstm_model_{current_datetime}_{accuracy}'
  print("Model is saved to " + filename)
  model.save(filename)

def check_swing_trade(seq):
    NaN = float('nan')
    def lt(a,b):
      return ~math.isnan(a) and ~math.isnan(b) and a < b
    prev1 = NaN
    prev2 = NaN
    prev3 = NaN
    last_peak = NaN
    for i in range(len(seq)):
      prev3 = prev2
      prev2 = prev1
      prev1 = seq[i]
      if lt(prev3,prev2) and lt(prev1,prev2):
        last_peak = prev2
    return lt(last_peak, seq[-1])

def create_test_examples(num_examples, num_columns, labeler):
  def generate_random_lists():
    list_of_lists = []
    list_size = num_columns #random.randint(4, num_columns)
    for _ in range(1, num_examples):
      seq = list()
      for _ in range(num_columns - list_size):
        seq.append(0)
      for _ in range(list_size):
        seq.append(random.randint(-6, 6) * 0.5)
      seq.append(1 if labeler(seq) else 0)
      list_of_lists.append(seq)
    return list_of_lists

  lists = generate_random_lists()
  print("before removing duplicate:", len(lists))
  unique_lists = []
  for lst in lists:
    if lst not in unique_lists:
      unique_lists.append(lst)
  print("after removing duplicate:", len(unique_lists))
  return unique_lists

def plot_training_performance(histories):
  for h in histories:
    loss = h.history["loss"]
    #val_loss = train_history.history["val_loss"]
    #plt.plot(val_loss,label="val_loss")
    plt.plot(loss,label="loss")
    plt.ylabel("loss & acc")
    plt.xlabel("epoch")
    plt.title("model loss")
    plt.legend(["train"],loc = "upper left")
    #acc = train_history.history["acc"]
    #plt.plot(acc,label="acc")
    plt.xlabel("epoch")
    plt.title("model - train")
    plt.legend(["loss","val_loss","acc"],loc = "upper left")

def compute_accuracy(models, test_data_x, test_data_y):
  result = pd.DataFrame()
  preds = []
  for i in range(0, len(models)):
    pred = models[i].predict(test_data_x)
    pred = pred.squeeze()
    pred = list(map(lambda x: 1 if x > 0.5 else 0, pred))
    result["P" + str(i)] = pred
    preds.append(pred)
  pred = list(zip(*preds))
  result["Predict"] = list(map(lambda x: 1 if sum(x) >= len(models)/2 else 0, pred))
  result["Truth"] = test_data_y.iloc[:, 0]

  correct = (result['Truth'] == result['Predict']).sum()
  total_predictions = len(result)
  accuracy = correct / total_predictions
  fig = plt.figure
  plot = result.groupby(["Predict","Truth"]).size().plot(kind="barh", color="grey")
  truth = list(result["Truth"])
  pred = list(result["Predict"])
  conf_matrix = confusion_matrix(truth, pred)
  print("Accuracy:", accuracy)
  print("Confusion Matrix:")
  print(conf_matrix)
  return accuracy,result
