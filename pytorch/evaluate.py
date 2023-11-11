import numpy as np
import logging
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from pytorch_utils import forward
from utilities import get_filename
import config
import os


def calculate_accuracy(y_true, y_score):
    N = y_true.shape[0]
    accuracy = np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_score, axis=-1)) / N
    return accuracy


class Evaluator(object):
    def __init__(self, model):
        self.model = model
        self.png_counter = 5

    def evaluate(self, data_loader):

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, # will be validation_loader
            return_target=True)

        # predict classes
        clipwise_output = output_dict['clipwise_output']   # (audios_num, classes_num)
        # label from dataset
        target = output_dict['target']    # (audios_num, classes_num)

        # accuracy = calculate_accuracy(target, clipwise_output)

        # convert the predicted labels into probabilities
        prob_clipwise_output = np.exp(clipwise_output)
        threshold = 0.2
        prob_clipwise_output[prob_clipwise_output > threshold] = 1
        prob_clipwise_output[prob_clipwise_output <= threshold] = 0

        f1 = metrics.f1_score(target, prob_clipwise_output, average = "weighted")
        self.plot_cm(target, prob_clipwise_output)
        
        statistics = {'f1': f1}

        return statistics

    def plot_cm(self, dataset_label, model_prediction):
       
      num_classes = 5
      
      f, axes = plt.subplots(1, num_classes, figsize=(8, 5))
      axes = axes.ravel()
      classes = ["Automotive", "Food & Drink", "Pets", "War & Conflicts", "Music"]
      for i in range(num_classes):
          disp = ConfusionMatrixDisplay(confusion_matrix(dataset_label[:, i],
                                                        model_prediction[:, i]),
                                        display_labels=[0, i])
          disp.plot(ax=axes[i], values_format='.4g')
          disp.ax_.set_title(f'class {classes[i]}')
          if i < 10:
              disp.ax_.set_xlabel('')
          if i % 5 != 0:
              disp.ax_.set_ylabel('')
          disp.im_.colorbar.remove()

      plt.subplots_adjust(wspace=0.30, hspace=0.1)
      f.colorbar(disp.im_, ax=axes)

      filename = f"{self.plot_counter}.png"
      filepath = os.path.join("/content/figures/", filename) 
      plt.savefig(filepath)
      

      
