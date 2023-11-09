import numpy as np
import logging
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from pytorch_utils import forward
from utilities import get_filename
import config


def calculate_accuracy(y_true, y_score):
    N = y_true.shape[0]
    accuracy = np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_score, axis=-1)) / N
    return accuracy


class Evaluator(object):
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader):

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, # will be validation_loader
            return_target=True)

        # predict classes
        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        # label from dataset
        target = output_dict['target']    # (audios_num, classes_num)
        print("clipwise_output:", clipwise_output)
        print("target:", target)
        print()

        cm = metrics.confusion_matrix(np.argmax(target, axis=-1), np.argmax(clipwise_output, axis=-1), labels=None)
        self.plot_cm(np.argmax(target, axis = -1), np.argmax(clipwise_output, axis=-1))
        # accuracy = calculate_accuracy(target, clipwise_output)
        # (y_true arr, y_label arr)
        f1 = metrics.f1_score(target, clipwise_output, average = "weighted")
        statistics = {'f1': f1}

        return statistics
    def plot_cm(self, dataset_label, model_prediction):
        cm = metrics.confusion_matrix(dataset_label, model_prediction)
        labels = np.unique(dataset_label)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap = "Blues", values_format = "d")
