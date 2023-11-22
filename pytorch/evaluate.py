import numpy as np
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, roc_auc_score
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
        self.num_classes = 5

    def evaluate(self, data_loader):

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)

        # predict classes
        clipwise_output = output_dict['clipwise_output']   # (audios_num, classes_num)
        # label from dataset
        target = output_dict['target']    # (audios_num, classes_num)
        print("target dimension:", target.shape)
        # accuracy = calculate_accuracy(target, clipwise_output)

        # convert the predicted labels into probabilities
        # prob_clipwise_output = np.exp(clipwise_output) # might use sigmoid?
        # threshold = 0.5
        # prob_clipwise_output[prob_clipwise_output > threshold] = 1
        # prob_clipwise_output[prob_clipwise_output <= threshold] = 0

        # f1 = f1_score(target, prob_clipwise_output, average = "weighted")
        
        # ////// ROC CURVE PART //////
        # for each label (i), compute the roc_curve
        class_metrics = {}
        prob_clipwise_output = 1/(1 + np.exp(-clipwise_output)) # sigmoid 
        print("prob_clipwise_output:", prob_clipwise_output)
        predicted_output = np.zeros_like(prob_clipwise_output)
        for i in range(self.num_classes):
          fpr, tpr, thresholds = roc_curve(target[:, i], prob_clipwise_output[:, i])
          # how confident the model is for being right; range: 
          auc = roc_auc_score(target[:, i], prob_clipwise_output[:, i])
        
          youden_j_stat = tpr - fpr
          print("tpr:", tpr)
          print("fpr:", fpr)
          print("youden_j_stat:", youden_j_stat, "w/ dimensions", youden_j_stat.shape)

          optimal_threshold = thresholds[np.argmax(youden_j_stat)]
          print(f"optimal threshold for class {i}: {optimal_threshold} at index {np.argmax(youden_j_stat)}")
          # predicted_output[:, i] = 1 if prob_clipwise_output[:, i] >= optimal_threshold else 0
          predicted_output = np.where(prob_clipwise_output >= optimal_threshold, 1, 0)
          class_metrics[i] = {
              "fpr" : fpr,
              "tpr" : tpr,
              "auc" : auc,
              "threshold" : optimal_threshold
          }
        print("predicted_output (after loop):", predicted_output)
        self.plot_roc(class_metrics, self.num_classes)

  
        self.plot_cm(target, predicted_output, self.num_classes)

        # TO DO: f1 based on 0.5 vs optimal threshold
        f1 = f1_score(target, predicted_output, average = "weighted")
        statistics = {'f1': f1}
        self.png_counter += 5

        return statistics
    
    def plot_roc(self, classes_metrics, num_classes):
      classes = ["Automotive", "Food & Drink", "Pets", "War & Conflicts", "Music"]
      f, axes = plt.subplots(1, num_classes, figsize=(15, 4))
      axes = axes.ravel()

      for i in range(num_classes):
        metric = classes_metrics[i]
        axes[i].plot(metric["fpr"], metric["tpr"], label=f'Class {i + 1} (AUC = {metric["auc"]:.2f}, Threshold = {metric["threshold"]:.2f})')
        axes[i].set_title(f'{classes[i]}', fontsize = 8)
        if i < 10:
            axes[i].set_xlabel('')
        if i % 5 != 0:
            axes[i].set_ylabel('')
        axes[i].legend(loc = "lower right", bbox_to_anchor=(1.0, 0.0), fontsize= 6)

      plt.subplots_adjust(wspace=0.50, hspace=0.1)

      filename = f"epoch_roc_{self.png_counter}.png"
      filepath = os.path.join("/content/figures/", filename) 
      
      plt.savefig(filepath)

    def plot_cm(self, dataset_label, model_prediction, num_classes):
       
      
      f, axes = plt.subplots(1, num_classes, figsize=(8, 5))
      axes = axes.ravel()
      classes = ["Automotive", "Food & Drink", "Pets", "War & Conflicts", "Music"]
      # Confusion Matrix Order:
      # True Neg (Top-Left), False Pos (Top-Right), False Neg (Bottom-Left), True Pos (Bottom-Right)
      for i in range(num_classes):
          disp = ConfusionMatrixDisplay(confusion_matrix(dataset_label[:, i],
                                                        model_prediction[:, i]),
                                        display_labels=[0, i])
          disp.plot(ax=axes[i], values_format='.4g')
          disp.ax_.set_title(f'{classes[i]}', fontsize = 8)
          if i < 10:
              disp.ax_.set_xlabel('')
          if i % 5 != 0:
              disp.ax_.set_ylabel('')
          disp.im_.colorbar.remove()
      
      plt.subplots_adjust(wspace=0.50, hspace=0.1)
      f.colorbar(disp.im_, ax=axes)

      filename = f"epoch_cm_{self.png_counter}.png"
      filepath = os.path.join("/content/figures/", filename) 
      
      plt.savefig(filepath)
      

      
