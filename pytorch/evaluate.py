import numpy as np
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt

from pytorch_utils import forward
from utilities import get_filename
import config
import os



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

        # predicted classes
        clipwise_output = output_dict['clipwise_output']   # (audios_num, classes_num)
        # label from dataset
        target = output_dict['target'].astype(float)   # (audios_num, classes_num)
        
        sigmoid_clipwise_output = 1/(1 + np.exp(-clipwise_output))
        threshold = 0.5
        binary_model_predictions = (sigmoid_clipwise_output > threshold).astype(float)
        
        classes = ["Automotive", "Food & Drink", "Pets", "War & Conflicts", "Music"]
        print(classification_report(target, binary_model_predictions, target_names = classes))
        report = classification_report(target, binary_model_predictions, target_names=classes, output_dict = True, zero_division=0)
        statistics = {'f1': report['macro avg']['f1-score'], "report": report}
        self.plot_cm(target, binary_model_predictions, self.num_classes)
        self.png_counter += 5

        return statistics

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
