import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
import matplotlib.pyplot as plt
from pytorch_utils import forward
from utilities import get_filename
import config
import os


class Evaluator(object):
  """
  Evaluator class for evaluating the model.

  Args:
      model: The classification model to be evaluated and compared with the labels from the dataset.

  Attributes:
      model: The classification model.
      png_counter (int): Counter for file names of the confusion matrices.
      num_classes (int): Total number of classes
      classes (list): List of class names.

  References:
    Adapated code to print multiple confusion matrices from https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
  """
  def __init__(self, model, workspace):
      self.model = model
      self.png_counter = 5
      self.classes = ["Automotive", "Food & Drink", "Pets", "War & Conflicts", "Music"]
      self.num_classes = len(self.classes)
      self.workspace = workspace

  def evaluate(self, data_loader):
      """
      Evaluate the model on the given data loader.

      Args:
          data_loader: DataLoader providing all the audio waveform and target tensor in a dataset.

      Returns:
          dict: A dictionary with the evaluation statistics of the averaged F1 score and the overall classification report.
      """
      # Forward
      output_dict = forward(
          model=self.model, 
          generator=data_loader, 
          return_target=True)

      # Retrieve the model's predictions (clipwise_output) and labels from the dataset (target)
      model_prediction = output_dict['clipwise_output']   # (audios_num, classes_num)
      target = output_dict['target'].astype(float)   # (audios_num, classes_num)
      
      # Convert model's prediction to be binary 
      sigmoid_model_prediction = 1/(1 + np.exp(-model_prediction))
      threshold = 0.5
      binary_model_predictions = (sigmoid_model_prediction > threshold).astype(float)
      
      print("\n", classification_report(target, binary_model_predictions, target_names = self.classes))
      report = classification_report(target, binary_model_predictions, target_names=self.classes, output_dict = True, zero_division=0)
      statistics = {'f1': report['macro avg']['f1-score'], "report": report}
      self.plot_cm(target, binary_model_predictions, self.num_classes)
      self.png_counter += 5

      return statistics

  def plot_cm(self, dataset_label, model_prediction, num_classes):
    """
    Plot confusion matrix for each class.

    Args:
        dataset_label: Ground truth labels.
        model_prediction: Model predictions.
        num_classes: Number of classes.
    """
    f, axes = plt.subplots(1, num_classes, figsize=(8, 5))
    axes = axes.ravel()

    # Confusion Matrix Order:
    # True Neg (Top-Left), False Pos (Top-Right), False Neg (Bottom-Left), True Pos (Bottom-Right)
    for i in range(num_classes):
        disp = ConfusionMatrixDisplay(confusion_matrix(dataset_label[:, i],
                                                      model_prediction[:, i]),
                                      display_labels=[0, i])
        disp.plot(ax=axes[i], values_format='.4g')
        disp.ax_.set_title(f'{self.classes[i]}', fontsize = 8)
        if i < 10:
            disp.ax_.set_xlabel('')
        if i % 5 != 0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()
    
    plt.subplots_adjust(wspace=0.50, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)

    filename = f"epoch_cm_{self.png_counter}.png"
    folder = f"{self.workspace}/figures"
    if not os.path.exists(folder):
      os.makedirs(folder)
    filepath = os.path.join(folder, filename) 
    # filepath = os.path.join("/content/figures/", filename) 
    
    plt.savefig(filepath)
