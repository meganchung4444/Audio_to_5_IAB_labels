import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import (sample_rate, classes_num, mel_bins, fmin, fmax, window_size, 
    hop_size, window, pad_mode, center, ref, amin, top_db)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup
from utilities import (create_folder, get_filename, create_logging, StatisticsContainer, Mixup)
from data_generator import AudioDataset
from models import Transfer_Cnn14
from evaluate import Evaluator


def train(args):
  """
  Train, validate, and test a neural network model.

  Args:
      args (argparse.Namespace): Command-line arguments and configurations from the shellfile.
  """
  # Handling the arugments & parameters
  training_dataset_dir = args.training_dataset_dir
  val_dataset_dir = args.val_dataset_dir
  test_dataset_dir = args.test_dataset_dir
  workspace = args.workspace
  holdout_fold = args.holdout_fold
  model_type = args.model_type
  pretrained_checkpoint_path = args.pretrained_checkpoint_path
  freeze_base = args.freeze_base
  loss_type = args.loss_type
  augmentation = args.augmentation
  learning_rate = args.learning_rate
  batch_size = args.batch_size
  max_epoch = args.max_epoch
  device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
  filename = args.filename
  num_workers = 2 

  loss_func = get_loss_func(loss_type)
  pretrain = True if pretrained_checkpoint_path else False

  # Creating folder for logging information
  checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
      'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain), 
      'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
        'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
  create_folder(checkpoints_dir)

  statistics_path = os.path.join(workspace, 'statistics', filename, 
      'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain), 
      'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
      'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base), 
      'statistics.pickle')
  create_folder(os.path.dirname(statistics_path))
  
  logs_dir = os.path.join(workspace, 'logs', filename, 
      'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain), 
      'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
      'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
  create_logging(logs_dir, 'w')
  logging.info(args)

  if 'cuda' in device:
      logging.info('Using GPU.')
  else:
      logging.info('Using CPU. Set --cuda flag to use GPU.')
  
  # Model
  Model = eval(model_type)
  model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
      classes_num, freeze_base)

  # Statistics
  statistics_container = StatisticsContainer(statistics_path)

  if pretrain:
      logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
      model.load_from_pretrain(pretrained_checkpoint_path)

  # Parallel
  print('GPU number: {}'.format(torch.cuda.device_count()))
  # model = torch.nn.DataParallel(model)

  # Dataset Class for training and validation
  training_dataset = AudioDataset(training_dataset_dir)
  validation_dataset = AudioDataset(val_dataset_dir)
  
  # Data Loader for training and validation
  train_loader = torch.utils.data.DataLoader(dataset=training_dataset, 
      batch_size=32, shuffle = True, 
      num_workers=num_workers, pin_memory=True)

  validate_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
      batch_size=1, shuffle = False, 
      num_workers=num_workers, pin_memory=True)

  if 'cuda' in device:
      model.to(device)

  # Optimizer
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
      eps=1e-08, weight_decay=0., amsgrad=True)
    
  # Evaluator
  evaluator = Evaluator(model=model)
  
  full_training_start = time.time()
  val_list = [] # list to check the best f1 score to determine the best checkpoint
  epoch_loss = [] # list to keep track of each epoch's loss
  # Training Loop
  for epoch in range(1, max_epoch + 1):
      print()

      # Train on Mini Batches
      batch_count = 0
      train_bgn_time = time.time()
      total_loss = 0
      total_epoch_training_time = 0
      for batch_data_dict in train_loader:
          
        # Move data to GPU
        for key in batch_data_dict.keys(): 
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
            
        # Train
        model.train()

        # Inference for Training Data
        batch_output_dict = model(batch_data_dict['audio'], None)
        """{'clipwise_output': (batch_size, classes_num), ...}"""
        batch_target_dict = {'target': batch_data_dict['target']}
        """{'target': (batch_size, classes_num)}"""

        # Loss
        loss = loss_func(batch_output_dict, batch_target_dict)
        
        # Backward (to update model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        train_time = time.time() - train_bgn_time 
        total_epoch_training_time += train_time
        logging.info('Epoch #{} for Iteration #{}'.format(epoch, batch_count))
        batch_count += 1
        logging.info('\t• Train Time: {:.3f} s'.format(train_time))
        
        logging.info('\t• Loss: {:.3f}'.format(loss.item())) 

      # Evaluation for every 5th epoch
      if epoch % 5 == 0 and epoch > 0:
          model.eval()

          logging.info('Validation for Epoch #{}'.format(epoch))

          val_begin_time = time.time()

          statistics = evaluator.evaluate(validate_loader)
          logging.info(f"\t• F1 Score: {statistics['f1']}")
          logging.info(f"\t• Classification Report: {statistics['report']}")
          val_list.append(statistics["f1"])

          validate_time = time.time() - val_begin_time
          logging.info('\t• Validate Time: {:.3f} s\n'.format(validate_time))

      # Save model 
      if epoch % 5 == 0 and epoch > 0:
          checkpoint = {
              'epoch': epoch, 
              'model': model.module.state_dict() }

          checkpoint_path = os.path.join(checkpoints_dir, '{}_epochs.pth'.format(epoch))
                      
          torch.save(checkpoint, checkpoint_path)
          logging.info('\t• Model saved to {}'.format(checkpoint_path)) 
      logging.info('------------------------------------') 
      epoch_loss.append(total_loss)
      average_loss = total_loss / len(train_loader)
      logging.info('Average Loss for Epoch #{}: {:.3f}'.format(epoch, average_loss))
      logging.info('Total Training Time for Epoch #{}: {:.3f} s'.format(epoch, total_epoch_training_time))

  logging.info('------------------------------------')                  
  total_training_time = time.time() - full_training_start 
  logging.info('Average Overall Loss: {:.3f} s'.format(sum(epoch_loss)/len(epoch_loss))) 
  logging.info('Average Overall F1: {:.3f} s'.format(sum(val_list)/len(val_list))) 
  logging.info('Full Training Time: {:.3f} s'.format(total_training_time)) 
  plot_loss_and_f1(epoch_loss, val_list)

  # Find and Load in Best Checkpoint (based off F1 Score)
  best_checkpoint = np.argmax(np.array(val_list)) 
  best_checkpoint_idx = best_checkpoint * 5
  logging.info('Best Checkpoint Found at Epoch {}'.format(best_checkpoint_idx))  
  best_checkpoint_path = f"/content/checkpoints/main/holdout_fold=1/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=none/batch_size=32/freeze_base=False/{best_checkpoint_idx}_epochs.pth"
  model.load_state_dict(torch.load(best_checkpoint_path)['model']) # choose the best checkpoint and load it

  # Dataset and Dataloader for Testing
  test_dataset = AudioDataset(test_dataset_dir)
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
      batch_size=1, shuffle = False,
      num_workers=num_workers, pin_memory=True)
  
  # Testing Loop
  testing_begin_time = time.time()
  model.eval()
  statistics = evaluator.evaluate(test_loader)
  logging.info('Testing F1 Score: {:.3f}'.format(statistics['f1']))
  testing_time = time.time() - testing_begin_time
  logging.info('Total Testing Time: {:.3f} s'.format(testing_time))

def plot_loss_and_f1(loss_arr, f1_arr):

  f1_range = list(range(5, 101, 5))  # for every 5th epoch
  loss_range = list(range(1, 101, 1))  # for all 100 epochs

  fig, ax1 = plt.subplots(figsize=(10, 5))

  ax1.plot(loss_arr, loss_range, marker='o', linestyle='-', color='b', label='F1 Score')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('F1 Score', color='b')
  ax1.tick_params('y', colors='b')

  ax2 = ax1.twinx()
  ax2.plot(f1_arr, f1_range, marker='o', linestyle='-', color='r', label='Loss')
  ax2.set_ylabel('Loss', color='r')
  ax2.tick_params('y', colors='r')

  lines, labels = ax1.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()
  ax2.legend(lines + lines2, labels + labels2, loc='upper left')

  plt.title('F1 Score and Loss over Epochs')
  plt.grid(True)
  filename = f"loss_and_f1_plot.png"
  filepath = os.path.join("/content/figures/", filename) 
  
  plt.savefig(filepath)
  



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Example of parser. ')
  subparsers = parser.add_subparsers(dest='mode')

  # Train
  parser_train = subparsers.add_parser('train')
  parser_train.add_argument('--training_dataset_dir', type=str, required=True, help='Directory of training dataset.')
  parser_train.add_argument('--val_dataset_dir', type=str, required=True, help='Directory of validation dataset.')
  parser_train.add_argument('--test_dataset_dir', type=str, required=True, help='Directory of testing dataset.')
  parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
  parser_train.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], required=True)
  parser_train.add_argument('--model_type', type=str, required=True)
  parser_train.add_argument('--pretrained_checkpoint_path', type=str)
  parser_train.add_argument('--freeze_base', action='store_true', default=False)
  parser_train.add_argument('--loss_type', type=str, required=True)
  parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
  parser_train.add_argument('--learning_rate', type=float, required=True)
  parser_train.add_argument('--batch_size', type=int, required=True)
  parser_train.add_argument("--max_epoch", type = int, required = True)
  parser_train.add_argument('--cuda', action='store_true', default=False)

  # Parse arguments
  args = parser.parse_args()
  args.filename = get_filename(__file__)

  if args.mode == 'train':
      train(args)

  else:
      raise Exception('Error argument!')
