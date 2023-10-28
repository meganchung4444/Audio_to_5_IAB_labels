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
# from data_generator import GtzanDataset, TrainSampler, EvaluateSampler, collate_fn
from data_generator import GtzanDataset
from models import Transfer_Cnn14
from evaluate import Evaluator


def train(args):

    # Arugments & parameters
    training_dataset_dir = args.training_dataset_dir
    val_dataset_dir = args.val_dataset_dir
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
    # resume_iteration = args.resume_iteration
    # stop_iteration = args.stop_iteration
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = 2 # updated from 8

    loss_func = get_loss_func(loss_type)
    pretrain = True if pretrained_checkpoint_path else False
    
    hdf5_path = os.path.join(workspace, 'features', 'waveform.h5')

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
    model = torch.nn.DataParallel(model)

    training_dataset = GtzanDataset(training_dataset_dir)
    validation_dataset = GtzanDataset(val_dataset_dir)
    
    # Data loader
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
    
    # iteration means updating the value once
    # for every epoch, model will trained num of training samples/batch size
    full_training_start = time.time()
    for epoch in range(max_epoch):
        # Evaluate/validate for every 5 epoch
        if epoch % 5 == 0 and epoch > 0:
            model.eval()
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(epoch))

            val_begin_time = time.time()

            statistics = evaluator.evaluate(validate_loader)
            logging.info('Validate accuracy: {:.3f}'.format(statistics['accuracy']))

            # train_time = val_begin_time - train_bgn_time
            validate_time = time.time() - val_begin_time
            logging.info(
                'Validate time: {:.3f} s'
                ''.format(validate_time))
            # logging.info(
            #     'Train time: {:.3f} s, validate time: {:.3f} s'
            #     ''.format(train_time, validate_time))

        # Train on mini batches
        # 800 training samples, batch size 32, train_loader made 800/32 batches 
        train_bgn_time = time.time()
        for batch_data_dict in train_loader:
            
            # Move data to GPU
            # batch_data_dict: {"audio": audio_normalised, "target": label_tensor}
            for key in batch_data_dict.keys(): 
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
                
            # Train
            model.train()

            # inference for training data
            # target that model predicted
            batch_output_dict = model(batch_data_dict['audio'], None)
            """{'clipwise_output': (batch_size, classes_num), ...}"""
            # target: label
            batch_target_dict = {'target': batch_data_dict['target']}
            """{'target': (batch_size, classes_num)}"""

            # loss
            loss = loss_func(batch_output_dict, batch_target_dict)
            
            # Backward (to update model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_time = train_bgn_time - time.time()
            logging.info('Train time (for this epoch): {:.3f} s'
                        ''.format(train_time))
            # Save model 
            if epoch % 5 == 0 and epoch > 0:
                checkpoint = {
                    'epoch': epoch, 
                    'model': model } # checkpoint dict (before: model.module.state_dict())

                checkpoint_path = os.path.join(checkpoints_dir, '{}_epochs.pth'.format(epoch))
                            
                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))

            print(epoch, loss.item())
    total_training_time = full_training_start - time.time()
    logging.info('Train time: {:.3f} s'
                        ''.format(total_training_time)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--training_dataset_dir', type=str, required=True, help='Directory of training dataset.')
    parser_train.add_argument('--val_dataset_dir', type=str, required=True, help='Directory of validation dataset.')
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
    # parser_train.add_argument('--resume_iteration', type=int)
    # parser_train.add_argument('--stop_iteration', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')
