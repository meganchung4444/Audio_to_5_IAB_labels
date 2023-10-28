#!/bin/bash

# DATASET_DIR="/home/tiger/datasets/GTZAN/dataset_root"
TRAINING_DATASET_DIR="/content/drive/MyDrive/GumGum/Notebooks/Panns_inference_files/audioset-processing/output/ORGANIZED FILES/Dataset/GumGum_Training_Set.csv"
VAL_DATASET_DIR="/content/drive/MyDrive/GumGum/Notebooks/Panns_inference_files/audioset-processing/output/ORGANIZED FILES/Dataset/GumGum_Validation_Set.csv"
WORKSPACE="/content" 

# python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --mini_data

PRETRAINED_CHECKPOINT_PATH="/content/Cnn14_mAP=0.431.pth"
# MODEL_TYPE="Transfer_Cnn13"
# CUDA_VISIBLE_DEVICES=3 python3 pytorch/main.py train --training_dataset_dir=$TRAINING_DATASET_DIR --val_dataset_dir=$VAL_DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='none' --learning_rate=1e-4 --batch_size=32 --max_epoch=400 --cuda
CUDA_VISIBLE_DEVICES=0 python3 /content/panns_transfer_to_gtzan/pytorch/main.py train --training_dataset_dir=$TRAINING_DATASET_DIR --val_dataset_dir=$VAL_DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='none' --learning_rate=1e-4 --batch_size=32 --max_epoch=400 --cuda


#####

# PRETRAINED_CHECKPOINT_PATH="/vol/vssp/msos/qk/bytedance/workspaces_important/pub_audioset_tagging_cnn_transfer/checkpoints/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/660000_iterations.pth"
# python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --freeze_base --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --few_shots=10 --random_seed=1000 --resume_iteration=0 --stop_iteration=10000 --cuda

# python3 utils/plot_statistics.py 1 --workspace=$WORKSPACE --select=2_cnn13
