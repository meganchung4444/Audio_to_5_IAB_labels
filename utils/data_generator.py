import numpy as np
import time
import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import wave
import config


class Dataset(object):
    """
      Dataset class for processing audio clips.

      This class loads the audio clips from the csv file as input and returns a tensor 
      of the audio waveform and target. 

      Args:
          dataset_file (str): File path to the dataset file 

      Attributes:
          dataframe (pd.DataFrame): Pandas DataFrame with the audio ID and labels.
          labels (pd.Series): Series containing the labels.
          features (pd.DataFrame): DataFrame containing the features.

      Methods:
          __getitem__(idx): Retrieves the audio waveform and target labels of the audio clip at the given index.
          __len__(): Returns the number of audio clips in the dataset.
      
      References:
        Adapted code for the getitem method from https://stackoverflow.com/questions/16778878/python-write-a-wav-file-into-numpy-float-array
    """
    def __init__(self, dataset_file):
      self.dataframe = pd.read_csv(dataset_file, header = 0)
      self.labels = self.dataframe["IAB Vector"] # column name of label
      self.features = self.dataframe.drop(columns = ["IAB Vector"])
        
    def __getitem__(self, idx):
      """
      Retrieves the audio waveform and target labels of the audio clip at the given index.

      Args:
          idx (int): Index of the audio clip.

      Returns:
          dict: A dictionary containing 'audio' (waveform tensor) and 'target' (label tensor) keys.
      """
      # Get audio clip path
      root_path = '/content/drive/My Drive'
      folder = os.path.join(root_path, "GumGum/Notebooks/Panns_inference_files/audioset-processing/output/ORGANIZED_FILES/AllAudioClips")
      audio_file_id = self.features["ID"].iloc[idx]
      audio_path = os.path.join(folder, audio_file_id)
      
      # Read file to get buffer                                                                                               
      audio_file = wave.open(audio_path)
      samples = audio_file.getnframes()
      audio = audio_file.readframes(samples)

      # Convert buffer to float32 using NumPy                                                                                 
      audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
      audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

      # Normalise float32 array so that values are between -1.0 and +1.0                                                      
      max_int16 = 2**15
      audio_normalized = torch.tensor(audio_as_np_float32 / max_int16)

      # Convert all audio waveform tensors to 320000
      if audio_normalized.shape[0] < 320000:
        pad = audio_normalized[-1].repeat(320000 - audio_normalized.shape[0])
        combined_audio_normalized = torch.cat((audio_normalized, pad))
        audio_normalized = combined_audio_normalized

      # Get label and convert to tensor
      label = self.labels.iloc[idx]
      label = label.strip('[]')
      label = [int(val) for val in label if val != ',' and val != ' ']
      label_tensor = torch.tensor(label)
      return {"audio": audio_normalized, "target": label_tensor}
        
    def __len__(self):
      """
      Returns the number of audio clips in the dataset.

      Returns:
          int: Number of audio clips.
      """
      return len(self.dataframe)
