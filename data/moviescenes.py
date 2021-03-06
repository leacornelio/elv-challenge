import numpy as np
import pickle
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader


class MovieScenes(Dataset):
    def __init__(self, data_root, mode, seq_len, shot_num, **kwargs):
        self.data_root = data_root
        self.offset = seq_len * shot_num
        self.seq_len = seq_len
        self.shot_num = shot_num
        self.movie_data = parse_data(data_root, mode, self.offset, seq_len, shot_num)
        self.mode = mode
    
    def __len__(self):
        return len(self.movie_data['scene_transition_boundary_ground_truth'])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        sample = []
        for key in ['place', 'cast', 'action', 'audio']:
            data = []
            sub_data = []
            # Divide segments into clips
            for i in range(self.offset):
                if key == 'audio':
                    sub_data.append(pad_audio(self.movie_data[key][index][i]))
                else:
                    sub_data.append(self.movie_data[key][index][i])
                if (i+1)%self.shot_num == 0:
                    data.append(sub_data)
                    sub_data = [] 
            sample.append(torch.tensor(data))
        
        sample = tuple(sample)
        labels = torch.tensor(self.movie_data['scene_transition_boundary_ground_truth'][index], dtype=torch.long)

        if self.mode == 'train':
            return sample, labels
        else:
            return sample, labels, self.movie_data['imdb_id'][index], self.movie_data['shot_end_frame'][index]


def pad_audio(audio):
    """
    Pads audio data to the input size of the model.

    Args:
        audio (numpy.ndarray): Audio data.

    Returns:
        reshaped_audio (numpy.ndarray): Padded and reshapred audio data.

    """
    padded_audio = np.concatenate((audio, np.zeros((257*90)-512, dtype="float32")))
    reshaped_audio = np.reshape(padded_audio, (257, 90))

    return reshaped_audio


def parse_data(data_root, mode, offset, seq_len, shot_num):
    """
    Parses and processes the data for the dataloader.

    Args:
        data_root (string): Root directory of the data.
        mode (string): Train or test.
        offset (int): Total number of shots given a sequence length.
        shot_num (int): Number of shots.

    Returns:
        movie_data (dict): Dictionary containing the processed and parsed data.

    """
    with open(data_root, 'r') as data_paths:
        movie_data = {
            'place': [], 
            'cast': [], 
            'action':[], 
            'audio': [], 
            'scene_transition_boundary_ground_truth': [], 
            'imdb_id': [], 
            'shot_end_frame': []
        }
        for data_path in data_paths:
            pickleFile = open(data_path.strip(), 'rb')
            data = pickle.load(pickleFile)

            for key in movie_data.keys():
                if key not in ['imdb_id', 'shot_end_frame']:
                    # Limit samples due to limited training time
                    feat = data[key][:100].cpu().detach().numpy()
                    
                    if key == 'scene_transition_boundary_ground_truth':
                        split_data = []
                        # Obtain shot boundary per sequence
                        for i in range(shot_num, len(feat), shot_num):  
                            split_data.append(feat[i])
                            if len(split_data) == seq_len:  
                                movie_data[key].append(split_data)
                                split_data = []
                                movie_data['imdb_id'].append(data['imdb_id'])
                                movie_data['shot_end_frame'].append(data['shot_end_frame'])
                    else:
                        # Partition semantic elements into segments
                        for i in range(0, len(feat), offset):
                            if i + offset < len(feat):
                                split_data = []
                                for j in range(offset):
                                    split_data.append(feat[i+j])
                                movie_data[key].append(split_data)

        for key in movie_data.keys():
            if key not in ['imdb_id', 'shot_end_frame']:
                movie_data[key] = np.array(movie_data[key])

        return movie_data
