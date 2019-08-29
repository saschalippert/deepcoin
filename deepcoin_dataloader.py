import numpy as np
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
import math

def create_dataloader_train(data, sequence_length, batch_size):
    window_len = sequence_length + 1
    sequences = len(data) - window_len

    inputs = np.zeros((sequences, sequence_length), dtype=np.float32)
    targets = np.zeros((sequences), dtype=np.float32)

    for start in range(0, sequences):
        end = start + sequence_length

        inputs[start] = np.array(data[start:end])
        targets[start] = np.array(data[end])

    dataset_train = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))

    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, drop_last=False)

    return dataloader_train

def create_dataloader_full(data, sequence_length, batch_size, test_percentage=0.75, balance = False):
    window_len = sequence_length + 1
    sequences = len(data) - window_len

    inputs = np.zeros((sequences, sequence_length), dtype=np.float32)
    targets = np.zeros((sequences), dtype=np.float32)

    for start in range(0, sequences):
        end = start + sequence_length

        inputs[start] = np.array(data[start:end])
        targets[start] = np.array(data[end])

    idx_shuffle = list(range(0, len(inputs)))
    random.shuffle(idx_shuffle)

    inputs = inputs[idx_shuffle]
    targets = targets[idx_shuffle]

    if(balance):
        idx_pos = []
        idx_neg = []

        for i in range(0, sequences):
            first_target = targets[i]

            if (first_target > 0):
                idx_pos.append(i)
            else:
                idx_neg.append(i)

        len_pos = len(idx_pos)
        len_neg = len(idx_neg)
        len_balanced = min(len_pos, len_neg)

        inputs = inputs[idx_pos[0:len_balanced]]
        targets = targets[idx_neg[0:len_balanced]]

    idx_eval = math.trunc(len(inputs) * test_percentage)

    dataset_train = TensorDataset(torch.from_numpy(inputs[:idx_eval]), torch.from_numpy(targets[:idx_eval]))
    dataset_eval = TensorDataset(torch.from_numpy(inputs[idx_eval:-1]), torch.from_numpy(targets[idx_eval:-1]))

    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, drop_last=False)
    dataloader_eval = DataLoader(dataset_eval, shuffle=True, batch_size=len(dataset_eval), drop_last=False)

    return (dataloader_train, dataloader_eval)