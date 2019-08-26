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

def create_dataloader_full(data, sequence_length, batch_size, train_percentage=0.7, eval_percentage=0.15):
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

    idx_eval = math.trunc(len(inputs) * train_percentage)
    idx_test = (math.trunc(len(inputs) * eval_percentage)) + idx_eval

    dataset_train = TensorDataset(torch.from_numpy(inputs[:idx_eval]), torch.from_numpy(targets[:idx_eval]))
    dataset_eval = TensorDataset(torch.from_numpy(inputs[idx_eval:idx_test]), torch.from_numpy(targets[idx_eval:idx_test]))
    dataset_test = TensorDataset(torch.from_numpy(inputs[idx_test:]), torch.from_numpy(targets[idx_test:]))

    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, drop_last=False)
    dataloader_eval = DataLoader(dataset_eval, shuffle=True, batch_size=len(dataset_eval), drop_last=False)
    dataloader_test = DataLoader(dataset_test, shuffle=False, batch_size=len(dataset_test), drop_last=False)

    return (dataloader_train, dataloader_eval, dataloader_test)