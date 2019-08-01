import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import visdom

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, drop_prob):
        super(Model, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.net = nn.LSTM(input_size, hidden_size, num_layers = n_layers, dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)

        batch_size = input.size(0)

        hidden = model.init_hidden(batch_size)
        out, _ = self.net(input, hidden)
        out = out[:,-1,:] #only last sequence is evaluated
        out = self.fc(out)

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        return (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device),
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device))

seq_length = 64
input_size = 1
output_size = 1
hidden_dim = 128
n_layers = 2
batch_size = 512
n_epoches = 1000
drop_prob = 0.5

time = np.arange(0.001, 100, 0.01);
data  = np.sin(time)

def create_dataloader(data, sequence_length, batch_size, train_percentage = 0.75, shuffle_indices = True):
    window_len = sequence_length + 1
    sequences = len(data) - window_len

    inputs = np.zeros((sequences, sequence_length), dtype=np.float32)
    targets = np.zeros((sequences), dtype=np.float32)

    for start in range(0, sequences):
        end = start + sequence_length

        inputs[start] = np.array(data[start:end])
        targets[start] = np.array(data[end])

    i_train = math.trunc(len(inputs) * train_percentage)

    indices = list(range(sequences))

    if shuffle_indices:
        random.shuffle(indices)

    indices_train = indices[:i_train]
    indices_validation = indices[i_train:]

    dataset_train = TensorDataset(torch.from_numpy(inputs[indices_train]), torch.from_numpy(targets[indices_train]))
    dataset_validation = TensorDataset(torch.from_numpy(inputs[indices_validation]), torch.from_numpy(targets[indices_validation]))

    dataloader_train = DataLoader(dataset_train, shuffle=False, batch_size=batch_size, drop_last=True)
    dataloader_validation = DataLoader(dataset_validation, shuffle=False, batch_size=batch_size, drop_last=True)

    return dataloader_train, dataloader_validation

dataloader_train, dataloader_validation = create_dataloader(data, seq_length, batch_size)

def test_model(model, criterion, data, seq_length):
    model.eval()

    predict_len = len(data) - seq_length

    gen_seq = np.array(data[0:seq_length], dtype=np.float32)
    gen_out = np.zeros((predict_len), dtype=np.float32)

    loss_sum = 0.0

    for i in range(0, predict_len):
        data_offset = seq_length + i

        gen_seq_torch = torch.from_numpy(gen_seq)

        input = gen_seq_torch.reshape((1, seq_length, 1)).to(device)
        target = torch.tensor([[[data[data_offset]]]]).to(device)

        out = model(input)

        loss = criterion(out.view(1, 1, 1), target)

        np_out = out.detach().cpu().numpy()
        loss_sum += loss.detach().cpu().numpy()

        gen_out[i] = np_out

        gen_seq[0] = np_out
        gen_seq = np.roll(gen_seq, -1)

    loss_sum /= predict_len

    return gen_out, loss_sum

def train_model(model, optimizer, criterion, n_epochs):
    best_loss = float("inf")
    best_model = None

    epoch_losses = []
    validation_losses = []

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        model.train()

        epoch_loss = 0

        for batch_i, (inputs, targets) in enumerate(dataloader_train):
            inputs = inputs.reshape((batch_size, seq_length, 1)).to(device)
            targets = targets.reshape((batch_size, 1)).to(device)

            optimizer.zero_grad()

            out = model(inputs)

            loss = criterion(out, targets)

            loss.backward()

            clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            np_loss = loss.detach().cpu().numpy()

            epoch_loss += np_loss

        epoch_loss = epoch_loss / (batch_size * len(dataloader_train))

        epoch_losses.append(epoch_loss)

        model.eval()

        validation_loss = 0

        for batch_i, (inputs, targets) in enumerate(dataloader_validation):
            inputs = inputs.reshape((batch_size, seq_length, 1)).to(device)
            targets = targets.reshape((batch_size, 1)).to(device)

            out = model(inputs)

            loss = criterion(out, targets)

            np_loss = loss.detach().cpu().numpy()

            validation_loss += np_loss

        validation_loss = validation_loss / (batch_size * len(dataloader_validation))

        validation_losses.append(validation_loss)

        if(validation_loss < best_loss):
            best_loss = validation_loss
            best_model = Model(input_size, output_size, hidden_dim, n_layers, drop_prob).to(device)
            best_model.load_state_dict(model.state_dict())

        print('Epoch: {:>4}/{:<4} Loss: {:.10f} AvgLoss: {:.10f} Valoss: {:.10f} AvgValoss: {:.10f} BstValLoss: {:.10f}'.format(epoch_i, n_epochs, epoch_loss, np.average(epoch_losses), validation_loss, np.average(validation_losses), best_loss))

    linecolor = np.array([
        [255, 0, 0],
        [0, 0, 255]
    ])

    vis = visdom.Visdom()
    vis.line(Y=np.column_stack((epoch_losses, validation_losses)), opts=dict(linecolor=linecolor))

    return best_model


model = Model(input_size, output_size, hidden_dim, n_layers, drop_prob).to(device)
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model = train_model(model, optimizer, criterion, n_epoches)

generated, loss_sum = test_model(model, criterion, data, seq_length)

print(loss_sum)

range_gen = range(0, len(generated))
plt.plot(range_gen, generated, range_gen, data[seq_length:len(generated) + seq_length])
plt.show()