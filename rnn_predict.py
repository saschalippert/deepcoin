import json
from datetime import timedelta, datetime
from collections import OrderedDict
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as utils

start_date = datetime(2016, 1, 2)
end_date = datetime(2019, 6, 24)


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_size = output_size

        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)

        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.contiguous().view(-1, self.hidden_dim)

        # get final output
        out = self.fc(r_out)

        out = out.view(batch_size, -1, self.output_size)
        out = out[:, -1]

        return out, hidden

def date_range(start: datetime, end: datetime, step: timedelta):
    date_list = []

    while start < end:
        date_list.append(start)
        start += step

    return date_list

candles = OrderedDict()

for chunk_date in date_range(start_date, end_date, timedelta(days=1)):
    filename = chunk_date.strftime('btceur/btceur_%Y_%m_%d.json')

    with open(filename) as json_file:
        data = json.load(json_file)
        candles.update(data)

data = np.zeros((len(candles)))

for index, key in enumerate(candles):

    candle = candles[key]

    time = candle["time"]
    low = candle["low"]
    high = candle["high"]
    open = candle["open"]
    close = candle["close"]
    volume = candle["volume"]

    data[index] = close

seq_length = 20
input_size=1
output_size=1
hidden_dim=32
n_layers=1

time = np.arange(0, 10, 0.01);
data   = np.sin(time)

# instantiate an RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

from torch.utils.data import TensorDataset, DataLoader


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """

    window_len = sequence_length + 1
    sequences = len(words) - window_len

    train_x = np.zeros((sequences, sequence_length), dtype=np.float32)
    train_y = np.zeros((sequences), dtype=np.float32)

    for start in range(0, sequences):
        end = start + sequence_length

        train_x[start] = np.array(words[start:end])
        train_y[start] = np.array(words[end])

    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    # return a dataloader
    return dataloader

# there is no test for this function, but you are encouraged to create
# print statements and tests of your own
batch_size = 16
train_loader = batch_data(data, seq_length, batch_size)
data_iter = iter(train_loader)
sample_x, sample_y = data_iter.next()

print(sample_x.shape)
print(sample_x)
print()
print(sample_y.shape)

def train(rnn, n_epochs, print_every):
    # initialize the hidden state
    hidden = None

    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            # outputs from the rnn

            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset) // batch_size

            if (batch_i > n_batches):
                break

            inputs = inputs.reshape((batch_size, seq_length, 1)) # input_size=1
            labels = labels.reshape((batch_size, 1))

            prediction, hidden = rnn(inputs, hidden)

            ## Representing Memory ##
            # make a new variable for hidden and detach the hidden state from its history
            # this way, we don't backpropagate through the entire history
            hidden = hidden.data

            # calculate the loss
            loss = criterion(prediction, labels)
            # zero gradients
            optimizer.zero_grad()
            # perform backprop and update weights
            loss.backward()
            optimizer.step()

            # display loss and predictions
            if batch_i % print_every == 0:
                print('Loss: ', loss.item())
                #plt.plot(time_steps[1:], x, 'r.')  # input
                #plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.')  # predictions
                #plt.show()

    return rnn

n_steps = 100
print_every = 5

trained_rnn = train(rnn, n_steps, print_every)