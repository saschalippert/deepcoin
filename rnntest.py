import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict, deque

plt.figure(figsize=(8,5))

# how many time steps/data pts are in one batch of data
seq_length = 20

# generate evenly spaced data pts
#time_steps = np.linspace(0, np.pi * 2, seq_length + 1)
#data = np.sin(time_steps)

time_steps = np.arange(0, 10, 0.01)
data = np.sin(time_steps)

x = data[:-1] # all but the last piece of data
y = data[1:] # all but the first

# display the data
plt.plot(time_steps[1:], x, 'r.', label='input, x') # x
plt.plot(time_steps[1:], y, 'b.', label='target, y') # y

plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.output_size = output_size

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)

        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out[:,-1,:] #only last sequence is evaluated

        # get final output
        output = self.fc(r_out)

        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim)

# decide on hyperparameters
input_size=1
output_size=1
hidden_dim=32
n_layers=1

# instantiate an RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

# MSE loss and Adam optimizer with a learning rate of 0.01
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

batch_size = 32
train_loader = batch_data(data, seq_length, batch_size)
data_iter = iter(train_loader)

# train the RNN
def train(rnn, n_steps, print_every):
    # initialize the hidden state


    for epoch_i in range(1, n_steps + 1):

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            hidden = rnn.init_hidden(batch_size)
            n_batches = len(train_loader.dataset) // batch_size

            if (batch_i > n_batches):
                break

            inputs = inputs.reshape((batch_size, seq_length, 1))  # input_size=1
            labels = labels.reshape((batch_size, 1))

            # outputs from the rnn
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

# train the rnn and monitor results
n_steps = 10
print_every = 15

trained_rnn = train(rnn, n_steps, print_every)


def generate(rnn, current_seq, predict_len=1000):
    rnn.eval()

    hidden = rnn.init_hidden(1)

    gen_seq = deque(current_seq, maxlen = seq_length)
    gen_out = deque(maxlen = predict_len)

    for i in range(predict_len):
        gen_seq_torch = torch.from_numpy(np.array(gen_seq, dtype=np.float32))
        inputs = gen_seq_torch.reshape((1, seq_length, 1))

        output, hidden = rnn(inputs, hidden)

        hidden = hidden.data

        np_out = output.data.numpy()[0]

        gen_out.append(np_out)
        gen_seq.append(np_out)

    return gen_out

n_steps = 100
print_every = 5

trained_rnn = train(rnn, n_steps, print_every)

generated = generate(trained_rnn, data[0:20])

plt.plot(time_steps, data, time_steps, generated)
plt.show()