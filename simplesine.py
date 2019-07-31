from collections import deque
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import math

np.set_printoptions(precision=10)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers = n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        out, hidden = self.rnn(input, hidden)

        out = out[:,-1,:] #only last sequence is evaluated

        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        #return torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device))

        #return torch.zeros(self.n_layers, batch_size, self.hidden_size)

seq_length = 64
input_size = 1
output_size = 1
hidden_dim = 128
n_layers = 2
batch_size = 512
n_epoches = 75

time = np.arange(0.001, 100, 0.01);
data  = np.sin(time)

rnn = RNN(input_size, output_size, hidden_dim, n_layers).to(device)
print(rnn)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)


def batch_data(data, sequence_length, batch_size, train_percentage = 0.75):
    window_len = sequence_length + 1
    sequences = len(data) - window_len

    inputs = np.zeros((sequences, sequence_length), dtype=np.float32)
    targets = np.zeros((sequences), dtype=np.float32)

    for start in range(0, sequences):
        end = start + sequence_length

        inputs[start] = np.array(data[start:end])
        targets[start] = np.array(data[end])

    i_train = math.trunc(len(inputs) * train_percentage)

    dataset_train = TensorDataset(torch.from_numpy(inputs[:i_train]), torch.from_numpy(targets[:i_train]))
    dataset_test = TensorDataset(torch.from_numpy(inputs[i_train:]), torch.from_numpy(targets[i_train:]))

    dataloader_train = DataLoader(dataset_train, shuffle=False, batch_size=batch_size, drop_last=True)
    dataloader_test = DataLoader(dataset_test, shuffle=False, batch_size=batch_size, drop_last=True)

    return dataloader_train, dataloader_test

dataloader_train, dataloader_test = batch_data(data, seq_length, batch_size)

def train(rnn, n_epochs):
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        epoch_losses = []

        #gen_out = deque(maxlen=len(dataloader_train) * batch_size)

        for batch_i, (inputs, targets) in enumerate(dataloader_train):
            hidden = rnn.init_hidden(batch_size)

            inputs = inputs.reshape((batch_size, seq_length, 1)).to(device)
            targets = targets.reshape((batch_size, 1)).to(device)

            prediction, hidden = rnn(inputs, hidden)

            loss = criterion(prediction, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.detach().cpu().numpy())

            #np_out = prediction.detach().numpy().flatten()

            #for out in enumerate(np_out):
             #   gen_out.append(out[1])

        #plt.plot(gen_out)
        #plt.show()

        print('Epoch: {:>4}/{:<4}  Loss: {}'.format(epoch_i, n_epochs, np.average(epoch_losses)))

    return rnn


trained_rnn = train(rnn, n_epoches)

def generate2(rnn, current_seq, predict_len=10000):
    rnn.eval()

    gen_seq = np.array(current_seq, dtype=np.float32)
    gen_out = np.zeros((predict_len), dtype=np.float32)

    for i in range(predict_len):
        hidden = rnn.init_hidden(1)

        gen_seq_torch = torch.from_numpy(gen_seq)
        inputs = gen_seq_torch.reshape((1, seq_length, 1)).to(device)

        output, hidden = rnn(inputs, hidden)

        np_out = output.detach().cpu().numpy()
        #print(np_out)

        gen_out[i] = np_out

        gen_seq[0] = np_out
        gen_seq = np.roll(gen_seq, -1)

    return gen_out

generated = generate2(trained_rnn, data[0:seq_length])

range_gen = range(0, len(generated))
plt.plot(range_gen, generated, range_gen, generated, data[0:len(generated)])
plt.show()

gen_out = np.zeros((len(dataloader_train) * batch_size), dtype=np.float32)

rnn.eval()
for batch_i, (inputs, targets) in enumerate(dataloader_train):
    hidden = rnn.init_hidden(batch_size)

    inputs = inputs.reshape((batch_size, seq_length, 1)).to(device)

    prediction, hidden = rnn(inputs, hidden)

    np_out = prediction.detach().cpu().numpy().flatten()
    #print(np_out)

    start = batch_i * batch_size
    end = start + batch_size

    gen_out[start:end] = np_out

range_gen = range(0, len(gen_out))
plt.plot(range_gen, gen_out, range_gen, data[0:len(gen_out)])
plt.show()

print(list(generated)[0:seq_length])
print(list(gen_out)[0:seq_length])
print(list(data)[seq_length:seq_length + seq_length])