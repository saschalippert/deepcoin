from collections import deque
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import math
from torch.nn.utils import clip_grad_norm_

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, drop_prob):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.net = nn.LSTM(input_size, hidden_size, num_layers = n_layers, dropout=drop_prob, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)

        hidden = rnn.init_hidden(batch_size)

        out, hidden = self.net(input, hidden)

        out = out[:,-1,:] #only last sequence is evaluated

        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # return torch.zeros(self.n_layers, batch_size, self.hidden_size)

        weight = next(self.parameters()).data

        return (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device),
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device))

        # return torch.zeros(self.n_layers, batch_size, self.hidden_size)

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

rnn = RNN(input_size, output_size, hidden_dim, n_layers, drop_prob).to(device)
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

    best_loss = float("inf")
    best_model = None

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        epoch_losses = []

        #gen_out = deque(maxlen=len(dataloader_train) * batch_size)

        for batch_i, (inputs, targets) in enumerate(dataloader_train):
            inputs = inputs.reshape((batch_size, seq_length, 1)).to(device)
            targets = targets.reshape((batch_size, 1)).to(device)

            optimizer.zero_grad()

            prediction, hidden = rnn(inputs)

            loss = criterion(prediction, targets)

            loss.backward()

            clip_grad_norm_(rnn.parameters(), 5)

            optimizer.step()

            np_loss = loss.detach().cpu().numpy()

            epoch_losses.append(np_loss)

            if(np_loss < best_loss):
                best_loss = np_loss
                best_model = RNN(input_size, output_size, hidden_dim, n_layers, drop_prob).to(device)
                best_model.load_state_dict(rnn.state_dict())


            #np_out = prediction.detach().numpy().flatten()

            #for out in enumerate(np_out):
             #   gen_out.append(out[1])

        #plt.plot(gen_out)
        #plt.show()

        print('Epoch: {:>4}/{:<4} Loss: {:.10f} AvgLoss: {:.10f} BstLoss: {:.10f}'.format(epoch_i, n_epochs, np_loss, np.average(epoch_losses), best_loss))

    return best_model


rnn = train(rnn, n_epoches)

def generate2(rnn, current_seq, predict_len=10000):
    rnn.eval()

    gen_seq = np.array(current_seq, dtype=np.float32)
    gen_out = np.zeros((predict_len), dtype=np.float32)

    for i in range(predict_len):
        gen_seq_torch = torch.from_numpy(gen_seq)
        inputs = gen_seq_torch.reshape((1, seq_length, 1)).to(device)

        output, hidden = rnn(inputs)

        np_out = output.detach().cpu().numpy()

        gen_out[i] = np_out

        gen_seq[0] = np_out
        gen_seq = np.roll(gen_seq, -1)

    return gen_out

generated = generate2(rnn, data[0:seq_length])

range_gen = range(0, len(generated))
plt.plot(range_gen, generated, range_gen, generated, data[0:len(generated)])
plt.show()

gen_out = np.zeros((len(dataloader_train) * batch_size), dtype=np.float32)

rnn.eval()

for batch_i, (inputs, targets) in enumerate(dataloader_train):
    inputs = inputs.reshape((batch_size, seq_length, 1)).to(device)

    prediction, hidden = rnn(inputs)

    np_out = prediction.detach().cpu().numpy().flatten()

    start = batch_i * batch_size
    end = start + batch_size

    gen_out[start:end] = np_out

range_gen = range(0, len(gen_out))
plt.plot(range_gen, gen_out, range_gen, data[0:len(gen_out)])
plt.show()

print(list(generated)[0:seq_length])
print(list(gen_out)[0:seq_length])
print(list(data)[seq_length:seq_length + seq_length])