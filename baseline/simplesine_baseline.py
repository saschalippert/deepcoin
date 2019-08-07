import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

plt.switch_backend('agg')

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, drop_prob):
        super(Model, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.noise_mean = 0.0
        self.noise_stddev = 0.1

        self.net = nn.LSTM(input_size, hidden_size, num_layers = n_layers, dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, add_noise = False):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)

        if(add_noise):
            noise = input.data.new(input.size()).normal_(self.noise_mean, self.noise_stddev)
            input = input + noise

        out, hidden = self.net(input, hidden)
        out = out[:,-1,:] #only last sequence is evaluated
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        return (weight.new(self.n_layers, batch_size, self.hidden_size).normal_(-1,1).to(device),
                weight.new(self.n_layers, batch_size, self.hidden_size).normal_(-1,1).to(device))

seq_length = 64
input_size = 1
output_size = 1
hidden_dim = 128
n_layers = 2
batch_size = 512
n_epoches = 1000
drop_prob = 0.5
lr = 0.001

time = np.arange(0.001, 100, 0.01);
data  = np.sin(time)

def create_dataloader(data, sequence_length, batch_size):
    window_len = sequence_length + 1
    sequences = len(data) - window_len

    inputs = np.zeros((sequences, sequence_length), dtype=np.float32)
    targets = np.zeros((sequences), dtype=np.float32)

    for start in range(0, sequences):
        end = start + sequence_length

        inputs[start] = np.array(data[start:end])
        targets[start] = np.array(data[end])

    dataset_train = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, drop_last=True)

    return dataloader_train

dataloader_train = create_dataloader(data, seq_length, batch_size)

def test_model(model, data, seq_length):
    model.eval()

    predict_len = len(data) - seq_length

    gen_seq = np.array(data[0:seq_length], dtype=np.float32)
    gen_out = np.zeros((predict_len), dtype=np.float32)

    for i in range(0, predict_len):
        hidden = model.init_hidden(1)

        gen_seq_torch = torch.tensor(gen_seq)

        input = gen_seq_torch.reshape((1, seq_length, 1)).to(device)

        out, _ = model(input, hidden)

        np_out = out.detach().cpu().numpy()

        gen_out[i] = np_out

        gen_seq[0] = np_out
        gen_seq = np.roll(gen_seq, -1)

    return gen_out

writer = SummaryWriter(log_dir="/home/sascha/logs/simplesine9")

def train_model(model, optimizer, criterion, n_epochs):
    best_loss = float("inf")
    best_model = None

    epoch_losses = []

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        model.train()

        epoch_loss = 0

        for batch_i, (inputs, targets) in enumerate(dataloader_train):
            hidden = model.init_hidden(batch_size)

            inputs = inputs.reshape((batch_size, seq_length, 1)).to(device)
            targets = targets.reshape((batch_size, 1)).to(device)

            optimizer.zero_grad()

            out, _ = model(inputs, hidden, add_noise = True)

            loss = criterion(out, targets)

            loss.backward()

            clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            np_loss = loss.detach().cpu().numpy()

            epoch_loss += np_loss

        writer.add_scalar('data/Loss', epoch_loss, epoch_i)

        epoch_losses.append(epoch_loss)

        if(epoch_loss < best_loss):
            best_loss = epoch_loss
            best_model = Model(input_size, output_size, hidden_dim, n_layers, drop_prob).to(device)
            best_model.load_state_dict(model.state_dict())

        print('Epoch: {:>4}/{:<4} Loss: {:.10f} AvgLoss: {:.10f} BstLoss: {:.10f}'.format(epoch_i, n_epochs, epoch_loss, np.average(epoch_losses), best_loss))

    writer.close()

    return best_model

model = Model(input_size, output_size, hidden_dim, n_layers, drop_prob).to(device)
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model = train_model(model, optimizer, criterion, n_epoches)

#torch.save(model.state_dict(), 'checkpoint_simplesine.pth')

#model.load_state_dict(torch.load('checkpoint_simplesine.pth'))

generated = test_model(model, data, seq_length)

range_gen = range(0, len(generated))

fig = plt.figure()

plt.plot(range_gen, generated, range_gen, data[seq_length:len(generated) + seq_length])
writer.add_figure('matplotlib/figure', fig)

print("end")
