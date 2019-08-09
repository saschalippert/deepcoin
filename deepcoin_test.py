import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import math
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from datetime import timedelta, datetime
from collections import OrderedDict

np.random.seed(0)
torch.manual_seed(0)
scale = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, drop_prob):
        super(Model, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.noise_mean = 0.0
        self.noise_stddev = 0.1

        self.net = nn.LSTM(input_size, hidden_size, num_layers=n_layers, dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, add_noise=False):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)

        if (add_noise):
            noise = input.data.new(input.size()).normal_(self.noise_mean, self.noise_stddev)
            input = input + noise

        out, hidden = self.net(input, hidden)
        out = out[:, -1, :]  # only last sequence is evaluated
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        return (weight.new(self.n_layers, batch_size, self.hidden_size).normal_(-1, 1).to(device),
                weight.new(self.n_layers, batch_size, self.hidden_size).normal_(-1, 1).to(device))


seq_length = 64
input_size = 1
output_size = 1
hidden_dim = 128
n_layers = 2
batch_size = 512
n_epoches = 50
drop_prob = 0.5
lr = 0.001

time_sine = np.arange(0.001, 100, 0.01)
data_sine = np.sin(time_sine)


def create_dataloader(data, sequence_length):
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


def create_dataloaderBTC(data, sequence_length, train_percentage=0.7, eval_percentage=0.15):
    window_len = sequence_length + 1
    sequences = len(data) - window_len

    inputs = np.zeros((sequences, sequence_length), dtype=np.float32)
    targets = np.zeros((sequences), dtype=np.float32)

    for start in range(0, sequences):
        end = start + sequence_length

        inputs[start] = np.array(data[start:end])
        targets[start] = np.array(data[end])

    idx_eval = math.trunc(len(inputs) * train_percentage)
    idx_test = (math.trunc(len(inputs) * eval_percentage)) + idx_eval

    dataset_train = TensorDataset(torch.from_numpy(inputs[:idx_eval]), torch.from_numpy(targets[:idx_eval]))
    dataset_eval = TensorDataset(torch.from_numpy(inputs[idx_eval:idx_test]), torch.from_numpy(targets[idx_eval:idx_test]))
    dataset_test = TensorDataset(torch.from_numpy(inputs[idx_test:]), torch.from_numpy(targets[idx_test:]))

    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, drop_last=False)
    dataloader_eval = DataLoader(dataset_eval, shuffle=True, batch_size=len(dataset_eval), drop_last=False)
    dataloader_test = DataLoader(dataset_test, shuffle=False, batch_size=len(dataset_test), drop_last=False)

    return (dataloader_train, dataloader_eval, dataloader_test)


dataloader_sine_train = create_dataloader(data_sine, seq_length)


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


def test_modelBTC(model, data, seq_length):
    model.eval()

    predict_len = len(data) - seq_length

    gain = 0.0

    gen_seq = np.array(data[0:seq_length], dtype=np.float32)
    gen_out = np.zeros((predict_len), dtype=np.float32)

    for i in range(0, predict_len):
        hidden = model.init_hidden(1)

        gen_seq_torch = torch.tensor(gen_seq)

        input = gen_seq_torch.reshape((1, seq_length, 1)).to(device)

        out, _ = model(input, hidden)

        predicted_price = out.detach().cpu().numpy()

        gen_seq[0] = predicted_price
        gen_seq = np.roll(gen_seq, -1)

        gen_out[i] = predicted_price

        if i > 0 and i % seq_length == 0:
            data_future = data[i]
            data_past = data[i - seq_length]
            diff_prediction = abs(data_future - data_past)

            if (predicted_price > data_past):
                if data_future > data_past:
                    gain += diff_prediction
                else:
                    gain -= diff_prediction
            else:
                if data_future < data_past:
                    gain += diff_prediction
                else:
                    gain -= diff_prediction

            gen_seq = np.array(data[i - seq_length:i], dtype=np.float32)

    print("gain", gain * scale)

    return gen_out


def train_model(model, optimizer, criterion, n_epochs, dataloaders):
    best_loss = float("inf")
    best_model = None

    epoch_losses = []

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        model.train()

        train_losses = []

        for batch_i, (inputs, targets) in enumerate(dataloaders[0]):
            batch_size_train = inputs.size(0)

            hidden = model.init_hidden(batch_size_train)

            inputs = inputs.reshape((batch_size_train, seq_length, 1)).to(device)
            targets = targets.reshape((batch_size_train, 1)).to(device)

            optimizer.zero_grad()

            out, _ = model(inputs, hidden, add_noise=True)

            loss = criterion(out, targets)

            loss.backward()

            clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            train_loss = loss.detach().cpu().numpy()

            train_losses.append(train_loss)

        avg_train_loss = np.average(train_losses)

        epoch_losses.append(avg_train_loss)

        model.eval()

        eval_losses = []

        for batch_i, (inputs, targets) in enumerate(dataloaders[1]):
            batch_size_eval = inputs.size(0)
            hidden = model.init_hidden(batch_size_eval)

            inputs = inputs.reshape((batch_size_eval, seq_length, 1)).to(device)
            targets = targets.reshape((batch_size_eval, 1)).to(device)

            out, _ = model(inputs, hidden, add_noise=True)

            loss = criterion(out, targets)

            np_loss = loss.detach().cpu().numpy()

            eval_losses.append(np_loss)

        avg_eval_loss = np.average(eval_losses)

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_model = Model(input_size, output_size, hidden_dim, n_layers, drop_prob).to(device)
            best_model.load_state_dict(model.state_dict())

        print(f'Epoch: {epoch_i}/{n_epochs} TrainLoss: {avg_train_loss:.8f} EvalLoss: {avg_eval_loss:.8f} BstLoss: {best_loss:.8f}')

    return best_model


def start(dataloader_train, test_data, name):
    model = Model(input_size, output_size, hidden_dim, n_layers, drop_prob).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = train_model(model, optimizer, criterion, n_epoches, dataloader_train)

    torch.save(model.state_dict(), f'checkpoint_deepcoin_{name}.pth')

    # model.load_state_dict(torch.load(f'checkpoint_deepcoin_{name}.pth'))

    generated = test_model(model, test_data, seq_length)

    range_gen = range(0, len(generated))
    plt.plot(range_gen, generated, range_gen, test_data[seq_length:])
    plt.show()


def startBTC(dataloaders, test_data, name):
    model = Model(input_size, output_size, hidden_dim, n_layers, drop_prob).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = train_model(model, optimizer, criterion, n_epoches, dataloaders)

    torch.save(model.state_dict(), f'checkpoint_deepcoin_{name}.pth')

    #model.load_state_dict(torch.load(f'checkpoint_deepcoin_{name}.pth'))

    #idx_test = len(dataloaders[0].dataset) + len(dataloaders[1].dataset)
    idx_test = 0

    generated = test_modelBTC(model, test_data[idx_test:], seq_length)


    range_gen = range(0, len(generated))
    plt.plot(range_gen, generated, range_gen, test_data[idx_test + seq_length:])
    plt.show()


# start(dataloader_sine_train, data_sine, "sine")


def date_range(start: datetime, end: datetime, step: timedelta):
    date_list = []

    while start < end:
        date_list.append(start)
        start += step

    return date_list


start_date = datetime(2019, 6, 2)
end_date = datetime(2019, 6, 24)
candles = OrderedDict()

for load_candles in date_range(start_date, end_date, timedelta(days=1)):
    filename = load_candles.strftime('btceur/btceur_%Y_%m_%d.json')

    with open(filename) as json_file:
        data_candles = json.load(json_file)
        candles.update(data_candles)

data_candles = np.zeros((len(candles)))

for index, key in enumerate(candles):
    candle = candles[key]

    time = candle["time"]
    low = candle["low"]
    high = candle["high"]
    open = candle["open"]
    close = candle["close"]
    volume = candle["volume"]

    data_candles[index] = close

data_max = np.max(data_candles)
data_min = np.min(data_candles)
scale = data_max - data_min

data_candles = (data_candles - data_min) / scale


plt.hist(data_candles, 50, normed=1, facecolor='green', alpha=0.75)
plt.show()

dataloaders = create_dataloaderBTC(data_candles, seq_length)

startBTC(dataloaders, data_candles, "candles")

print("end")
