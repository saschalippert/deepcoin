from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from bookkeeper import Bookkeeper
from deepcoin_sine import test_model_sine

import deepcoin_norm as norm
import deepcoin_candles as candles
import deepcoin_dataloader as dataloader
from deepcoin_model import Model

np.random.seed(0)
torch.manual_seed(0)
scale = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hp_seq_length = 64
hp_input_size = 1
hp_output_size = 1
hp_hidden_dim = 128
hp_n_layers = 2
hp_batch_size = 512
hp_n_episodes = 100
hp_drop_prob = 0.5
hp_lr = 0.001

time_sine = np.arange(0.001, 100, 0.01)
data_sine = np.sin(time_sine)

start_date = datetime(2018, 3, 2)
end_date = datetime(2018, 4, 24)
data_candles = candles.load_candles(".", "btceur", start_date, end_date)
data_candles = data_candles[['close']].to_numpy().flatten()
data_candles = norm.norm_min_max(data_candles)

plt.hist(data_candles, 50, facecolor='green', alpha=0.75)
plt.show()

dataloader_sine = dataloader.create_dataloader_train(data_sine, hp_seq_length, hp_batch_size)
dataloaders_btc = dataloader.create_dataloader_full(data_candles, hp_seq_length, hp_batch_size)

hyperparameters = {k: v for k, v in globals().items() if k.startswith("hp_")}
bookkeeper = Bookkeeper("deepcoin")

def test_model_btc(model, data, seq_length, device):
    model.eval()

    predict_len = len(data) - seq_length

    gain = 0.0
    total_loss = 0

    criterion = nn.MSELoss()

    gen_seq = np.array(data[0:seq_length], dtype=np.float32)
    gen_out = np.zeros((predict_len), dtype=np.float32)

    gains = []
    gen_pos = np.zeros((predict_len), dtype=np.float32)

    for i in range(0, predict_len):
        hidden = model.init_hidden(1)

        gen_seq_torch = torch.tensor(gen_seq)

        input = gen_seq_torch.reshape((1, seq_length, 1)).to(device)

        target = torch.tensor(data[seq_length + i])
        target = target.reshape((1, 1)).to(device)

        out, _ = model(input, hidden)

        loss = criterion(out, target)

        total_loss += loss.detach().cpu().numpy()

        predicted_price = out.detach().cpu().numpy()

        gen_seq[0] = predicted_price
        gen_seq = np.roll(gen_seq, -1)

        gen_out[i] = predicted_price

        if i > 0 and i % seq_length == 0:
            data_future = data[i]
            data_past = data[i - seq_length]
            diff_prediction = abs(data_future - data_past)
            fee = 0.0026

            if (predicted_price > data_past):
                gen_pos[i - seq_length:i] = 1

                if data_future > data_past:
                    gain += diff_prediction
                else:
                    gain -= diff_prediction
            else:
                gen_pos[i - seq_length:i] = -1

                if data_future < data_past:
                    gain += diff_prediction
                else:
                    gain -= diff_prediction

            gain -= fee

            gains.append(gain)

            gen_seq = np.array(data[i - seq_length:i], dtype=np.float32)

    return gen_out, total_loss, gains, gen_pos


def train_model(model, optimizer, criterion, n_epochs, bookkeeper, dataloaders, data_name, hyperparameters):
    bookkeeper.train_start(n_epochs, data_name, hyperparameters)

    for epoch_i in range(1, n_epochs + 1):
        model.train()

        train_loss = 0

        for batch_i, (inputs, targets) in enumerate(dataloaders[0]):
            batch_size_train = inputs.size(0)

            hidden = model.init_hidden(batch_size_train)

            inputs = inputs.reshape((batch_size_train, hp_seq_length, 1)).to(device)
            targets = targets.reshape((batch_size_train, 1)).to(device)

            optimizer.zero_grad()

            out, _ = model(inputs, hidden, add_noise=True)

            loss = criterion(out, targets)

            loss.backward()

            clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            train_loss += loss.detach().cpu().numpy()

        train_loss /= len(dataloaders[0])

        model.eval()

        eval_loss = 0

        if (len(dataloaders) > 1):
            for batch_i, (inputs, targets) in enumerate(dataloaders[1]):
                batch_size_eval = inputs.size(0)
                hidden = model.init_hidden(batch_size_eval)

                inputs = inputs.reshape((batch_size_eval, hp_seq_length, 1)).to(device)
                targets = targets.reshape((batch_size_eval, 1)).to(device)

                out, _ = model(inputs, hidden, add_noise=True)

                loss = criterion(out, targets)

                eval_loss += loss.detach().cpu().numpy()

            eval_loss /= len(dataloaders[1])
        else:
            eval_loss = train_loss

        model_copy = lambda: Model(hp_input_size, hp_output_size, hp_hidden_dim, hp_n_layers, hp_drop_prob).to(device)
        bookkeeper.train_step(train_loss, eval_loss, model, model_copy)

    return bookkeeper.eval_best_model()


def create_and_train_model(bookkeeper, dataloaders, name, n_episodes, hyperparameters):
    model = Model(hp_input_size, hp_output_size, hp_hidden_dim, hp_n_layers, hp_drop_prob).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp_lr)

    #model = train_model(model, optimizer, criterion, n_episodes, bookkeeper, dataloaders, name, hyperparameters)

    #torch.save(model.state_dict(), f'checkpoint_deepcoin_{name}.pth')

    model.load_state_dict(torch.load(f'checkpoint_deepcoin_{name}.pth'))

    return model

def plot_figure(original, generated):
    range_gen = range(0, len(generated))

    fig = plt.figure()
    plt.plot(range_gen, generated, range_gen, original[hp_seq_length:hp_seq_length + len(generated)])

    return fig

def plot_figure2(data):
    range_gen = range(0, len(data))

    fig = plt.figure()
    plt.plot(range_gen, data)

    return fig

model_sine = create_and_train_model(bookkeeper, [dataloader_sine], "sine", 1, hyperparameters)
sine_test_generated, sine_test_loss = test_model_sine(model_sine, data_sine, hp_seq_length, device)
bookkeeper.add_figure(lambda : plot_figure(data_sine, sine_test_generated), "sine/test")
bookkeeper.add_text("sine/test/loss", str(sine_test_loss))

model_btc = create_and_train_model(bookkeeper, dataloaders_btc, "btc", hp_n_episodes, hyperparameters)
btc_test_generated, btc_test_loss, btc_test_gains, btc_test_pos = test_model_btc(model_btc, data_candles, hp_seq_length, device)
bookkeeper.add_figure(lambda : plot_figure(data_candles, btc_test_generated), "btc/test")
bookkeeper.add_figure(lambda : plot_figure2(btc_test_gains), "btc/test/gains")
bookkeeper.add_figure(lambda : plot_figure2(btc_test_pos), "btc/test/pos")
bookkeeper.add_text("btc/test/loss", str(btc_test_loss))



print("end")
