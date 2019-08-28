from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from deepcoin_logger import Logger

from deepcoin_norm import Normalizer_Noop, Normalizer_Min_Max
import deepcoin_candles as candles
import deepcoin_dataloader as dataloader
from deepcoin_order import Order, Accountant
from deepcoin_model import Model
from deepcoin_transformer import Transformer_Return, Transformer_Noop
from tqdm import tqdm

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hp_seq_length = 24
hp_input_size = 1
hp_output_size = 1
hp_hidden_dim = 128
hp_n_layers = 2
hp_batch_size = 512
hp_n_episodes = 300
hp_drop_prob = 0.5
hp_lr = 0.0001
hp_n_future = 1

time_sine = np.arange(0.001, 100, 0.01)
data_sine = np.sin(time_sine)

normalizer = Normalizer_Noop()
transformer = Transformer_Return()

#normalizer = Normalizer_Min_Max()
#transformer = Transformer_Noop()

start_date = datetime(2018, 3, 2)
end_date = datetime(2018, 4, 24)

data_candles = candles.load_candles(".", "btceur1h", start_date, end_date)
data_candles = data_candles[['close']].to_numpy().flatten()

data_input = transformer.transform(data_candles)
data_comp = transformer.revert_list(data_candles[0], data_input)

data_input = normalizer.normalize(data_input)

plt.hist(data_input, 50, facecolor='green', alpha=0.75)
plt.show()

dataloader_sine = dataloader.create_dataloader_train(data_sine, hp_seq_length, hp_batch_size)
dataloaders_btc = dataloader.create_dataloader_full(data_input, hp_seq_length, hp_batch_size)

hyperparameters = {k: v for k, v in globals().items() if k.startswith("hp_")}
logger = Logger("deepcoin")


def predict_price(model, history, n_future, device):
    model.eval()

    gen_seq = np.array(history, dtype=np.float32)
    gen_out = np.zeros(n_future, dtype=np.float32)

    for i in range(0, n_future):
        hidden = model.init_hidden_zero(1)

        gen_seq_torch = torch.tensor(gen_seq)
        input = gen_seq_torch.reshape((1, len(history), 1)).to(device)

        out, _ = model(input, hidden)

        predicted_price = out.detach().cpu().numpy().flatten()[0]

        gen_seq[0] = predicted_price
        gen_seq = np.roll(gen_seq, -1)

        gen_out[i] = predicted_price

    return gen_out


def test_model_btc(model, data, seq_length, normalizer, device, data_start, transformer, hp_n_future):
    accountant = Accountant()

    order = None

    current_price = data_start

    for i in range(0, seq_length - 1):
        current_price = transformer.revert_single(current_price, normalizer.denormalize(data[i]))

    for i in tqdm(range(seq_length - 1, len(data))):
        current_price = transformer.revert_single(current_price, normalizer.denormalize(data[i]))
        seq_end_idx = i + 1
        history = data[seq_end_idx - seq_length: seq_end_idx]
        future = predict_price(model, history, hp_n_future, device)

        predicted_price = transformer.revert_list(current_price, normalizer.denormalize(future))[-1]

        is_long = predicted_price > current_price

        if (not order):
            order = Order(current_price, is_long)
        elif (is_long != order._long):
            accountant.close(order, current_price, 0.0026)
            order = Order(current_price, is_long)

    if (order):
        accountant.close(order, current_price, 0.0026)

    print("count", accountant._count)
    print("profits", accountant._profits)
    print("fees", accountant._fees)
    print("gain", accountant._gain)
    print("gain total", sum(accountant._gain))


def train_model(model, optimizer, criterion, n_epochs, logger, dataloaders, data_name, hyperparameters):
    logger.train_start(n_epochs, data_name, hyperparameters)

    for epoch_i in range(1, n_epochs + 1):
        model.train()

        train_loss = 0

        for batch_i, (inputs, targets) in enumerate(dataloaders[0]):
            batch_size_train = inputs.size(0)

            hidden = model.init_hidden_zero(batch_size_train)

            inputs = inputs.reshape((batch_size_train, hp_seq_length, 1)).to(device)
            targets = targets.reshape((batch_size_train, 1)).to(device)

            optimizer.zero_grad()

            out, _ = model(inputs, hidden)

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

                hidden = model.init_hidden_zero(batch_size_eval)

                inputs = inputs.reshape((batch_size_eval, hp_seq_length, 1)).to(device)
                targets = targets.reshape((batch_size_eval, 1)).to(device)

                out, _ = model(inputs, hidden)

                loss = criterion(out, targets)

                eval_loss += loss.detach().cpu().numpy()

            eval_loss /= len(dataloaders[1])
        else:
            eval_loss = train_loss

        model_copy = lambda: Model(hp_input_size, hp_output_size, hp_hidden_dim, hp_n_layers, hp_drop_prob).to(device)
        logger.train_step(train_loss, eval_loss, model, model_copy)

    return logger.eval_best_model()


def create_and_train_model(logger, dataloaders, name, n_episodes, hyperparameters):
    model = Model(hp_input_size, hp_output_size, hp_hidden_dim, hp_n_layers, hp_drop_prob).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp_lr)

    model = train_model(model, optimizer, criterion, n_episodes, logger, dataloaders, name, hyperparameters)

    torch.save(model.state_dict(), f'checkpoint_deepcoin_{name}.pth')

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


# model_sine = create_and_train_model(logger, [dataloader_sine], "sine", 1, hyperparameters)
# sine_test_generated, sine_test_loss = test_model_sine(model_sine, data_sine, hp_seq_length, device)
# logger.add_figure(lambda : plot_figure(data_sine, sine_test_generated), "sine/test")
# logger.add_text("sine/test/loss", str(sine_test_loss))

model_btc = create_and_train_model(logger, dataloaders_btc, "btc", hp_n_episodes, hyperparameters)
test_model_btc(model_btc,
               data_input,
               hp_seq_length,
               normalizer, device,
               data_candles[0], transformer, hp_n_future)
# logger.add_figure(lambda: plot_figure(normalizer.denormalize(data_candles), btc_test_generated), "btc/test")
# logger.add_figure(lambda: plot_figure2(btc_test_gains), "btc/test/gains")
# logger.add_figure(lambda: plot_figure2(btc_test_pos), "btc/test/pos")

# print("profits", btc_test_profits)
# print("fees", btc_test_fees)

print("end")
