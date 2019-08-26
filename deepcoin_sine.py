import numpy as np
import torch
from torch import nn


def test_model_sine(model, sine_data, seq_length, device):
    model.eval()

    criterion = nn.MSELoss()

    predict_len = len(sine_data) - seq_length

    gen_seq = np.array(sine_data[0:seq_length], dtype=np.float32)
    gen_out = np.zeros((predict_len), dtype=np.float32)

    total_loss = 0

    for i in range(0, predict_len):
        hidden = model.init_hidden(1)

        input = torch.tensor(gen_seq)
        input = input.reshape((1, seq_length, 1)).to(device)

        target = torch.tensor(sine_data[seq_length + i])
        target = target.reshape((1, 1)).to(device)

        out, _ = model(input, hidden)
        loss = criterion(out, target)

        np_out = out.detach().cpu().numpy()
        np_loss = loss.detach().cpu().numpy()

        gen_out[i] = np_out

        gen_seq[0] = np_out
        gen_seq = np.roll(gen_seq, -1)

        total_loss += np_loss

    return gen_out, total_loss