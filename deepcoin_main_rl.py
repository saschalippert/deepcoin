from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from deepcoin_norm import Normalizer_Noop, Normalizer_Min_Max, Normalizer_ClipStdDev, Normalizer_Min_Max_Target
import deepcoin_candles as candles
import deepcoin_dataloader as dataloader
from deepcoin_order import Order, Accountant
from deepcoin_model import Model
from deepcoin_transformer import Transformer_SimpleReturn, Transformer_Noop, Transformer_LogReturn

from torch.distributions import Categorical

hp_seq_length = 3
hp_chart = "btceur1h"
hp_lr = 1e-3

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

normalizer = Normalizer_Noop()
transformer = Transformer_SimpleReturn()

train_start_date = datetime(2018, 1, 2)
train_end_date = datetime(2018, 12, 24)

time_sine = np.arange(0.1, 25, 0.1)
data_sine = np.sin(time_sine)

train_data_candles = candles.load_candles(".", hp_chart, train_start_date, train_end_date)
train_data_candles = train_data_candles[['close']].to_numpy().flatten()

train_data_input = transformer.transform(data_sine)
train_data_input = normalizer.normalize(train_data_input)

class Policy(nn.Module):
    def __init__(self, s_size=hp_seq_length, h1_size=8, h2_size=4, a_size=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, a_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

def reinforce(data, seq_length, data_start, lr, n_episodes=2000, max_t=1000, gamma=1, print_every=1):
    print("reinforce", lr)

    policy = Policy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        actions = []

        accountant = Accountant()

        order = None

        current_price = data_start

        for i in range(0, seq_length - 1):
            current_price = transformer.revert_single(current_price, normalizer.denormalize(data[i]))

        multi = 0

        for i in range(seq_length - 1, len(data)):
            current_price = transformer.revert_single(current_price, normalizer.denormalize(data[i]))
            seq_end_idx = i + 1
            state = data[seq_end_idx - seq_length: seq_end_idx]

            action, log_prob = policy.act(state)

            is_long = action

            reward = 0

            if (not order):
                order = Order(current_price, is_long)
                multi = 1
            elif (is_long != order._long):
                gain = accountant.close(order, current_price, 0.002)
                order = Order(current_price, is_long)

                if gain > 0:
                    reward = 1
                else:
                    reward = 0

                for m in range(0, multi):
                    rewards.append(reward)

                multi = 1
            elif (is_long == order._long):
                multi = multi + 1

            saved_log_probs.append(log_prob)
            actions.append(action)

        if order:
            gain = accountant.close(order, current_price, 0.002)

            if gain > 0:
                reward = 1
            else:
                reward = 0

            for m in range(0, multi):
                rewards.append(reward)

        cnt = 0
        for (r, a) in zip(rewards, actions):
            if r != a:
                cnt = cnt + 1

        spacing = 20
        print("gain".ljust(spacing), sum(accountant._gain), accountant._gain)

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm(policy.parameters(), 5)

        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores

scores = reinforce(train_data_input, hp_seq_length, data_sine[0], hp_lr)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

#test