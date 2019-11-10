from random import random

import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class TraderEnv(gym.Env):

    def __init__(self, data, past, train):
        super(TraderEnv, self).__init__()
        self.past = past
        self.idx_time = self.past - 1
        self.reward_sum = 0
        self.data = data
        self.data_features = data.shape[1]
        self.last_action = None
        self.train = train

        self.reward_range = (-1, 1)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.past, self.data_features))
        self.action_space = spaces.Discrete(2)

    def _build_state(self):
        idx_start = self.idx_time - self.past + 1
        idx_end = self.idx_time + 1

        state = np.array(self.data[idx_start : idx_end])

        return state.reshape(self.past, self.data_features)

    def reset(self):
        self.idx_time = self.past - 1
        self.reward_sum = 0
        self.last_action = None

        return self._build_state()

    def step(self, action):
        self.idx_time = self.idx_time + 1
        reward = 0

        if action:
            new_value = self.data[self.idx_time][0]

            roll = random()

            if roll > 0.5:
                new_value *= -1

            reward = new_value

            reward = max(reward, -0.05)
            reward = min(reward, 0.10)

            reward - 0.0016

            #if action != self.last_action:
                #reward -= 0.0016

        self.reward_sum = self.reward_sum + reward
        self.last_action = action

        reward = np.clip(reward, -1, 1)

        state = self._build_state()
        done = self.idx_time >= len(self.data) - 1

        return state, reward, done, {}

    def render(self):
        print(self.idx_time, self.reward_sum)

from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy, CnnPolicy, MlpLstmPolicy
#from stable_baselines.deepq import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, DQN, A2C, TRPO

from deepcoin_norm import Normalizer_Noop, Normalizer_Min_Max, Normalizer_ClipStdDev, Normalizer_Min_Max_Target
import deepcoin_candles as candles
import deepcoin_dataloader as dataloader
from deepcoin_order import Order, Accountant
from deepcoin_model import Model
from deepcoin_transformer import Transformer_SimpleReturn, Transformer_Noop, Transformer_LogReturn
from datetime import datetime

hp_chart = "btceur1h"
train_start_date = datetime(2017, 1, 2)
train_end_date = datetime(2018, 6, 24)

eval_start_date = datetime(2018, 7, 1)
eval_end_date = datetime(2018, 12, 24)

test_start_date = datetime(2019, 1, 2)
test_end_date = datetime(2019, 11, 2)

def change(data, column):
    data[column] = (data[column] / data[column].shift(1)) - 1

def preprocess(data):
    data = data.drop('time', axis=1)

    #data["volatility"] = (data["high"] - data["low"]) / max(data["high"])

    #change(data, "volume")

    data["low24"] = data["low"].rolling(window=24).min()
    data["high24"] = data["high"].rolling(window=24).max()

    data["close_low24"] = (data["close"] - data["low24"]) / data["high24"]
    data["close_high24"] = (data["high"] - data["low24"]) / data["high24"]

    data = data.drop('low24', axis=1)
    data = data.drop('high24', axis=1)

    data["volume"] = (data["volume"] - data["volume"].mean()) / data["volume"].std()

    change(data, "open")
    change(data, "close")
    change(data, "high")
    change(data, "low")

    data = data.clip(-1, 1)
    data = data.iloc[24:]

    return data.to_numpy()

train_data_candles = candles.load_candles(".", hp_chart, train_start_date, train_end_date)
train_data_input = preprocess(train_data_candles)

eval_data_candles = candles.load_candles(".", hp_chart, eval_start_date, eval_end_date)
eval_data_input = preprocess(eval_data_candles)

test_data_candles = candles.load_candles(".", hp_chart, test_start_date, test_end_date)
test_data_input = preprocess(test_data_candles)

past = 5

def eval(model, data):
    env = DummyVecEnv([lambda: TraderEnv(data, past, False)])

    obs = env.reset()
    rewards = []
    done = False
    reward_sum = 0
    actions = [0, 0]
    action_rewards = [0, 0]

    while not done:
        action, _states = model.predict(obs)
        actions[action[0]] = actions[action[0]] + 1

        obs, reward, done, info = env.step(action)

        action_rewards[action[0]] = action_rewards[action[0]] + reward[0]

        reward_sum = reward_sum + reward
        rewards.append(reward_sum)

    return rewards, actions, action_rewards

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: TraderEnv(train_data_input, past, True)])
#env = SubprocVecEnv([lambda: TraderEnv(train_data_input, past) for i in range(5)])
model = PPO2(MlpPolicy, env, verbose=0)
#model = A2C(MlpPolicy, env, verbose=0)
#model = DQN(MlpPolicy, env, verbose=0, prioritized_replay=True)
#model = TRPO(MlpPolicy, env, verbose=0)

best = None
# for i in range(1, 20):
#     model.learn(total_timesteps=20000)
#
#     rewards, actions, action_rewards = eval(model, eval_data_input)
#     reward = rewards[-1]
#
#     print(i, reward, actions, action_rewards)
#
#     if not best or reward > best:
#         best = reward
#         model.save("trader_ppo2")

model = model.load("trader_ppo2")
rewards, actions, action_rewards = eval(model, test_data_input)

print(rewards[-1], actions, action_rewards)

fig = plt.figure()
plt.plot(rewards)
plt.show()