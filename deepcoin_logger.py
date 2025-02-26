from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import json


class Logger:

    def __init__(self, model_name):
        self._train_losses = []
        self._eval_losses = []
        self._home = str(Path.home())
        self._min_loss = float("inf")
        self._best_model = None
        self._model_name = model_name
        self._num_episodes = None
        self._data_name = None

        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")

        log_dir = os.path.join(self._home, "tensorboard", "logs", self._model_name, now_str)

        print("log_dir", log_dir)

        self._writer = SummaryWriter(log_dir=log_dir)

    def __add_loss(self, loss, losses):
        losses.append(loss)

        avg_loss = np.average(losses)
        min_loss = np.min(losses)

        return avg_loss, min_loss

    def train_start(self, num_episodes, data_name, hyperparameters):
        self._num_episodes = num_episodes
        self._data_name = data_name
        self._train_losses = []
        self._eval_losses = []
        self._min_loss = float("inf")
        self._best_model = None

        self._writer.add_text(f"text/{self._data_name}/hyperparameters", json.dumps(hyperparameters))

    def train_step(self, train_loss, eval_loss, model, model_copy):
        t_avg_loss, t_min_loss = self.__add_loss(train_loss, self._train_losses)
        e_avg_loss, e_min_loss = self.__add_loss(eval_loss, self._eval_losses)

        episode = len(self._train_losses)

        self._writer.add_scalar(f'data/{self._data_name}/train/loss', train_loss, episode)
        self._writer.add_scalar(f'data/{self._data_name}/eval/loss', eval_loss, episode)

        print(f'Episode: {episode:>5}/{self._num_episodes:<5} TrainLoss: {train_loss:.8f} AvgTrainLoss: {t_avg_loss:.8f} MinTrainLoss: {t_min_loss:.8f} EvalLoss: {eval_loss:.8f} AvgEvalLoss: {e_avg_loss:.8f} MinEvalLoss: {e_min_loss:.8f}')

        if (eval_loss < self._min_loss):
            self._best_model = model_copy()
            self._best_model.load_state_dict(model.state_dict())

    def eval_best_model(self):
        self._writer.add_text(f"text/{self._data_name}/model", self._best_model.__repr__())

        return self._best_model

    def close(self):
        self._writer.close()

    def add_figure(self, plot_figure, figure_name):
        old_backend = plt.get_backend()
        plt.switch_backend('agg')

        figure = plot_figure()
        self._writer.add_figure(f'figure/{self._data_name}/{figure_name}', figure)

        plt.switch_backend(old_backend)

    def add_text(self, text_name, text_value):
        self._writer.add_text(f"text/{text_name}", text_value)
