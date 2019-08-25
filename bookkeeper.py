from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import json

class Bookkeeper:

    def __init__(self, model_name, num_episodes, hyper_parameters):
        self._train_losses = []
        self._eval_losses = []
        self._test_losses = []
        self._num_episodes = num_episodes
        self._home = str(Path.home())
        self._min_loss = float("inf")
        self._best_model = None
        self._model_name = model_name

        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")

        log_dir = os.path.join(self._home, "tensorboard", "logs", self._model_name, now_str)

        self._writer = SummaryWriter(log_dir=log_dir)
        self._writer.add_text("text/hyper_parameters", json.dumps(hyper_parameters))

    def __add_loss(self, loss, losses):
        losses.append(loss)

        avg_loss = np.average(losses)
        min_loss = np.min(losses)

        return avg_loss, min_loss

    def train_step(self, train_loss, eval_loss, model, model_copy):
        t_avg_loss, t_min_loss = self.__add_loss(train_loss, self._train_losses)
        e_avg_loss, e_min_loss = self.__add_loss(eval_loss, self._eval_losses)

        episode = len(self._train_losses)

        self._writer.add_scalar('data/train/loss', train_loss, episode)
        self._writer.add_scalar('data/eval/loss', eval_loss, episode)

        print(f'Episode: {episode:>5}/{self._num_episodes:<5} TrainLoss: {train_loss:.8f} AvgTrainLoss: {t_avg_loss:.8f} MinTrainLoss: {t_min_loss:.8f} EvalLoss: {eval_loss:.8f} AvgEvalLoss: {e_avg_loss:.8f} MinEvalLoss: {e_min_loss:.8f}')

        if(eval_loss < self._min_loss):
            self._best_model = model_copy()
            self._best_model.load_state_dict(model.state_dict())

    def get_best_model(self):
        return self._best_model

    def close(self):
        self._writer.close()

    def add_figure(self, plot_figure, figure_name):
        old_backend = plt.get_backend()
        plt.switch_backend('agg')

        figure = plot_figure()
        self._writer.add_figure(f'figure/{figure_name}', figure)

        plt.switch_backend(old_backend)
