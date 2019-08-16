from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import os
from datetime import datetime

class Bookkeeper:

    def _add_loss(self, loss, losses):
        avg_loss = np.average(losses)
        min_loss = np.min(losses)

        return avg_loss, min_loss

    def step(self, train_loss, eval_loss):
        self.train_losses.append(train_loss)
        self.eval_losses.append(eval_loss)

        t_avg_loss, t_min_loss = self._add_loss(train_loss, self.train_losses)
        e_avg_loss, e_min_loss = self._add_loss(eval_loss, self.eval_loss)

        episode = len(self.train_losses)

        self.writer.add_scalar('data/train/loss', train_loss, episode)

        print(f'Episode: {episode}/{self.num_episodes} Loss: {train_loss} AvgLoss: {t_avg_loss} BstLoss: {t_min_loss}')



    def start(self, model_name, num_episodes):
        self.train_losses = []
        self.eval_losses = []
        self.num_episodes = num_episodes
        self.home = str(Path.home())
        self.model_name = model_name

        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")

        log_dir = os.path.join(self.home, "tensorboard", "logs", model_name, now_str)

        self.writer = SummaryWriter(log_dir=log_dir)

    def stop(self):
        self.writer.close()

#writer.add_figure('matplotlib/figure', fig)
#plt.switch_backend('agg')
#matplotlib.pyplot.get_backend()
