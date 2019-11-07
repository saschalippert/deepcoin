import numpy as np


class Order:

    def __init__(self, entry_price, long):
        self._entry_price = entry_price
        self._long = long

    def close(self, current_price, fee_percentage):
        profit = current_price - self._entry_price

        if (not self._long):
            profit *= -1

        return profit, self._entry_price * fee_percentage


class Accountant:

    def __init__(self, start_balance = 0):
        self._balance = start_balance

        self._profits = [0, 0]
        self._fees = [0, 0]
        self._gain = [0, 0]
        self._count = [0, 0]

        self._balance_history = [start_balance]
        self._balance_max = start_balance
        self._balance_min = start_balance

        self._wins = []
        self._losses = []

        self._drawdown_max = 0

        self._continuous_losses = 0
        self._max_loss_streak = 0

        self._avg_win = 0
        self._avg_loss = 0
        self._avg_gain = 0

    def close(self, order, current_price, fee_percentage):
        profit, fee = order.close(current_price, fee_percentage)
        current_gain = (profit - fee)
        order_side = not order._long

        self._balance += current_gain
        self._balance_max = max(self._balance_max, self._balance)
        self._balance_min = min(self._balance_min, self._balance)
        self._balance_history.append(self._balance)

        if (profit > 0):
            self._continuous_losses = 0
            self._wins.append(profit)
        else:
            self._continuous_losses += 1
            self._max_loss_streak = max(self._max_loss_streak, self._continuous_losses)
            self._losses.append(profit)

        drawdown = self._balance - self._balance_max
        self._drawdown_max = min(self._drawdown_max, drawdown)

        self._gain[order_side] += current_gain
        self._count[order_side] += 1
        self._profits[order_side] += profit
        self._fees[order_side] += fee

        if (len(self._wins) > 0):
            self._avg_win = np.average(self._wins)

        if (len(self._losses) > 0):
            self._avg_loss = np.average(self._losses)

        sum_count = sum(self._count)

        if (sum_count > 0):
            self._avg_gain = sum(self._gain) / sum_count

        return current_gain
