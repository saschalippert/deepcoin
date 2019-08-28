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

    def __init__(self):
        self._profits = [0, 0]
        self._fees = [0, 0]
        self._gain = [0, 0]
        self._count = [0, 0]
        self._gains = []

    def close(self, order, current_price, fee_percentage):
        profit, fee = order.close(current_price, fee_percentage)

        is_long = order._long

        self._count[is_long] += 1
        self._profits[is_long] += profit
        self._fees[is_long] += fee

        self._gain[is_long] += (profit - fee)

        self._gains.append(self._gain)
