class Order:

    def __init__(self, entry_price, long):
        self._entry_price = entry_price
        self._long = long

    def close(self, current_price, fee):
        profit = current_price - self._entry_price

        if (not self._long):
            profit *= -1

        return profit, self._entry_price * fee

    def direction(self):
        return self._long
