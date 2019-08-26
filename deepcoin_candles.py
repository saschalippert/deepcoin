import numpy as np
import json
from datetime import timedelta, datetime
from collections import OrderedDict


def date_range(start: datetime, end: datetime, step: timedelta):
    date_list = []

    while start < end:
        date_list.append(start)
        start += step

    return date_list


def load_candles(currency_pair, start_date, end_date):
    candles = OrderedDict()

    for c in date_range(start_date, end_date, timedelta(days=1)):
        filename = c.strftime(f'{currency_pair}/{currency_pair}_%Y_%m_%d.json')

        with open(filename) as json_file:
            json_candles = json.load(json_file)
            candles.update(json_candles)

    np_candles = np.zeros((len(candles)))

    for index, key in enumerate(candles):
        candle = candles[key]

        c_time = candle["time"]
        c_low = candle["low"]
        c_high = candle["high"]
        c_open = candle["open"]
        c_close = candle["close"]
        c_volume = candle["volume"]

        np_candles[index] = c_close

    return np_candles
