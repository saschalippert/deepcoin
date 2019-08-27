from datetime import timedelta, datetime
import pandas as pd


def date_range(start: datetime, end: datetime, step: timedelta):
    date_list = []

    while start < end:
        date_list.append(start)
        start += step

    return date_list


def load_candles(candles_path, currency_pair, start_date, end_date):
    candles = pd.DataFrame()

    for c in date_range(start_date, end_date, timedelta(days=1)):
        filename = c.strftime(f'{candles_path}/{currency_pair}/{currency_pair}_%Y_%m_%d.json')
        df = pd.read_json(filename, orient="index")
        candles = candles.append(df)

    return candles
