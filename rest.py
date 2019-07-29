import requests
from collections import OrderedDict
from datetime import timedelta, datetime
from time import sleep
import json

def load_candles(start, end, granularity):
    url = 'https://api.pro.coinbase.com/products/BTC-USD/candles?start=' + start + '&end=' + end + '&granularity=' + granularity

    response = requests.get(url)
    content = json.loads(response.content)

    candles = dict()

    for candle in content:
        new_candle = dict()

        new_candle["time"] = int(candle[0])
        new_candle["low"] = float(candle[1])
        new_candle["high"] = float(candle[2])
        new_candle["open"] = float(candle[3])
        new_candle["close"] = float(candle[4])
        new_candle["volume"] = float(candle[5])

        candles[new_candle["time"]] = new_candle

    return candles

def date_range(start: datetime, end: datetime, step: timedelta):
    date_list = []

    while start < end:
        date_list.append((start, start + step))
        start += step

    return date_list

start_date = datetime(2016, 1, 1)
end_date = datetime(2019, 6, 25)

candles_loaded = dict()

candles_date = start_date.date()
for chunk_start_date, chunk_end_date in date_range(start_date, end_date, timedelta(hours=5)):
    iso_start = chunk_start_date.replace(microsecond=0).isoformat()
    iso_end = chunk_end_date.replace(microsecond=0).isoformat()

    candles_loaded.update(load_candles(iso_start, iso_end, '60'))

    candles_write = dict()

    if(candles_date != chunk_end_date.date()):
        for ts_key in candles_loaded.copy().keys():
            candle_time = datetime.utcfromtimestamp(ts_key)

            if(candle_time.date() == candles_date):
                candles_write[ts_key] = candles_loaded[ts_key]
                del candles_loaded[ts_key]

        filename = candles_date.strftime('btceur/btceur_%Y_%m_%d.json')

        with open(filename, 'w') as outfile:
            json.dump(OrderedDict(sorted(candles_write.items())), outfile, indent=4)

        expected_len = 60 * 24
        actual_len = len(candles_write)
        print("written", candles_date, actual_len, (expected_len - actual_len) / expected_len)

        candles_date = chunk_end_date.date()

    sleep(0.5)

print("reminders", candles_loaded)
