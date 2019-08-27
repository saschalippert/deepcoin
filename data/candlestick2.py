import plotly.graph_objects as go
from deepcoin_candles import load_candles
import plotly
import plotly.express as px
import pandas as pd
from datetime import datetime

import plotly.io as pio

start_date = datetime(2018, 3, 2)
end_date = datetime(2018, 4, 24)
df = load_candles("..", "btceur", start_date, end_date)

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

plotly.io.orca.config.executable = '/home/sascha/Downloads/orca'

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], hoverinfo='skip'),
    secondary_y=False,
)

fig.add_trace(
    go.Bar(name= "df", x=df['time'], y=df['volume'], hoverinfo='skip'),
    secondary_y=True,
)

# fig.write_html("hans.html")
# fig.write_image("fig1.png")

pio.write_html(fig, file='hello_world.html', auto_open=True)
