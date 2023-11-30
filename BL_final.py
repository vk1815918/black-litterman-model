import pandas as pd
import numpy as np
import pandas_datareader as web
import yfinance as yf

from keras.models import Sequential
from keras.layers import Dense, LSTM

import seaborn as sb
import matplotlib.pyplot as plt
import tensorflow as tf

tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0][['Symbol', 'GICS Sector']]
tickers['Symbol'] = tickers['Symbol'].str.replace('.', '-')

sector_breakdown = tickers.groupby('GICS Sector')['Symbol'].apply(list)
sector_breakdown = sector_breakdown.to_dict()

ticker_list = []
for sector in sector_breakdown:
    ticker_list.extend(sector_breakdown[sector])

end = pd.Timestamp.today().strftime('%Y-%m-%d')
start = (pd.Timestamp.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')

data = yf.download(ticker_list, start=start, end=end)['Adj Close']

data = data.fillna(method='bfill', axis = 0)
data = data.fillna(method='ffill', axis = 0)

''' historical data loaded

TO DO: 

1) aggregate tickers into their respective sector
2) write a function that fills average return for each sector into dataframe
3) write a function that trains RNN and returns predicted return for each sector
4) create a loop to run through each sector and return predicted return
5) find out some way to give weights to each sector according to their historical predicted return???

'''