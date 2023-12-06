import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential

tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0][['Symbol', 'GICS Sector']]

type(tickers['Symbol'])
tickers['Symbol'] = tickers['Symbol'].str.replace('.', '-')
tickers[tickers['Symbol'] == 'BF-B']

sector_breakdown = tickers.groupby('GICS Sector')['Symbol'].apply(list)
sector_breakdown = sector_breakdown.to_dict()

ticker_list = []
for sector in sector_breakdown:
    ticker_list.extend(sector_breakdown[sector])

today = pd.Timestamp.today().strftime('%Y-%m-%d')
month_ago = pd.Timestamp.today() - pd.DateOffset(months=60)

data = yf.download(ticker_list, start=month_ago, end=today)
data = data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

#update with bfill and ffill method instead
data = data.fillna(method='bfill', axis=0)
data = data.fillna(method='ffill', axis=0)

returns = data.pct_change()
returns = returns.droplevel(0, axis=1)

raw_data = {}

for sector, tickers in sector_breakdown.items():
    sector_data = returns[tickers]
    raw_data[sector] = sector_data.mean(axis=1)

raw_data = pd.DataFrame(raw_data)

raw_data = raw_data.dropna()

results = pd.DataFrame(columns=raw_data.columns)

for i, column in enumerate(raw_data.columns):
    y = raw_data[column]
    X = raw_data.drop(column, axis=1)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.33, random_state=42)

    model = Sequential()
    model.add(LSTM(units=5, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=5, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    set_epoch = 20
    train = model.fit(X_train, y_train, batch_size=16, epochs=set_epoch, validation_split=0.1, verbose=0)
    raw_predictions = model.predict(X_test)
    pred = scaler_y.inverse_transform(raw_predictions)

    results[column] = pred.flatten()

print(results)