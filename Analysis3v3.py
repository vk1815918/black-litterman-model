# PYTHON 3
# START OF PROGRAM INFORMATION
# -------------------------------------------
# Analysis2v3.py
# Author: Sharv Save
# Purpose:
# To conduct public sentiment analysis of an equity through the bag-of-words model
# High level description:
# To web-scrape 4 different articles that relate to the earnings of a certain stock,
# and then to analyze and look for certain key-words that indicate public sentiment
# of the stock, ie. "overbought" "oversold" "meets expectations" etc.
# END OF PROGRAM INFORMATION
# -------------------------------------------

from datetime import timedelta
import requests as rq
import numpy as np
import pandas as pd
import json as js
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import ta

plt.style.use("fivethirtyeight")

# ## Analysis 3
# ### Short term Prediction (~2 months)
# must train the model, and return the last price of the 30 day prediction to give the user a price target.
class Analysis3():
    
    def __init__(self, ticker):
        print("\ninside Analysis3.__init__")
        self.ticker = ticker
        self.api_key = '1bd3a664e9e53b21b0264f0a985fce0f'
        self.url = "https://financialmodelingprep.com/api/v3/historical-price-full/{}?apikey={}".format(ticker.upper(),
                                                                                                        self.api_key)
        self.raw_data = 'analysis3_raw_data_' +  str(self.ticker) + '.csv'
        self.clean_data = 'analysis3_clean_data_' +  str(self.ticker) + '.csv'

        self.figure1 = 'analysis3_figure1_LOSS_' +  str(self.ticker) + '.png'
        self.figure2 = 'analysis3_figure2_ACCURACY_' +  str(self.ticker) + '.png'
        self.figure3 = 'analysis3_figure3_CLOSING_PRICES_' +  str(self.ticker) + '.png'
        self.figure4 = 'analysis3_figure4_PREDICTION_' +  str(self.ticker) + '.png'
    # end of function

    def get_data(self):
        print("\ninside Analysis3.get_data")
    
        r_data = rq.get(self.url)
        data_historical = r_data.text
        js_data = js.loads(data_historical)
        df = pd.DataFrame.from_dict(js_data['historical'])
        df = df[['date', 'open', 'high', 'low', 'close', 'adjClose', 'volume',
                 'unadjustedVolume', 'change', 'changePercent', 'vwap', 'label',
                 'changeOverTime']]
        df.to_csv(self.raw_data)
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df.to_csv(self.clean_data)
        return df
    # end of function
    
    def split_sequence(self, seq, n_steps_in, n_steps_out):
        print("\ninside Analysis3.split_sequence")
        X, y = [], []
        for i in range(len(seq)):
            end = i + n_steps_in
            out_end = end + n_steps_out
            if out_end > len(seq):
                break
            seq_x, seq_y = seq[i:end, :], seq[end:out_end, 0]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    # end of function
    
    def visualize_training_results(self, results):
        print("\ninside Analysis3.visualize_training_results")
        history = results.history
        plt.figure(figsize=(16,5))
        plt.plot(history['val_loss'])
        plt.plot(history['loss'])
        plt.legend(['val_loss', 'loss'])
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        #plt.show()
        plt.savefig(self.figure1)
        
        plt.figure(figsize=(16,5))
        plt.plot(history['val_accuracy'])
        plt.plot(history['accuracy'])
        plt.legend(['val_accuracy', 'accuracy'])
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        #plt.show()
        plt.savefig(self.figure2)
        return
    # end of function
    
    def layer_maker(self, n_layers, n_nodes, activation, drop=None, d_rate=.5):
        print("\ninside Analysis3.layer_maker")
        for x in range(1,n_layers+1):
            self.model.add(LSTM(n_nodes, activation=activation, return_sequences=True))
            try:
                if x % drop == 0:
                    self.model.add(Dropout(d_rate))
            except:
                pass
            # end try
        # end for
        
        return
    # end of function
    
    def validater(self, n_per_in, n_per_out):
        print("\ninside Analysis3.validater")
        predictions = pd.DataFrame(index=self.df.index, columns=[self.df.columns[0]])
        for i in range(n_per_in, len(self.df)-n_per_in, n_per_out):
            x = self.df[-i - n_per_in:-i]
            yhat = self.model.predict(np.array(x).reshape(1, n_per_in, self.n_features))
            yhat = self.close_scaler.inverse_transform(yhat)[0]
            pred_df = pd.DataFrame(yhat,
                                   index=pd.date_range(start=x.index[-1],
                                                       periods=len(yhat),
                                                       freq="B"),
                                   columns=[x.columns[0]])
            predictions.update(pred_df)
        return predictions
    # end of function
    
    def val_rmse(self, df1, df2):
        print("\ninside Analysis3.val_rmse")
        df = df1.copy()
        df['close2'] = df2.close
        df.dropna(inplace=True)
        df['diff'] = df.close - df.close2
        rms = (df[['diff']]**2).mean()
        return float(np.sqrt(rms))
    # end of function

    def perform_analysis3(self):
        print("\ninside Analysis3.perform_analysis3")

        # # 1. Data
        self.df = self.get_data()
        print(self.df.dtypes)
    
        # # 2. Data Cleaning
        self.df['date'] = pd.to_datetime(self.df.date)
        self.df.set_index('date', inplace=True)
        self.df.dropna(inplace=True)
        self.df = ta.add_all_ta_features(self.df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
        self.df.drop(['open', 'high', 'low', 'volume'], axis=1, inplace=True)
        self.df = self.df.tail(1000)
        self.close_scaler = RobustScaler()
        self.close_scaler.fit(self.df[['close']])
        scaler = RobustScaler()
        self.df = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns, index=self.df.index)
    
        n_per_in  = 90
        n_per_out = 30
        self.n_features = self.df.shape[1]
        X, y = self.split_sequence(self.df.to_numpy(), n_per_in, n_per_out)
        
        # # 3. Build/Train the Model
        self.model = Sequential()
        activ = "tanh"
        self.model.add(LSTM(90,
                       activation=activ,
                       return_sequences=True,
                       input_shape=(n_per_in, self.n_features)))
        
        self.layer_maker(n_layers=1, n_nodes=30, activation=activ)
        
        self.model.add(LSTM(60, activation=activ))
        self.model.add(Dense(n_per_out))
        self.model.summary()
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        #self.model.compile(optimizer='adam', loss='mse', metrics=['precision'])
        res = self.model.fit(X, y, epochs=50, batch_size=128, validation_split=0.1)
        
        # # 4. Visualize Training Results
        self.visualize_training_results(res)
        
        # # 5. Use the Model to Predict
        actual = pd.DataFrame(self.close_scaler.inverse_transform(self.df[["close"]]),
                              index=self.df.index,
                              columns=[self.df.columns[0]])
        
        predictions = self.validater(n_per_in, n_per_out)
        print("RMSE:", self.val_rmse(actual, predictions))
        
        plt.figure(figsize=(16,6))
        
        plt.plot(predictions, label='Predicted')
        
        plt.plot(actual, label='Actual')
        
        plt.title(f"Predicted vs Actual Closing Prices")
        plt.ylabel("Price")
        plt.legend()
        #plt.show()
        plt.savefig(self.figure3)
    
        yhat = self.model.predict(np.array(self.df.tail(n_per_in)).reshape(1, n_per_in, self.n_features))
        
        yhat = self.close_scaler.inverse_transform(yhat)[0]
        
        preds = pd.DataFrame(yhat,
                             index=pd.date_range(start=self.df.index[-1]+timedelta(days=1),
                                                 periods=len(yhat),
                                                 freq="B"),
                             columns=[self.df.columns[0]])
        
        pers = n_per_in
        
        actual = pd.DataFrame(self.close_scaler.inverse_transform(self.df[["close"]].tail(pers)),
                              index=self.df.close.tail(pers).index,
                              columns=[self.df.columns[0]]).append(preds.head(1))
        
        print(preds)
        
        plt.figure(figsize=(40,6))
        plt.plot(actual, label="Training Values")
        plt.plot(preds, label="Predicted Values")
        plt.ylabel("Price")
        plt.xlabel("Dates")
        plt.title(f"Prediction the next {len(yhat)} days")
        plt.legend()
        #plt.show()
        plt.savefig(self.figure4)
        
        return True
    # end of function

# end of class