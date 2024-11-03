import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout

#Downloading Dataset form yfinance
ticker = 'APPL'
start_date = '2012-01-01'
end_date = datetime.now()
data = yf.download(ticker, start=start_date, end=end_date)

#Extracting closing prices
closing_prices = data[['Close']]

#Preprocessing the data to make it more suitable for the LSTM model
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(closing_prices)

#Using a 60 day frame for training in order to predict the next day
days_sequence_len = 60
X_train, y_train = [], []

for i in range(days_sequence_len, len(scaled_data)):
    X_train.append(scaled_data[i-days_sequence_len:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Building the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
#Setting Prediciton to one day in future
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
