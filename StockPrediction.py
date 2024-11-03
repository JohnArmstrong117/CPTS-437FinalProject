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