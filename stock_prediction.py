import numpy as np 
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tabulate import tabulate

parser = argparse.ArgumentParser(description="Stock Price Prediction with n_days")
parser.add_argument(
    '--days', 
    type=int, 
    default=1,
    help="Number of future days to predict"
)

args = parser.parse_args()

n = args.days

# Downloading Dataset from yfinance
ticker = 'AAPL'
start_date = '2012-01-01'
end_date = '2024-01-01'
data = yf.download(ticker, start=start_date, end=end_date)

# Extracting closing prices
closing_prices = data[['Close']]

# Preprocessing the data to make it more suitable for the models
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Using a 60-day frame for training in order to predict the next day
days_sequence_len = 60
X_train, y_train = [], []

for i in range(days_sequence_len, len(scaled_data) - n):
    X_train.append(scaled_data[i - days_sequence_len:i, 0])
    y_train.append(scaled_data[i + n - 1, 0])  # Target is n days ahead
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Split data for testing
test_data = yf.download(ticker, start='2021-01-01', end=datetime.datetime.now())
actual_closing_prices = test_data['Close'].values[n:]

total_data = pd.concat((data['Close'], test_data['Close']), axis=0)
test_inputs = total_data[len(total_data) - len(test_data) - days_sequence_len:].values
test_inputs = test_inputs.reshape(-1, 1)
test_inputs = scaler.transform(test_inputs)

X_test = []
for i in range(days_sequence_len, len(test_inputs) - n):
    X_test.append(test_inputs[i - days_sequence_len:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 1. LSTM Model
# Building the LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Training the LSTM model
epochs = 50
batch_size = 32
history = lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# LSTM prediction
predicted_prices_lstm = lstm_model.predict(X_test)
predicted_prices_lstm = scaler.inverse_transform(predicted_prices_lstm)

# 2. Random Forest Model
# Reshaping data for Random Forest
X_train_rf = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
X_test_rf = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train)

# Random Forest prediction
predicted_prices_rf = rf_model.predict(X_test_rf)
predicted_prices_rf = predicted_prices_rf.reshape(-1, 1)

# 3. Support Vector Regressor (SVR)
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_rf, y_train)

# SVR prediction
predicted_prices_svr = svr_model.predict(X_test_rf)
predicted_prices_svr = predicted_prices_svr.reshape(-1, 1)

# 4. Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_rf, y_train)

# Linear Regression prediction
predicted_prices_lr = lr_model.predict(X_test_rf)
predicted_prices_lr = predicted_prices_lr.reshape(-1, 1)

# Inverse scaling
predicted_prices_rf = scaler.inverse_transform(predicted_prices_rf)
predicted_prices_svr = scaler.inverse_transform(predicted_prices_svr)
predicted_prices_lr = scaler.inverse_transform(predicted_prices_lr)

# Plotting
plt.figure(figsize=(14, 5))
plt.plot(actual_closing_prices, color="black", label="Actual Price")
plt.plot(predicted_prices_lstm, color="green", label="LSTM Predicted Price")
plt.plot(predicted_prices_rf, color="blue", label="Random Forest Predicted Price")
plt.plot(predicted_prices_svr, color="red", label="SVR Predicted Price")
plt.plot(predicted_prices_lr, color="orange", label="Linear Regression Predicted Price")
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Calculate performance metrics
metrics = {
    "Model": ["LSTM", "Random Forest", "SVR", "Linear Regression"],
    "MSE": [
        mean_squared_error(actual_closing_prices, predicted_prices_lstm),
        mean_squared_error(actual_closing_prices, predicted_prices_rf),
        mean_squared_error(actual_closing_prices, predicted_prices_svr),
        mean_squared_error(actual_closing_prices, predicted_prices_lr)
    ],
    "R2": [
        r2_score(actual_closing_prices, predicted_prices_lstm),
        r2_score(actual_closing_prices, predicted_prices_rf),
        r2_score(actual_closing_prices, predicted_prices_svr),
        r2_score(actual_closing_prices, predicted_prices_lr)
    ]
}

# Convert metrics dictionary to a pandas DataFrame
metrics_df = pd.DataFrame(metrics)

# Format the metrics as a table
table = tabulate(metrics_df, headers='keys', tablefmt='fancy_grid', showindex=False)

# Save the formatted table to a txt file with utf-8 encoding
with open('model_performance_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(table)

# Plot the loss over epochs
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('LSTM Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# LSTM
plt.subplot(2, 2, 1)
plt.plot(actual_closing_prices, color='red', label='Actual Prices')
plt.plot(predicted_prices_lstm, color='blue', label='Predicted Prices')
plt.title('LSTM Model')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# Random Forest
plt.subplot(2, 2, 2)
plt.plot(actual_closing_prices, color='red', label='Actual Prices')
plt.plot(predicted_prices_rf, color='blue', label='Predicted Prices')
plt.title('Random Forest Model')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# SVR
plt.subplot(2, 2, 3)
plt.plot(actual_closing_prices, color='red', label='Actual Prices')
plt.plot(predicted_prices_svr, color='blue', label='Predicted Prices')
plt.title('SVR Model')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# Linear Regression
plt.subplot(2, 2, 4)
plt.plot(actual_closing_prices, color='red', label='Actual Prices')
plt.plot(predicted_prices_lr, color='blue', label='Predicted Prices')
plt.title('Linear Regression Model')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# MSE and R2 Score
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(metrics["Model"], metrics["MSE"], color='orange')
plt.title('Mean Squared Error (MSE)')
plt.xlabel('Model')
plt.ylabel('MSE')

plt.subplot(1, 2, 2)
plt.bar(metrics["Model"], metrics["R2"], color='green')
plt.title('R2 Score')
plt.xlabel('Model')
plt.ylabel('R2 Score')
plt.tight_layout()
plt.show()