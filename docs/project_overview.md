# Project Overview
For this project, we propose a predictive model to analyze historical stock market data. We have
two major practical motivations for this project. The first is to analyze the risk of investing in a
given stock based on its prior market performance data. Next, we want the model to be able to
predict the future prices of stocks. In the pursuit of these goals, we will look at how the change
in stock price over time, daily return, moving average, and correlation between different stocks
can be used to forecast future behavior. A model like this would allow investors to make
informed decisions about where to put their money based on hard data rather than speculation.
This will have utility to any investor looking to maximize the potential of their investments.

# Model
The primary model used will be a recurrent neural network(RNN). More specifically we will be
using the long short-term memory(LSTM) variant. The reason we are using an LSTM is due to
the timeframe involved with stock data. The current behavior of stock prices are not just
determined by immediate past data but also by historical trends that must be considered when
predicting the future of that stock. A traditional RNN suffers from the vanishing gradient problem
which can cause the neural network to essentially stop learning after a certain point due to the
model’s weights not being effectively updated when the gradients get too small. The LSTM was
created specifically as a response to the vanishing gradient problem. It should be sufficient for
our need to maintain important information about long-term dependencies over many iterations.

# Data 
In order to create a machine-learning program with the purpose of predicting changes in stock
prices, we need a dataset with pertinent information to that topic. We will be using data largely
sourced from Yahoo Finance, a reliable source of historical stock price information. The data
that we use will have a number of features, including opening, high, low, and closing prices. We
will have this information for a number of different stocks, and have data points for each stock
on each day between set two dates. This data should be sufficient for creating and training our
machine-learning algorithm. The dataset can be found [here](https://finance.yahoo.com/quote/AAPL/).

