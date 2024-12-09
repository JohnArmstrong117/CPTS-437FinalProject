# Project Overview
For this project, we propose a predictive model to analyze historical stock market data. We have
two major practical motivations for this project. The first is to analyze the risk of investing in a
given stock based on its prior market performance data. Next, we want the model to be able to
predict the future prices of stocks. In the pursuit of these goals, we will look at how the change
in stock price over time, daily return, moving average, and correlation between different stocks
can be used to forecast future behavior. A model like this would allow investors to make
informed decisions about where to put their money based on hard data rather than speculation.
This will have utility to any investor looking to maximize the potential of their investments.
We will be comparing and contrasting various models, discussing the accuracy and eventually deciding
which model fits the data best.

# Models
## `LSTM` - Long Short Term Memory
A type of RNN (Recurrent Neural Network) which uses gates to control the flow of information. They are tailored to learn long-term dependencies bewteen time steps in data.

## `Random Forest`
An ensemble of decision trees commonly used in both regression and classification problems. The output of multiple decision trees are combined to reach a single result.

## `SVR` - Support Vector Regression
An extension of the popular model SVM (Support Vector Machine) using the same principle (finding the best fit). In `SVR` the best fit is a hyperplane that has the maximum number of points.

## `Linear Regression`
A model which estimates the linear relationship between a dependent and independent variable(s).

# Data 
In order to create a machine-learning program with the purpose of predicting changes in stock
prices, we need a dataset with pertinent information to that topic. We will be using data largely
sourced from Yahoo Finance, a reliable source of historical stock price information. The data
that we use will have a number of features, including opening, high, low, and closing prices. We
will have this information for a number of different stocks, and have data points for each stock
on each day between set two dates. This data should be sufficient for creating and training our
machine-learning algorithm. The dataset can be found [here](https://finance.yahoo.com/quote/AAPL/).

