# Model Testing

## Comparison
Model comparison is a useful tool in evaluating the performance of our LSTM model. We compare our model with three others:

- Random Forest
- SVR
- Linear Regression

And plot the results of each model vs. the actual data. To see the plot, run the program using the steps outlined in [the readme](../README.md)

## Metrics
Common metrics used to test regression models are applied to our model, as well as the comparison models:

- `MSE` is a measure of the squared difference between the predicted and actual values. In other words, it is a quantification of how far off the predicted values are from the actual.

-  ![rsquared](https://latex.codecogs.com/svg.image?{\color{White}r^{2}}) is the coeffiecient of determination, which is a measure of how well the model fits the data. It describes the proportion of the variance in the dependent variable that is predictable from the independent variables.

The result of these metric calculations are printed in `model_performance_metrics.txt`.