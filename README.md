# Stock-Predictor
A neural net I made to predict the  future value of a stock (still a WIP).
The predictor works by putting training and testing data. The data consists of two parts: the x_data and y_data. The y_data is the value of a stock at a certain day and the x_value is a list of the stock values of the previous 40 days. The time interval for the x_data can be changed at will depending on how many days you want the values to be based on.
The neural network is based off of the tensorflow module and is a Long-Short Term Memory neural net. 
The stock which is tracked can be changed by downloading a history of that stock price (I downloaded it from Yahoo finance) and then inputting it for the data value.
Currently, the predictor is soley based on previous data points of a stock rather than external factors, but these factors will be updated overtime in order to increase accuracy and precision
