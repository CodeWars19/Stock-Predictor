import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib as plt
import pandas as pd
import ta
from datetime import timedelta

data = pd.read_csv("QQQ.csv")
data["Date"] = pd.to_datetime(data.Date)
data.set_index('Date', inplace=True)
data.dropna(inplace=True)
data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
data.drop(['Open', 'High', "Low", "Adj Close", "Volume"], axis=1, inplace=True)

data = np.array(data)

x_train = []
y_train = []
interval = 40
for i in range(len(data)):
    if i+size < len(data):
        x_train.append(data[i:i+size, 0])
        y_train.append(data[i+size, 0])
    else:
        break
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(60, activation="tanh", return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.LSTM(60, activation="tanh", return_sequences=True))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.LSTM(60, activation="tanh", return_sequences=True))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(30))
model.add(tf.keras.layers.Dense(1))

model.compile(
    optimizer='adam',
    loss='MSLE',
    metrics=['accuracy'],
)


print(model.summary())

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

print(x_train[-1])

new_data = x_train
days = 20
for i in range(days):
    x = model.predict(np.array([new_data[-1]]))
    print(x)
    np.append(new_data, x, axis=0)

