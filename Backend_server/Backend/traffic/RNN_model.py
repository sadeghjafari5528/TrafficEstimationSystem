import numpy as np
import matplotlib.pyplot as plt
from django.core.cache import cache

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from traffic.models import Record

import threading

def get_series_data(window, data):
    X = []
    Y = []
    for i in range(len(data)):
        if i > window:
            X.append(data[i - window:i])
            Y.append(data[i])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def get_model(X_train_shape):
    model = keras.Sequential(
        [
            layers.Input((X_train_shape,1)),
            layers.SimpleRNN(50),
            layers.RepeatVector(1),
            layers.Dropout(0.1),
            layers.SimpleRNN(50),
            layers.Dropout(0.1),
            layers.RepeatVector(1),
            layers.SimpleRNN(50),
            layers.Dropout(0.1),
            layers.RepeatVector(1),
            layers.SimpleRNN(50),
            layers.Dense(1)
        ]
    )
    return model

value_train = []
records = Record.objects.values('no_cars')
for record in records:
    value_train.append(record["no_cars"])

last_data = value_train[:10]
cache.set('last_data', last_data)

# reshape data
value_train = np.array(value_train).reshape(-1, 1)

# normalize data
scaler = MinMaxScaler()
scaler.fit(value_train)
value_train = scaler.transform(value_train)

# preparing data for using LSTM model.
window = 5
X_train, Y_train = get_series_data(window, value_train)

model = get_model(X_train.shape[1])
print(model.summary())

model.compile(
    loss=keras.losses.mean_absolute_error,
    optimizer="adam",
)

history = {}
history['model'] = model.fit(
    X_train, Y_train, validation_split=0.2, batch_size=4, epochs=2
)
model.save('model')



