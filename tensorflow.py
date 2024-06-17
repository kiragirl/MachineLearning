import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[200.0, 17.0],
              [120.0, 5.0],
              [425.0, 20.0],
              [212.0, 18.0]])
y = np.array([1, 0, 0, 1])

model = Sequential([Dense(units=25, activation='sigmoid'),
                    Dense(units=15, activation='sigmoid'),
                    Dense(units=1, activation='sigmoid')])
#model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y)
print(model.predict([210.0]))

git clone -c http.proxy="PROXY dalian-webproxy.openjawtech.com:3128" https://github.com/