import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy

x = np.array([[200.0, 17.0],
              [120.0, 5.0],
              [425.0, 20.0],
              [355.0, 20.0],
              [400.0, 20.0],
              [395.0, 20.0],
              [100.0, 20.0],
              [150.0, 10.0],
              [300.0, 20.0],
              [212.0, 18.0]])
y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
layer_1 = keras.layers.Dense(units=3, activation='relu')
# a1 = layer_1(x)
layer_2 = keras.layers.Dense(units=1, activation='sigmoid')
# a2 = layer_2(a1)
# print(a2)

model = tf.keras.Sequential([layer_1, layer_2])
model.compile(loss=BinaryCrossentropy())
model.fit(x, y, epochs=100)
print(model.predict([[100.0, 1.0]]))
print(model.predict([[200.0, 18.0]]))
print(model.predict([[400.0, 30.0]]))
