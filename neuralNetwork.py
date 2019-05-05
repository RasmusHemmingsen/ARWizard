from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
#from tensorflow import keras
from sklearn.model_selection import train_test_split

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadData():
    X = pd.read_csv(r"AR Wizard\Assets\TrainingData\HandGestureData2.csv")
    y = np.array([0]*131 + [1]*131 + [2]*125 + [3]*127)

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=7)

    return X_train, X_test, y_train, y_test


train_data, test_data, train_label, test_label = loadData()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4560, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2280, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(4560, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2280, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(4560, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=train_data, y=train_label, epochs=25)
loss, acc = model.evaluate(x=test_data, y=test_label)

print("Loss {}, Accuracy {}".format(loss, acc))