from keras.layers import Dense, Activation
from keras.models import Sequential
import numpy as np

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

model = Sequential()
model.add(Dense(784, input_shape=(784,1,)))
model.add(Activation('sigmoid'))
model.add(Dense(15))
model.add(Activation('sigmoid'))
model.add(Dense(10))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=1000, batch_size=32)

