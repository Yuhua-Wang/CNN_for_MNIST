import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from matplotlib import pyplot as plt

print("tensorflow version:" + tf.__version__)

# This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# RGB is 8 bit so each color is ranged from 0-255 (2^8 = 256).
# By dividing by 255, the 0-255 range can be described with a 0.0-1.0 range where 0.0 means 0 (0x00) and 1.0 means 255 (0xFF).
x_train, x_test = x_train / 255.0, x_test / 255.0
inputShape = x_train[1,:,:].shape

# shapes of the parameters
#'''
print("input shape: " + str(inputShape))
print("x_train shape:" + str(x_train.shape)) ##(60000, 28, 28)
print("x_test shape:" + str(x_test.shape) ) ##(10000, 28, 28)
print("y_train shape:" + str(y_train.shape)) #(60000,)
print("y_test shape:" + str(y_test.shape)) #(10000,)
#'''

# Show an image in the collection
def showImage(collection, index):
    example_image = collection[index]
    plt.imshow(example_image.reshape(28,28), cmap='gray')
    plt.show()

'''
showImage(x_train,38415)
'''

## NN model, test set acuracy: 97.18%, training time: 10s
'''
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(30))
model.add(Dense(10,activation='softmax'))

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(loss=loss_fn,
              optimizer='adam',
              metrics=['accuracy'])

print("**************************** START TRAINING ***************************")
model.fit(x_train, y_train, batch_size=32, epochs=5)
print("**************************** START EVALUATION ***************************")
model.evaluate(x_test,  y_test, verbose=2)
'''


## CNN model, test set accuracy: 97.59%, training time: 90s
#'''
# reshape the examples from (28,28) to (28,28,1) where the last dim is the channel
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
print("x_train shape after reshape:" + str(x_train.shape))
print("x_test shape after reshape:" + str(x_test.shape))
inputShape = x_train[1,:,:].shape

model = Sequential()
model.add(Conv2D(2, (4,4), strides= 1, padding="same", activation = 'relu', input_shape= inputShape)) #(28,28)
#model.add(MaxPool2D((4,4), strides = 4)) #(7,7)
model.add(Conv2D(3, (4,4), strides= 1, padding="same", activation = 'relu'))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(10,activation='softmax'))

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(loss=loss_fn,
              optimizer='adam',
              metrics=['accuracy'])

print("**************************** START TRAINING ***************************")
model.fit(x_train, y_train, batch_size=32, epochs=5)
print("**************************** START EVALUATION ***************************")
model.evaluate(x_test,  y_test, verbose=2)
#'''