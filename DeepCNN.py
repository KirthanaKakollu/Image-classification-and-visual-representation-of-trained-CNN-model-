from __future__ import print_function
from keras.datasets import cifar10
import matplotlib
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import print_summary, to_categorical
import sys
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import os
from keras.datasets import mnist, cifar10
from keras import backend as K
from keras.models import Sequential
import pickle


from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import time

from tensorflow.contrib.distributions.python.ops.bijectors import inline

import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt
#%matplotlib inline # Only use this if using iPython
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')

x_train.shape

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=1)

###################layer_dict = dict([(layer.name, layer) for layer in model.layers])

#print(layer_dict)
#exit()

model.evaluate(x_test, y_test)


############################cifar-10###################################



#sys.path.insert(0, 'drive/cifar10')
#os.chdir(“drive/cifar10”)

batch_size = 64
num_classes = 10
epochs =1
model_name = 'keras_cifar10_model'
save_dir = '/model/' + model_name

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(80))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes))
model.add(Activation('softmax'))



opt = SGD(lr=0.01, momentum=0.9, decay=0, nesterov=False)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

########################## visulization ##################################

weights = model.get_weights()
len(weights)
weights[0][0].shape


inp = model.input
outputs = [layer.output for layer in model.layers]

functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]
img = x_test[839]

plt.imshow(img)
plt.show()

t = np.expand_dims(img, axis=0)

t.shape

layer_outs = [func([t, 1.]) for func in functors]

imgs = []

for i in layer_outs[:-4]:
    for j in i:
        k = j[0].T
    for im in k:
        imgs.append(im)

len(imgs)

fig = plt.figure()

for i in range(8):
    for j in range(4):
        plt.subplot2grid((8, 4), (i, j))
        plt.imshow(imgs[i * 4 + j])
        plt.axis('off')
        plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.05, right=0.4, top=0.9)
plt.show()


fig = plt.figure()

for i in range(8):
    for j in range(8):
        plt.subplot2grid((8, 8), (i,j))
        plt.imshow(imgs[128+i*8+j])
        plt.axis('off')
        plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.05, right=0.5, top=0.6)
plt.show()

fig = plt.figure()

for i in range(8):
    for j in range(8):
        plt.subplot2grid((8, 8), (i,j))
        plt.imshow(imgs[384+i*8+j])
        plt.axis('off')
        plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.05, right=0.5, top=0.6)
plt.show()

fig = plt.figure()

for i in range(8):
    for j in range(8):
        plt.subplot2grid((8, 8), (i,j))
        plt.imshow(imgs[512+i*8+j])
        plt.axis('off')
        plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.05, right=0.5, top=0.6)

plt.show()


