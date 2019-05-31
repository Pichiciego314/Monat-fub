#!/usr/bin/env python

#imports
import keras
keras.__version__
import rospy
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import load_model




(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.ndim)
print(test_images.ndim)



#declaracion de los sets de imagenes
train_images.shape
len(train_labels)
train_labels


test_images.shape
len(test_labels)
test_labels


#"declaracion" de la arquitectura de la red. Con dos capas (dense), y sus diferenes funciones de activacion (relu y softmax)
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#convertimiento de valores a float
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32')

#dice que viene mas explicado en el libro
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#la parte de entrenamiento donde defino las epocas (iteraciones) que quiero que tenga
network.fit(train_images, train_labels, epochs=100, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

network.save_weights('/home/alvarado/catkin_emiliano/src/assigment_publisher_subscriber/src/trained.h5')

