#!/usr/bin/env python
from keras.datasets import mnist
from keras import models, layers
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D
from keras.layers import Conv2D, BatchNormalization
from keras.utils import to_categorical
import cv2
import os
import numpy as np
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator

batch_size = 1
load_my_database = True # Modificable
load_mnist_database = False # Modificable

def build_net():
    ''' Aqui se arma la arquitectura de la red'''
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28, 28, 1)))
    network.add(layers.Dense(10, activation='softmax'))
    network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return network

''' Aqui declaramos los directorios '''
base_dir = '/home/alvarado/catkin_emiliano/Imagenes'
base_cuatro = os.path.join(base_dir, '4-images')
base_cinco = os.path.join(base_dir, '5-images')
base_cero = os.path.join(base_dir, '0-images')

# network.load_weights(base_dir + 'my-test-model.h5')

if load_mnist_database:
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

if load_my_database:
    ''' Aqui cargamos las imagenes de nuestra base de datos'''

    # Aqui las del cuatro
    no_de_imagenes = 93 # Modificable
    no_inicial = 1 # Modificable
    my_train_images = np.zeros((no_de_imagenes, 28, 28))
    my_train_labels = np.zeros((no_de_imagenes))
    for i in range(0, no_de_imagenes):
        img = cv2.imread((base_cuatro + ('/%03d' % (i+no_inicial)) + '.png'), 0)
        my_train_images[i, : , ] = crop_img
        my_train_labels[i] = 1

    # Aqui las del cinco
    no_de_imagenes = 99 # Modificable
    no_inicial = 1 # Modificable
    my_train_images_ = np.zeros((no_de_imagenes, 28, 28))
    my_train_labels_ = np.zeros((no_de_imagenes))
    for i in range(0, no_de_imagenes):
        img = cv2.imread((base_cinco + ('/%03d' % (i+no_inicial)) + '.png'), 0)
        my_train_images_[i, :, :] = crop_img
        my_train_labels_[i] = 2

    my_train_images = np.concatenate((my_train_images_, my_train_images))
    my_train_labels = np.concatenate((my_train_labels_, my_train_labels))

    # Aqui las del cero
    no_de_imagenes = 206 # Modificable
    no_inicial = 1 # Modificable
    my_train_images_ = np.zeros((no_de_imagenes, 28, 28))
    my_train_labels_ = np.zeros((no_de_imagenes))
    for i in range(0, no_de_imagenes):
        img = cv2.imread((base_cero + ('/%03d' % (i+no_inicial)) + '.png'), 0)
        my_train_images_[i, :, :] = crop_img
        my_train_labels_[i] = 0

    my_train_images = np.concatenate((my_train_images_, my_train_images))
    my_train_labels = np.concatenate((my_train_labels_, my_train_labels))

    """Revolver la lista de datos"""
    indices = np.arange(my_train_labels.shape[0])
    np.random.shuffle(indices)
    my_train_images = my_train_images[indices, :, :]
    my_train_labels = my_train_labels[indices]

    ''' Dividir la base de datos en entrenamiento y validacion '''

    k = int(np.round(my_train_images.shape[0] * 0.8))
    my_test_images = my_train_images[k:, :, :]
    my_train_images = my_train_images[:k, :, :]
    my_test_labels = my_train_labels[k:]
    my_train_labels = my_train_labels[:k]

    if not load_mnist_database:

        train_images = my_train_images
        train_labels = my_train_labels
        test_images = my_test_images
        test_labels = my_test_labels

    else:

        train_images = np.concatenate((train_images, my_train_images))
        train_labels = np.concatenate((train_labels, my_train_labels))
        test_images = np.concatenate((test_images, my_test_images))
        test_labels = np.concatenate((test_labels, my_test_labels))

''' Aqui cambiamos la forma de las imagenes y las etiquetas para adaptarlas a la red'''

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_images = train_images.astype('float32')

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_images = test_images.astype('float32')

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


''' Aqui declaramos las modificaciones deseadas a las imagenes para expandir la base de datos y entrenar la red'''

train_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    )

test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen.fit(train_images)
# configure batch size and retrieve one batch of images
# for X_batch, y_batch in train_datagen.flow(train_images, train_labels, batch_size=9):
# 	# create a grid of 3x3 images
# 	for i in range(0, 9):
# 		pyplot.subplot(330 + 1 + i)
# 		pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
# 	# show the plot
# 	pyplot.show()
# 	break

''' Aqui declaramos los generadores de datos que alimentan la red'''
train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
validation_generator = test_datagen.flow(test_images, test_labels, batch_size=batch_size)

''' Se compila la red y se entrena'''
network = build_net()
network.fit_generator(train_generator, steps_per_epoch=train_images.shape[0]//batch_size,
                      validation_data=validation_generator, validation_steps=test_images.shape[0]//batch_size,
                      epochs=10)

''' Se valida la red'''
test_loss, test_acc = network.evaluate(test_images, test_labels)

print("Restored model, accuracy: {:5.2f}%".format(100*test_acc))


# imagen_test = train_images[936, :, :]
#print("Real: {}".format(train_images[100]))
# cv2.imshow('mnist', imagen_test)
# cv2.waitKey(0)
#
#
# cv2.imwrite('mnist1.png', imagen_test)

# imagen_test = imagen_test.reshape((1, 28 * 28))
# imagen_test = imagen_test.astype('float32') / 255
#
# cv2.imshow('mnist', imagen_test)
# cv2.waitKey(0)

# resultado = network.predict_classes(imagen_test)

# print("Resultado: {}".format(resultado))

# test_images = test_images.reshape((10000, 28 * 28))
# test_images = test_images.astype('float32') / 255
# test_labels = to_categorical(test_labels)
#
#
# test_loss, test_acc = network.evaluate(test_images, test_labels)
#
# print("Restored model, accuracy: {:5.2f}%".format(100*test_acc))

#cv2.destroyAllWindows()

network.save_weights('/home/alvarado/catkin_emiliano/src/assigment_publisher_subscriber/src/eje.h5')
