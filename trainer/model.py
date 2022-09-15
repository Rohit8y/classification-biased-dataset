#!/usr/bin/env python3

"""Model to classify mugs

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value.
    """
    return 16

def get_epochs():
    """Returns number of epochs that will be used by your solution.
    """
    return 20

def get_lr():
    """Returns the initial learning rate for training.
    """
    return 0.05

def solution(input_layer):
    """Returns a compiled model.

    This function is expected to return a model to identity the different mugs.
    The model's outputs are expected to be probabilities for the classes and
    and it should be ready for training.
    The input layer specifies the shape of the images. The preprocessing
    applied to the images is specified in data.py.

    Add your solution below.

    Parameters:
        input_layer: A tf.keras.layers.InputLayer() specifying the shape of the input.
            RGB colored images, shape: (width, height, 3)
    Returns:
        model: A compiled model
    """

    # Code of building model
    model = Sequential()

    # Loading pretrained weights
    pretrained_model = tf.keras.applications.vgg16.VGG16(include_top=False,weights='imagenet',
                                                    input_shape=(224,224,3),pooling='avg',classes=4)
    # Freezing layers for fine-tuning
    for layer in pretrained_model.layers:
        layer.trainable=False

    model.add(pretrained_model)
    # Adding Dense layer with softmax activation to get probability of the output classes
    model.add(Dense(4, activation='softmax'))

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.SGD(get_lr(),momentum=0.7),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model
