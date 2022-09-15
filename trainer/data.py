#!/usr/bin/env python3

"""Load image files and labels

This file contains the method that creates data and labels from a directory.
"""
import os
from pathlib import Path

import numpy as np
import cv2

import tensorflow as tf


def create_data_with_labels(dataset_dir,train=True):
    """Gets numpy data and label array from images that are in the folders
    that are in the folder which was given as a parameter. The folders
    that are in that folder are identified by the mug they represent and
    the folder name starts with the label.

    Parameters:
        dataset_dir: A string specifying the directory of a dataset
        train: A boolean value used for augmentation of training dataset
    Returns:
        data: A numpy array containing the images
        labels: A numpy array containing labels corresponding to the images
    """
    image_paths_per_label = collect_paths_to_files(dataset_dir)

    images = []
    labels = []
    for label, image_paths in image_paths_per_label.items():
        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)

    data = np.array([preprocess_image(image.astype(np.float32),train)
                     for image in images])
    labels = np.array(labels)

    return data, labels

def collect_paths_to_files(dataset_dir):
    """Returns a dict with labels for each subdirectory of the given directory
    as keys and lists of the subdirectory's contents as values.

    Parameters:
        dataset_dir: A string containing the path to a directory containing
            subdirectories to different classes.
    Returns:
        image_paths_per_label: A dict with labels as keys and lists of file
        paths as values.
    """
    dataset_dir = Path(dataset_dir)
    mug_dirs = [f for f in sorted(os.listdir(dataset_dir)) if not f.startswith('.')]
    image_paths_per_label = {
        label: [
            dataset_dir / mug_dir / '{0}'.format(f)
            for f in os.listdir(dataset_dir / mug_dir) if not f.startswith('.')
        ]
        for label, mug_dir in enumerate(mug_dirs)
    }
    return image_paths_per_label

def preprocess_image(image,train=True):
    """Returns a preprocessed image.

    Parameters:
        image: A RGB image with pixel values in range [0, 255].
        train: A boolean value used for augmenting training data.
    Returns
        image: The preprocessed image.
    """
    image = image / 255.
    if train:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        seed = (1, 2)
        image = tf.image.stateless_random_brightness(image, 0.1,seed)
        image = tf.image.random_saturation(image, 2, 4)
        temp=np.random.randint(200, size=1)[0]
        if temp % 2 ==0:
          image = tf.image.central_crop(image, np.random.uniform(low=.4, high=1.))
        #image = tf.image.random_crop(value=image, size=(50, 50, 3))

    image = tf.image.resize(image,[224,224])

    return image
