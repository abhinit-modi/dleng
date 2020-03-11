# -*- coding: utf-8 -*-
import click
import logging
import os
import time
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import tensorflow as tf

import run_config

def prepare_train_data_generator():
    """ Builds the configuration for the model in this directoty.
    """
    logger = logging.getLogger(__name__)
    data_config = run_config.get_data_config()

    logger.info('Building data generator')
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
            os.path.join(data_config['DATA_DIR'], 'train'),
            target_size=(224, 224),
            batch_size=data_config['BATCH_SIZE'],
            class_mode="sparse")

    return train_generator

def prepare_test_data_generator():
    """ Builds the configuration for the model in this directoty.
    """
    logger = logging.getLogger(__name__)
    data_config = run_config.get_data_config()

    logger.info('Building data generator')
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
            os.path.join(data_config['DATA_DIR'], 'test'),
            target_size=(224, 224),
            batch_size=data_config['BATCH_SIZE'],
            class_mode="sparse")

    return test_generator

def prepare_dataset_generator():
    data_config = run_config.get_data_config()

    SEED = data_config['DATA_SEED']
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    DATASET_SIZE = len(data_config['CLASS_NAMES'])*300

    def standardize(img):
        img = tf.image.per_image_standardization(img)
        return img

    def randomize(img):
        # may be flip and crop (not rescale)
        img = tf.image.random_flip_left_right(img, seed=SEED)
        # img = tf.image.random_crop(img, [IMG_HEIGHT, IMG_WIDTH, 3], 
        #                            seed=SEED, name="random_crop")
        img = tf.image.resize_with_crop_or_pad(img, IMG_HEIGHT, IMG_WIDTH)
        return img

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return tf.reduce_min(tf.where(tf.equal(data_config['CLASS_NAMES'], parts[-2])))

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        img = standardize(img)
        img = randomize(img)
        return img, label

    list_ds = tf.data.Dataset.list_files(os.path.join(data_config['DATA_DIR'], "*/*"))
    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    BATCH_SIZE = 32
    NUM_EPOCHS = 100

    train_ds = labeled_ds.take(int(0.85 * DATASET_SIZE))
    test_ds = labeled_ds.skip(int(0.85 * DATASET_SIZE))

    train_ds = train_ds.shuffle(1000000).batch(batch_size=BATCH_SIZE).repeat(NUM_EPOCHS)
    test_ds = test_ds.batch(batch_size=BATCH_SIZE)

    return train_ds, test_ds

