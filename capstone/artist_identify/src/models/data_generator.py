""" Data input functions for model training and testing. """

import logging
import os

import tensorflow as tf

from run_params import DATA_PROCESSING_CONFIG

def prepare_train_data_generator():
    """ Builds the configuration for the model in this directoty.
    """
    logger = logging.getLogger(__name__)

    logger.info('Building data generator')

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=DATA_PROCESSING_CONFIG['RESCALE_FACTOR'],
        validation_split=DATA_PROCESSING_CONFIG['VALIDATION_SPLIT'],
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_PROCESSING_CONFIG['DATA_DIR'], 'train'),
        shuffle=True,
        target_size=DATA_PROCESSING_CONFIG['IMAGE_SIZE'],
        batch_size=DATA_PROCESSING_CONFIG['BATCH_SIZE'],
        class_mode='sparse',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_PROCESSING_CONFIG['DATA_DIR'], 'train'),
        target_size=DATA_PROCESSING_CONFIG['IMAGE_SIZE'],
        batch_size=DATA_PROCESSING_CONFIG['BATCH_SIZE'],
        class_mode='sparse',
        subset='validation')

    return train_generator, validation_generator


def prepare_test_data_generator():
    """ Builds the configuration for the model in this directoty.
    """
    logger = logging.getLogger(__name__)

    logger.info('Building data generator')
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=DATA_PROCESSING_CONFIG['RESCALE_FACTOR'])

    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_PROCESSING_CONFIG['DATA_DIR'], 'test'),
        target_size=DATA_PROCESSING_CONFIG['IMAGE_SIZE'],
        batch_size=DATA_PROCESSING_CONFIG['BATCH_SIZE'],
        class_mode='sparse')

    return test_generator
