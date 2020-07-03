""" Do model training """

import json
import logging
import os

import click
import matplotlib.pyplot as plt
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv

from src.features import build_features
from run_params import MODEL_TAINING_CONFIG


def plot_performance(history):
    """ Plot curve of accuracy vs epoch for training and validation set.
    """
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')


@click.command()
@click.argument('model_basepath', type=click.Path(exists=True))
def train_model(model_basepath):
    """ Trains a model on the dataset.
    """
    with open(os.path.join(model_basepath, 'model_config.json')) as config_json:
        model_config = json.load(config_json)

    def _init_data_source():
        train_source, val_source = build_features.train_data_generator()
        return train_source, val_source

    def _build_callbacks():
        callbacks = []
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', min_delta=1e-4, patience=10,
            verbose=MODEL_TAINING_CONFIG['VERBOSITY'])
        callbacks.append(early_stop)

        if MODEL_TAINING_CONFIG['CHECKPOINT_SUBDIR']:
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    model_config['BASE_PATH'], MODEL_TAINING_CONFIG['CHECKPOINT_SUBDIR']),
                monitor='val_accuracy',
                verbose=MODEL_TAINING_CONFIG['VERBOSITY'])
            callbacks.append(checkpoint_callback)

        if MODEL_TAINING_CONFIG['LOG_SUBDIR']:
            log_dir = os.path.join(
                model_config['BASE_PATH'], MODEL_TAINING_CONFIG['LOG_SUBDIR'])
            file_writer = tf.summary.create_file_writer(log_dir)
            file_writer.set_as_default()
            tensor_board = tf.keras.callbacks.TensorBoard(log_dir)
            callbacks.append(tensor_board)

        return callbacks

    logger = logging.getLogger(__name__)
    logger.info('begin model training')

    model = tf.keras.models.load_model(os.path.join(
        model_config['BASE_PATH'], model_config['VERSION_ID']))
    train_source, val_source = _init_data_source()

    history = model.fit(train_source, epochs=MODEL_TAINING_CONFIG['EPOCHS'],
                        callbacks=_build_callbacks(),
                        verbose=MODEL_TAINING_CONFIG['VERBOSITY'],
                        validation_data=val_source)

    # plot_performance(history)

    logger.info('saving model')
    model.save(os.path.join(
        model_config['BASE_PATH'], model_config['VERSION_ID']), save_format='tf')
    return model_config['BASE_PATH']


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    train_model()
