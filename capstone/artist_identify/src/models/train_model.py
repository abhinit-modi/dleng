# -*- coding: utf-8 -*-
import click
import json
import logging
import os
import tensorflow as tf
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import data_generator
import run_config

@click.command()
@click.argument('model_basepath', type=click.Path(exists=True))
def train_model(model_basepath):
    """ Trains a model on the dataset.
    """
    with open(os.path.join(model_basepath, 'model_config.json')) as config_json:
        model_config = json.load(config_json)
    
    def _init_data_source():
        data_source = data_generator.prepare_data_generator()
        return data_source

    def _build_callbacks():
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=10, verbose=1)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath = os.path.join(model_config['BASE_PATH'], 'checkpoint'),
                save_best_only = True,
                monitor = 'val_acc',
                verbose = 2,
                mode='max')

        log_dir = os.path.join(model_config['BASE_PATH'], 'logs')
        file_writer = tf.summary.create_file_writer(log_dir)
        file_writer.set_as_default()
        tensor_board = tf.keras.callbacks.TensorBoard(log_dir)

        return [early_stop, checkpoint_callback, tensor_board]

    logger = logging.getLogger(__name__)
    logger.info('begin model training')

    try:
        model = tf.keras.model.load(os.path.join(model_config['BASE_PATH'], 'snapshot'))
    except:
        model = tf.keras.model.load(os.path.join(model_config['BASE_PATH'], 'checkpoint'))

    train_config = run_config.get()
    data_source = _init_data_source()

    model.fit(data_source, epochs = train_config["EPOCHS"],
                     callbacks = _build_callbacks(),
                     verbose = train_config["VERBOSITY"])
    
    logger.info('saving model')
    model.save(os.path.join(model_config['BASE_PATH'], 'snapshot'), save_format='tf')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    train_model()
