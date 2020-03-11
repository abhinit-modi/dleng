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
def evaluate_model(model_basepath):
    """ Trains a model on the dataset.
    """    
    def _init_data_source():
        data_source = data_generator.prepare_test_data_generator()
        return data_source

    logger = logging.getLogger(__name__)
    logger.info('begin model prediction')

    model = tf.keras.models.load_model(model_basepath)
    data_source = _init_data_source()

    return model.evaluate(data_source, verbose = 2)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    evaluate_model()
