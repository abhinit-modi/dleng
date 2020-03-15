""" Evaluate model with test dataset. """

import logging

import click
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv

import data_generator


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
def evaluate_model(model_path):
    """ Evaluates model on the dataset.
    """
    def _init_data_source():
        data_source = data_generator.prepare_test_data_generator()
        return data_source

    logger = logging.getLogger(__name__)
    logger.info('begin model prediction')

    model = tf.keras.models.load_model(model_path)
    data_source = _init_data_source()

    return model.evaluate(data_source, verbose=2)


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    evaluate_model()
