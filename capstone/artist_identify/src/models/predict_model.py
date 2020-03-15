""" Predict output for one input using a saved model """

import logging

import click
import numpy as np
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv
from tensorflow.keras.preprocessing import image

from run_params import DATA_PROCESSING_CONFIG

TOP_N = 3


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('image_path', type=click.Path(exists=True))
def predict_model(model_path, image_path):
    """ Trains a model on the dataset.
    """

    logger = logging.getLogger(__name__)
    logger.info('begin model prediction')

    img = image.img_to_array(image.load_img(
        image_path,
        target_size=DATA_PROCESSING_CONFIG['IMAGE_SIZE']))*DATA_PROCESSING_CONFIG['RESCALE_FACTOR']

    img = np.array([img.astype('float16')])

    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(img, verbose=2)[0]
    print(predictions)

    predicted_class = DATA_PROCESSING_CONFIG['CLASSES'][np.argmax(predictions)]
    print(predicted_class)

    top_n_idx = np.argsort(predictions)[-TOP_N:]
    top_n_values = [(DATA_PROCESSING_CONFIG['CLASSES'][i], predictions[i])
                    for i in top_n_idx[::-1]]
    print(top_n_values)


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    predict_model()
