"""" Simple client to docker server for tensorflow model. """

import json
import logging

import click
import numpy as np
import requests
from dotenv import find_dotenv, load_dotenv
from tensorflow.keras.preprocessing import image

from interface_params import CONTRACT_CONFIG, SERVICE_CONFIG


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
def request_prediction(image_path):
    """ Read the image, send bytes to tf server for prediction.
    """
    # Preprocessing our input image
    img = image.img_to_array(image.load_img(
        image_path, target_size=CONTRACT_CONFIG['IMAGE_SIZE'])) * CONTRACT_CONFIG['RESCALE_FACTOR']

    img = img.astype('float16')

    payload = {
        "instances": [img.tolist()]
    }

    # sending post request to TensorFlow Serving server
    resp = requests.post(
        SERVICE_CONFIG['API_ENDPOINT'], json=payload)
    pred = json.loads(resp.content.decode('utf-8'))
    print(np.array(pred['predictions']))

    # Decoding the response
    predicted_class = CONTRACT_CONFIG['CLASSES'][np.argmax(pred['predictions'], axis=1)[0]]
    print(predicted_class)


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    request_prediction()
