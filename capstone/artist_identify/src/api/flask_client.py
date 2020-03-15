""" Simple client to connect to flask server """

import base64
import logging

import click
import requests
from dotenv import find_dotenv, load_dotenv

# defining the flask-api-endpoint
API_ENDPOINT = "http://localhost:5000/artistclassifier/predict/"


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
def request_prediction(image_path):
    """ Read the image, send bytes to flas server for prediction.
    """
    b64_image = ""
    # Encoding the JPG,PNG,etc. image to base64 format
    with open(image_path, "rb") as image_file:
        b64_image = base64.b64encode(image_file.read())

    # data to be sent to api
    data = {'b64': b64_image}

    # sending post request and saving response as response object
    req = requests.post(url=API_ENDPOINT, data=data)

    # extracting the response
    print("{}".format(req.text))


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    request_prediction()
