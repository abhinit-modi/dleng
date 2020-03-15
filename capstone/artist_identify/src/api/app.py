""" Proxy flask server """

import base64
import json
from io import BytesIO

import numpy as np
import requests
from flask import Flask, request

from interface_params import CONTRACT_CONFIG, SERVICE_CONFIG
from keras.preprocessing import image

# from flask_cors import CORS

app = Flask(__name__)

# Uncomment this line if you are making a Cross domain request
# CORS(app)

@app.route('/artistclassifier/predict/', methods=['POST'])
def image_classifier():
    """ Send/Forward request to tensorflow serving instance.
    """
    # Decoding and pre-processing base64 image
    img = image.img_to_array(
        image.load_img(BytesIO(base64.b64decode(request.form['b64'])),
                       target_size=CONTRACT_CONFIG['IMAGE_SIZE']))/CONTRACT_CONFIG['RESCALE_FACTOR']

    img = img.astype('float16')

    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [img.tolist()]
    }

    # sending post request to TensorFlow Serving server
    req = requests.post(
        SERVICE_CONFIG['API_ENDPOINT'], json=payload)
    pred = json.loads(req.content.decode('utf-8'))

    # Decoding the response
    predicted_class = CONTRACT_CONFIG['CLASSES'][np.argmax(pred['predictions'], axis=1)[0]]
    return predicted_class
