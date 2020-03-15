""" Build and compile model """

import json
import logging
import os

from dotenv import find_dotenv, load_dotenv
from tensorflow.keras import layers, models

from model_params import COMPILE_CONFIG, MODEL_CONFIG


def make_model():
    """ Builds a model using the hyperparameters from the config file.
        Saves the empty model into the output path - Both training
        graph and weights which may/may not be initialized with earlier
        iterations.
    """
    def _compile_model(model):
        if model is None:
            raise TypeError
        model.compile(optimizer=COMPILE_CONFIG['OPTIMIZER'],
                      loss=COMPILE_CONFIG['LOSS'],
                      metrics=COMPILE_CONFIG['METRICS'])
        return model

    def _build_model(pretrained_path):
        base_model = models.load_model(pretrained_path)
        base_model.trainable = False

        global_average_layer = layers.GlobalAveragePooling2D()

        fc_layer = layers.Dense(units=224, activation="relu")
        prediction_layer = layers.Dense(units=45, activation="softmax")

        model = models.Sequential([
            base_model,
            global_average_layer,
            fc_layer,
            prediction_layer
        ])
        return model

    logger = logging.getLogger(__name__)

    # Create and save model config
    os.makedirs(MODEL_CONFIG['BASE_PATH'])
    with open(os.path.join(MODEL_CONFIG['BASE_PATH'], 'model_config.json'),
              'w+', encoding='utf-8') as config_file:
        json.dump(MODEL_CONFIG, config_file, ensure_ascii=False, indent=4)

    # Create and compile model
    model = _build_model(MODEL_CONFIG['PRETRAINED_PATH'])
    model = _compile_model(model)
    logger.info(model.summary())

    # Save model
    model.save(os.path.join(
        MODEL_CONFIG['BASE_PATH'], MODEL_CONFIG['VERSION_ID']), save_format='tf')
    return MODEL_CONFIG['BASE_PATH']


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    make_model()
