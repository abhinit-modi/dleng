# -*- coding: utf-8 -*-

import json
import os
import tensorflow as tf
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from tensorflow.keras import layers, models
import model_params

def make_model():
    """ Builds a model using the hyperparameters from the config file.
        Saves the empty model into the output path - Both training
        graph and weights which may/may not be initialized with earlier
        iterations.
    """
    def _compile_model(model):
        if model is None:
            raise TypeError

        compile_config = model_params.get_compile_config()
        model.compile(optimizer=compile_config['OPTIMIZER'],
                        loss=compile_config['LOSS'],
                        metrics=compile_config['METRICS'])
        return model


    def _build_model():
        model = models.Sequential()
        model.add(layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(2, 2), input_shape=(224, 224, 3), activation="relu", padding="same")) # check for stride
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same")) # check for stride
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=6272, activation="relu"))
        model.add(layers.Dense(units=228, activation="relu"))
        model.add(layers.Dense(units=45, activation="softmax")) # Number of classes
        return model

    logger = logging.getLogger(__name__)

    # Create and save model config
    model_config = model_params.get_model_config()
    os.makedirs(model_config['BASE_PATH'])
    with open(os.path.join(model_config['BASE_PATH'], 'model_config.json'), 'w+', encoding='utf-8') as f:
        json.dump(model_config, f, ensure_ascii=False, indent=4)

    # Create and compile model
    model = _build_model()
    model = _compile_model(model)
    logger.info(model.summary())

     # Save model
    model.save(os.path.join(model_config['BASE_PATH'], 'snapshot'), save_format='tf')
    
    return model_config['BASE_PATH']


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    make_model()
