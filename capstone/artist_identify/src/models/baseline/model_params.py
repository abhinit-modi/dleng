# -*- coding: utf-8 -*-
import click
import logging
import os
import time
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def create_model_config():
    """ Builds the configuration for the model in this directoty.
    """
    logger = logging.getLogger(__name__)
    model_name = Path(__file__).resolve().parents[1]
    logger.info('Building config for model {0}'.format(model_name))
    t = time.time()
    return {
        'MODEL_NAME': model_name,
        'BASE_PATH': os.path.join(Path(__file__).resolve().parents[3], 'models', model_name, int(t)),
        'NOTES': "As per paper baseline with 1 FC removed."
    }


def create_compile_config():
    return {
        'OPTIMIZER': 'adam',
        'LOSS': 'sparse_categorical_crossentropy',
        'METRICS': ['accuracy']
    }
