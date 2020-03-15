""" All config for model construction """

import datetime
import os

from pathlib import Path

MODEL_NAME = 'resnet_transfer'
ROOT_DIR = Path(__file__).resolve().parents[3]

COMPILE_CONFIG = {
    'OPTIMIZER': 'adam',
    'LOSS': 'sparse_categorical_crossentropy',
    'METRICS': ['accuracy']
}

MODEL_CONFIG = {
    'MODEL_NAME': MODEL_NAME,
    'VERSION_ID': str(0),
    'PRETRAINED_PATH': os.path.join(ROOT_DIR, 'models', 'pretrained', 'resnet_18_no_top'),
    'BASE_PATH': os.path.join(ROOT_DIR, 'models', MODEL_NAME,
                              datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
    'NOTES': "Resnet trasnfer learning, last layer training."
}
