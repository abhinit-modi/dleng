# -*- coding: utf-8 -*-

import datetime
import os

from pathlib import Path
from time import strftime

MODEL_NAME = 'baseline'

def get_compile_config():
    return {
        'OPTIMIZER': 'adam',
        'LOSS': 'sparse_categorical_crossentropy',
        'METRICS': ['accuracy']
    }

def get_model_config():
    return {
        'MODEL_NAME': MODEL_NAME,
        'BASE_PATH': os.path.join(Path(__file__).resolve().parents[3], 'models', MODEL_NAME, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
        'NOTES': "As per paper baseline."
    }