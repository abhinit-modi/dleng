""" All config for model construction """

import datetime
import os

from pathlib import Path

MODEL_NAME = 'baseline'
ROOT_DIR = Path(__file__).resolve().parents[3]

COMPILE_CONFIG = {
    'OPTIMIZER': 'adam',
    'LOSS': 'sparse_categorical_crossentropy',
    'METRICS': ['accuracy']
}

MODEL_CONFIG = {
    'MODEL_NAME': MODEL_NAME,
    'VERSION_ID': str(0),
    'BASE_PATH': os.path.join(ROOT_DIR, 'models', MODEL_NAME,
                              datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
    'NOTES': "As per paper baseline. Avoided rescale and added center crop instead. \
        Reduced val split. Added batch norm."
}
