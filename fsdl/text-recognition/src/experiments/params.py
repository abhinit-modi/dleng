""" All configs for training pipeline. """

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Builds the run configuration for training and data transformations.
TRAIN_CONFIG = {
    'data_class': 'mnist_data_module.MNISTDataModule',
    'model_config': {
        'name': 'mlp.model',
        'save_dir': os.path.join(ROOT_DIR, 'models', 'mlp.model', 'lab1'),
        'load_dir': os.path.join(ROOT_DIR, 'models', 'mlp.model', 'lab1'),
    },
    'train_config': {
        'epochs': 15,
        'verbosity': 2,
        'log_subdir': 'logs',
        'checkpoint_subdir': 'checkpoint',
        'base_subdir': '2021-05-18_17:14:52',
        'run_eval': True,
    }
}
