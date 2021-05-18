""" All config for dataset preparation """

import os

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_DOWNLOAD_CONFIG = {
    'DOWNLOAD_DIR': os.path.join(ROOT_DIR, "data", "raw", "mnist"),
    'TRAIN_IMAGES': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'TRAIN_LABELS': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'TEST_IMAGES': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'TEST_LABELS': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
}

MNIST_DATA_CONFIG = {
    'input_shape': (1, 784),
    'output_shape': (1, ),
    'shuffle': 5000,
    'batch_size': 32,
}
