"""MNIST DataModule class."""
from typing import Dict, Optional
import argparse
import os
import tensorflow as tf
import numpy as np

from src.data.base_data_module import BaseDataModule, load_module_and_print_info
from src.data.params import MNIST_DATA_CONFIG


class MNISTDataModule(BaseDataModule):
    """MNIST dataset manager."""
    def __init__(self, args: argparse.Namespace = None, config=MNIST_DATA_CONFIG) -> None:
        super().__init__(args)
        self.args = vars(args) if args is not None else {}
        self.data_config = config
        self.transformation = lambda ds: ds.reshape(-1, 784).astype("float32") / 255.0

    def download_data(self, config: Optional[Dict] = None) -> str:
        """Download mnist dataset locally."""
        if not config:
            return ''
        if not os.path.exists(os.path.abspath('.') + config['DOWNLOAD_DIR']):
            image_zip = tf.keras.utils.get_file('mnist_train_images.zip',
                                                cache_subdir=os.path.abspath('.'),
                                                origin=config['TRAIN_IMAGES'],
                                                extract=True)
            train_images_dir = os.path.dirname(image_zip) + config['DOWNLOAD_DIR']
            os.remove(image_zip)
        else:
            train_images_dir = os.path.abspath('.') + config['DOWNLOAD_DIR']
        return train_images_dir

    def prepare_data(self, *args, **kwargs) -> None:
        """Prepare train and test datasets."""
        train, test = tf.keras.datasets.mnist.load_data()
        x_train, y_train = train
        x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
        y_train = y_train.astype(np.int32)

        x_test, y_test = test
        x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
        y_test = y_test.astype(np.int32)

        self.data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
            self.data_config['shuffle']).batch(self.data_config['batch_size'])
        self.data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(
            self.data_config['shuffle']).batch(self.data_config['batch_size'])
        print({'train_size': self.data_train.cardinality(),
               'test_size': self.data_test.cardinality()})


if __name__ == "__main__":
    load_module_and_print_info(MNISTDataModule)
