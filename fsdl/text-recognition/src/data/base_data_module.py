"""Base DataModule class."""
from typing import Dict, Tuple, Optional
import argparse
import tensorflow as tf

BATCH_SIZE = 128


class BaseDataModule:
    """
    Base class to handle all data operations like
    download; prepare train, test; transformations etc.
    """

    def __init__(self, args: argparse.Namespace = None, config: Dict = {}) -> None:
        self.args = vars(args) if args is not None else {}
        self.data_config = config
        self.input_dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.data_train: Tuple[tf.data.Dataset, tf.data.Dataset]
        self.data_val: Tuple[tf.data.Dataset, tf.data.Dataset]
        self.data_test: Tuple[tf.data.Dataset, tf.data.Dataset]

    @staticmethod
    def add_to_argparse(parser):
        """Command line modify-able config params."""
        parser.add_argument(
            "--batch_size",
            type=int,
            help="Number of examples to operate on per forward step."
        )
        return parser

    def config(self):
        """
        Return important settings of the dataset, which will be passed to instantiate models.
        """
        return self.data_config

    def download_data(self, config: Optional[Dict] = None):
        """
        Download data inside data dir
        """

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Split into train, val, test, and set dims.
        Should assign `tf.data.Dataset` objects to self.data_train, self.data_val,
        and self.data_test.
        """

    def train_data_loader(self) -> tf.data.Dataset:
        """
        Return tf.data.Dataset or tf.data.DataFrameIterator
        """
        return self.data_train

    def val_data_loader(self) -> tf.data.Dataset:
        """
        Return tf.data.Dataset or tf.data.DataFrameIterator
        """
        return self.data_val

    def test_data_loader(self) -> tf.data.Dataset:
        """
        Return tf.data.Dataset or tf.data.DataFrameIterator
        """
        return self.data_test


def load_module_and_print_info(data_module_class: object) -> BaseDataModule:
    """Load dataset and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    data_module = data_module_class(args)
    data_module.download_data()
    data_module.prepare_data()
    print(data_module.config())
    return data_module
