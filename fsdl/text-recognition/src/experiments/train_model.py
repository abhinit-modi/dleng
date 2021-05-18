""" Do model training """

import datetime
import logging
import os
import importlib
from typing import Dict
import tensorflow as tf
from src.experiments.params import TRAIN_CONFIG, ROOT_DIR
from src.data.base_data_module import BaseDataModule, load_module_and_print_info
from src.models.init_model import build_and_compile_model


def _import_module_(module_name: str) -> type:
    """Import module from name, e.g. 'src.models.mlp.model'"""
    module = importlib.import_module(module_name)
    return module


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'src.models.mlp.model.MLPModel'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


class Trainer:
    """
    Functions to run a training experiment i.e
    a. Load dataset
    b. Build a model
    c. Train the model
    d. Evaluate the model
    """
    def __init__(self, config: Dict):
        self.config = config
        self.data_module: BaseDataModule
        self.model: tf.keras.Model

    def init(self) -> None:
        """Initialize training environment."""
        self.init_data_module()
        self.init_working_dir()
        self.init_model()

    def init_data_module(self) -> None:
        """Initialize data loaders."""
        self.data_module = load_module_and_print_info(
            _import_class('src.data.{0}'.format(self.config['data_class'])))

    def get_working_path(self, subdir) -> str:
        """Get working directory."""
        return os.path.join(self.config['working_dir'], subdir)

    def init_model(self) -> None:
        """Initialize model."""
        if self.config['model_config']['load_dir']:
            self.model = tf.keras.models.load_model(self.config['model_config']['load_dir'])
        else:
            self.model = build_and_compile_model(
                _import_module_('src.models.{0}'.format(self.config['model_config']['name'])))

    def init_working_dir(self) -> None:
        """Initialize working directory."""
        if not self.config['train_config']['base_subdir']:
            base_path = os.path.join(ROOT_DIR, 'models', self.config['model_config']['name'],
                                     datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
            os.makedirs(base_path)
            self.config['working_dir'] = base_path
        else:
            base_path = os.path.join(ROOT_DIR, 'models', self.config['model_config']['name'],
                                     self.config['train_config']['base_subdir'])
            self.config['working_dir'] = base_path

    def get_callbacks(self) -> list:
        """Prepare training callbacks."""
        callbacks = []
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy', min_delta=1e-4, patience=10,
            verbose=self.config['train_config']['verbosity'])
        callbacks.append(early_stop)

        if self.config['train_config']['checkpoint_subdir']:
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.get_working_path(self.config['train_config']['checkpoint_subdir']),
                monitor='val_accuracy',
                verbose=self.config['train_config']['verbosity'])
            callbacks.append(checkpoint_callback)

        if self.config['train_config']['log_subdir']:
            log_dir = self.get_working_path(self.config['train_config']['log_subdir'])
            file_writer = tf.summary.create_file_writer(log_dir)
            file_writer.set_as_default()
            tensor_board = tf.keras.callbacks.TensorBoard(log_dir)
            callbacks.append(tensor_board)
        return callbacks

    def run(self) -> object:
        """Run experiment."""
        logger = logging.getLogger(__name__)
        logger.info('begin model training')

        history = self.model.fit(self.data_module.train_data_loader(),
                                 epochs=self.config['train_config']['epochs'],
                                 callbacks=self.get_callbacks(),
                                 verbose=self.config['train_config']['verbosity'])

        if self.config['model_config']['save_dir']:
            logger.info('saving model')
            self.model.save(self.config['model_config']['save_dir'], save_format='tf')

        if self.config['train_config']['run_eval']:
            self.model.evaluate(self.data_module.test_data_loader(),
                                verbose=self.config['train_config']['verbosity'])

        return history


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    trainer = Trainer(TRAIN_CONFIG)
    trainer.init()
    print(trainer.config)
    trainer.run()
