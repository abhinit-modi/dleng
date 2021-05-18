"""Module to create models"""

import tensorflow as tf


def build_and_compile_model(model_module: object) -> tf.keras.Model:
    """Initialize a model given its module path"""
    return model_module.initialize()
