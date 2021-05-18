""" All config for model construction and compilation """
import tensorflow as tf

COMPILE_CONFIG = {
    'optimizer': 'adam',
    'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'metrics': ['accuracy']
}

MODEL_CONFIG = {
    'input_dim': (1, 784),
    'fc1_dim': 1024,
    'fc2_dim': 128,
    'labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}
