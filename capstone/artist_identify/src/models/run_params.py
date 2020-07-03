""" All configs for training pipeline. """

BATCH_SIZE = 32

# Builds the run configuration for training and data transformations.
MODEL_TAINING_CONFIG = {
    'EPOCHS': 15,
    'VERBOSITY': 2,
    'BATCH_SIZE': BATCH_SIZE,
    'CHECKPOINT_SUBDIR': 'checkpoint',
    'LOG_SUBDIR': 'logs',
}
