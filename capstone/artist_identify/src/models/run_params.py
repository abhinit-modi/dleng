""" All configs for training pipeline. """

from pathlib import Path
import os


BATCH_SIZE = 32
ROOT_DIR = str(Path(__file__).resolve().parents[2])

# Builds the run configuration for training and data transformations.
MODEL_TAINING_CONFIG = {
    'EPOCHS': 15,
    'VERBOSITY': 2,
    'BATCH_SIZE': BATCH_SIZE,
    'CHECKPOINT_SUBDIR': 'checkpoint',
    'LOG_SUBDIR': 'logs',
}

DATA_PROCESSING_CONFIG = {
    'IMAGE_SIZE': (224, 224),
    'VALIDATION_SPLIT': 0.125,
    'RESCALE_FACTOR': 1./255,
    'DATA_SEED': 1237,
    'DATA_DIR': os.path.join(ROOT_DIR, 'data', 'processed', 'wikiart_sampled'),
    'CLASSES': ['albrecht-durer', 'alfred-sisley', 'amedeo-modigliani', 'boris-kustodiev',
                'camille-corot', 'camille-pissarro', 'childe-hassam', 'claude-monet',
                'david-burliuk', 'edgar-degas', 'ernst-ludwig-kirchner', 'eugene-boudin',
                'francisco-goya', 'gustave-dore', 'henri-de-toulouse-lautrec', 'henri-matisse',
                'ilya-repin', 'isaac-levitan', 'ivan-aivazovsky', 'ivan-shishkin', 'james-tissot',
                'joaquã\xadn-sorolla', 'john-singer-sargent', 'konstantin-korovin',
                'konstantin-makovsky', 'marc-chagall', 'martiros-saryan', 'maurice-prendergast',
                'nicholas-roerich', 'odilon-redon', 'pablo-picasso', 'paul-cezanne', 'paul-gauguin',
                'peter-paul-rubens', 'pierre-auguste-renoir', 'pyotr-konchalovsky',
                'raphael-kirchner', 'rembrandt', 'salvador-dali', 'sam-francis', 'thomas-eakins',
                'utagawa-kuniyoshi', 'vincent-van-gogh', 'william-merritt-chase',
                'zinaida-serebriakova'],
    'BATCH_SIZE': BATCH_SIZE,
}
