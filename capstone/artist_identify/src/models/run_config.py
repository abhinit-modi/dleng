# -*- coding: utf-8 -*-
from pathlib import Path
import os
import time

BATCH_SIZE = 32

def get_train_config():
    """ Builds the run configuration for training and data transformations.
    """
    return {
        "EPOCHS": 15,
        "VERBOSITY" : 2,
        "BATCH_SIZE": BATCH_SIZE,
    }

def get_data_config():
    return {
        "DATA_SEED": 1237,
        "DATA_DIR": os.path.join(str(Path(__file__).resolve().parents[2]), 'data', 'processed', 'wikiart_sampled'),
        "CLASSES": ['william-merritt-chase', 'paul-gauguin', 'pablo-picasso', 'utagawa-kuniyoshi', 'raphael-kirchner', 'martiros-saryan', 'rembrandt', 'boris-kustodiev', 'vincent-van-gogh', 'marc-chagall', 'henri-de-toulouse-lautrec', 'nicholas-roerich', 'paul-cezanne', 'alfred-sisley', 'konstantin-makovsky', 'albrecht-durer', 'thomas-eakins', 'joaquã\xadn-sorolla', 'ilya-repin', 'gustave-dore', 'odilon-redon', 'john-singer-sargent', 'zinaida-serebriakova', 'sam-francis', 'maurice-prendergast', 'pierre-auguste-renoir', 'peter-paul-rubens', 'ivan-shishkin', 'claude-monet', 'ernst-ludwig-kirchner', 'francisco-goya', 'edgar-degas', 'amedeo-modigliani', 'camille-corot', 'konstantin-korovin', 'salvador-dali', 'henri-matisse', 'eugene-boudin', 'pyotr-konchalovsky', 'james-tissot', 'ivan-aivazovsky', 'david-burliuk', 'camille-pissarro', 'childe-hassam', 'isaac-levitan'],
        "BATCH_SIZE": BATCH_SIZE,
    }
