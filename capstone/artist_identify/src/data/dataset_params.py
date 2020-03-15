""" All config for dataset generation """

import os

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_GENERATE_CONFIG = {
    'INPUT_DIR': os.path.join(ROOT_DIR, "data", "raw", "wikiart"),
    'OUTPUT_DIR': os.path.join(ROOT_DIR, "data", "processed", "wikiart_sampled"),
    'NUM_SAMPLES': 300,
    'SPLIT_RATIO': 0.8,
}
