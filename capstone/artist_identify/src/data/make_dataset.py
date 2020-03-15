""" Prepare training and test data """

import logging
import os
import random
import shutil

from dotenv import find_dotenv, load_dotenv

from dataset_params import DATASET_GENERATE_CONFIG


def make_dataset():
    """ Runs data processing scripts to turn raw data from INPUT_DIR into
        cleaned data ready to be analyzed (saved in OUTPUT_DIR).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data %s', DATASET_GENERATE_CONFIG['INPUT_DIR'])

    if os.path.isdir(DATASET_GENERATE_CONFIG['OUTPUT_DIR']):
        shutil.rmtree(DATASET_GENERATE_CONFIG['OUTPUT_DIR'])
    os.mkdir(DATASET_GENERATE_CONFIG['OUTPUT_DIR'])
    os.mkdir(os.path.join(DATASET_GENERATE_CONFIG['OUTPUT_DIR'], "train"))
    os.mkdir(os.path.join(DATASET_GENERATE_CONFIG['OUTPUT_DIR'], "test"))

    dirs = os.listdir(DATASET_GENERATE_CONFIG['INPUT_DIR'])
    fmap = {}
    for genre_dir in dirs:
        if not os.path.isdir(os.path.join(DATASET_GENERATE_CONFIG['INPUT_DIR'], genre_dir)):
            continue
        files = os.listdir(os.path.join(DATASET_GENERATE_CONFIG['INPUT_DIR'], genre_dir))
        for im_file in files:
            author = im_file[:im_file.find('_')]
            if author not in fmap:
                fmap[author] = []

            fmap[author].append(
                os.path.join(DATASET_GENERATE_CONFIG['INPUT_DIR'], genre_dir, im_file),
                os.path.join(DATASET_GENERATE_CONFIG['OUTPUT_DIR'], '{split}', author,
                             im_file[im_file.find('_')+1:]))

    for author in fmap:
        if len(fmap[author]) < DATASET_GENERATE_CONFIG['NUM_SAMPLES']:
            continue

        os.mkdir(os.path.join(DATASET_GENERATE_CONFIG['OUTPUT_DIR'], "train", author))
        os.mkdir(os.path.join(DATASET_GENERATE_CONFIG['OUTPUT_DIR'], "test", author))

        random.shuffle(fmap[author])

        copied = set()
        for index, images in enumerate(fmap[author]):
            if images[1] in copied:
                continue
            copied.add(images[1])
            shutil.copyfile(images[0], images[1].format(
                split="train" if index*1.0/DATASET_GENERATE_CONFIG['NUM_SAMPLES'] <= DATASET_GENERATE_CONFIG['SPLIT_RATIO'] else "test"))
            if len(copied) == DATASET_GENERATE_CONFIG['NUM_SAMPLES']:
                break


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    make_dataset()