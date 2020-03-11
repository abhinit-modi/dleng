# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import os
import shutil
import random

N = 300
SPLIT = 0.8

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def make_dataset(input_dir, output_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data ' + input_dir)
    
    ROOTDIR = os.path.join(output_dir, "wikiart_sampled")
    if os.path.isdir(ROOTDIR):
        shutil.rmtree(ROOTDIR)
    os.mkdir(ROOTDIR)
    os.mkdir(os.path.join(ROOTDIR, "train"))
    os.mkdir(os.path.join(ROOTDIR, "test"))
    
    dirs = os.listdir(input_dir)
    fmap = {}
    for d in dirs:
        if not os.path.isdir(os.path.join(input_dir, d)):
            continue
        files = os.listdir(os.path.join(input_dir, d))
        for f in  files:
            author = f[:f.find('_')]
            if author not in fmap:
                fmap[author] = []

            fmap[author].append((os.path.join(input_dir, d, f), os.path.join(ROOTDIR, '{split}', author, f[f.find('_')+1:])))

    for author in fmap:
        if len(fmap[author]) < N:
            continue

        os.mkdir(os.path.join(ROOTDIR, "train", author))
        os.mkdir(os.path.join(ROOTDIR, "test", author))

        random.shuffle(fmap[author])

        copied = set()
        for index, images in enumerate(fmap[author]):
            if images[1] in copied:
                continue
            copied.add(images[1])
            shutil.copyfile(images[0], images[1].format(split = "train" if index*1.0/N <= SPLIT else "test"))
            if len(copied) == N:
                break


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    make_dataset()
