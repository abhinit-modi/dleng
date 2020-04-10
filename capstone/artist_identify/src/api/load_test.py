""" Load testing a tf model server """

import asyncio
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer

import numpy as np
import requests
from dotenv import find_dotenv, load_dotenv
from tensorflow.keras.preprocessing import image

from interface_params import CONTRACT_CONFIG, SERVICE_CONFIG


def request_prediction(session, image_path):
    """ Read the image, send bytes to tf server for prediction.
    """
    # Preprocessing our input image
    img = image.img_to_array(image.load_img(
        image_path, target_size=CONTRACT_CONFIG['IMAGE_SIZE'])) * CONTRACT_CONFIG['RESCALE_FACTOR']

    img = img.astype('float16')

    payload = {
        "instances": [img.tolist()]
    }

    start_time = default_timer()

    # sending post request to TensorFlow Serving server
    with session.post(SERVICE_CONFIG['API_ENDPOINT'], json=payload) as resp:
        elapsed = default_timer() - start_time
        time_completed_at = "{:5.2f}s".format(elapsed)
        print("{0:<30} {1:>20}".format(image_path, time_completed_at))

        # Decode response
        pred = json.loads(resp.content.decode('utf-8'))

        # Get classified class
        predicted_class = CONTRACT_CONFIG['CLASSES'][np.argmax(pred['predictions'], axis=1)[0]]
        print(predicted_class)

async def get_data_asynchronous():
    """ Set up and run to async function in loop to fetch predictions for multiple
        images concurrently.
    """
    base_path = "/Users/abmodi/2020/dleng/capstone/\
        artist_identify/data/processed/wikiart_sampled/test"
    artist_dirs = os.listdir(base_path)
    number_of_images = 50
    images_to_predict = set()
    while len(images_to_predict) < number_of_images:
        artist_dir = random.choice(artist_dirs)
        image_file = random.choice(os.listdir(os.path.join(base_path, artist_dir)))
        if image_file not in images_to_predict:
            images_to_predict.add(
                os.path.join(base_path, artist_dir, image_file))

    print("{0:<30} {1:>20}".format("File", "Completed at"))
    with ThreadPoolExecutor(max_workers=10) as executor:
        with requests.Session() as session:
            # Set any session parameters here before calling `fetch`
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor,
                    request_prediction,
                    *(session, img_path) # Allows us to pass in multiple arguments to `fetch`
                )
                for img_path in images_to_predict
            ]
            for response in await asyncio.gather(*tasks):
                pass

def main():
    """ Set up asyncio and orhcestrate the loop."""
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(get_data_asynchronous())
    loop.run_until_complete(future)

if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
