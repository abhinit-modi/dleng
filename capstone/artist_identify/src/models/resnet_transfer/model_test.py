""" Unit tests for model """

import logging

import click
import numpy as np
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv


class ModelSanityTest(tf.test.TestCase):
    """ Unit tests.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        tf.test.TestCase.__init__(self)

    def setUp(self):
        super(ModelSanityTest, self).setUp()
        self.test_model = tf.keras.models.load_model(self.model_path)

    def tearDown(self):
        del self.test_model

    def testModelInputShape(self):
        self.assertEqual(self.test_model.layers[0].input_shape,
                         (None, 224, 224, 3), "Input shape not as expected")

    def testModelOutputShape(self):
        self.assertEqual(self.test_model.get_layer(index=-1).output_shape,
                         (None, 45), "Output shape not as expected")

    def testWeightsUpdateOnStep(self):
        train_before = self.test_model.get_weights()
        dummy_image_batch = np.random.rand(1, 224, 224, 3)
        dummy_label = np.random.randint(low=0, high=45, size=1)
        history = self.test_model.fit(dummy_image_batch, dummy_label, epochs=1,
                                      batch_size=1)

        train_after = self.test_model.get_weights()

        param_updated = False
        for index, weight in enumerate(train_after):
            if (np.count_nonzero(train_before[index] - weight)) > 0:
                param_updated = True
                break

        self.assertTrue(param_updated, "No weights updated")
        self.assertTrue(history.history['loss'][0] != 0, "Loss is zero")

    def testOverFitSmallDataset(self):
        dummy_image_batch = np.random.rand(1, 224, 224, 3)
        dummy_label = np.random.randint(low=0, high=45, size=1)
        history = self.test_model.fit(dummy_image_batch, dummy_label, epochs=20,
                                      batch_size=1)
        print(history.history['loss'])
        self.assertTrue(round(history.history['loss'][-1], 4) == 0,
                        "Loss not zero, unable to overfit")

    def testZeroInputNonZeroInput(self):
        zero_predictions = self.test_model.predict(np.zeros((1, 224, 224, 3)))
        print(zero_predictions)
        dummy_image_batch = np.random.rand(1, 224, 224, 3)
        non_zero_predictions = self.test_model.predict(dummy_image_batch)
        print(non_zero_predictions)
        self.assertNotAllEqual(zero_predictions, non_zero_predictions,
                               "Same predictions for zero and valid inputs")

    def runTest(self):
        self.testModelInputShape()
        self.testModelOutputShape()
        self.testWeightsUpdateOnStep()
        self.testOverFitSmallDataset()
        self.testZeroInputNonZeroInput()


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
def test_model(model_path):
    """ Unit tests the default model built from model.py.
    """
    logger = logging.getLogger(__name__)

    result = ModelSanityTest(model_path).run()
    logger.info("Errors:")
    logger.info(result.errors)
    logger.info("Failures:")
    logger.info(result.failures)
    return result

if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    test_model()
