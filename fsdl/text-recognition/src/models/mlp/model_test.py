""" Unit testing model to catch coding errors."""

import logging
import numpy as np
import tensorflow as tf
from src.models.mlp.model import initialize
from src.models.mlp.params import MODEL_CONFIG, COMPILE_CONFIG


class ModelSanityTest(tf.test.TestCase):
    """ Unit tests.
    """

    def __init__(self):
        tf.test.TestCase.__init__(self)

    def setUp(self):
        super().setUp()
        self.test_model = initialize(MODEL_CONFIG, COMPILE_CONFIG)

        # self.test_model = keras.Sequential([MLP(MODEL_CONFIG)])

        # inputs = keras.Input(shape=[1, 784])
        # outputs = MLP(MODEL_CONFIG)(inputs)
        # self.test_model = keras.Model(inputs, outputs)

    def tearDown(self):
        del self.test_model

    def test_model_input_shape(self):
        """Check expected input shape"""
        dummy_image_batch = np.random.rand(1, 784)
        dummy_label = np.random.randint(low=0, high=9, size=1)
        self.test_model.fit(dummy_image_batch, dummy_label, epochs=1,
                            batch_size=1)
        print(self.test_model.summary())
        self.assertEqual(self.test_model.layers[0].input_shape,
                         (None, 1, 784), "Input shape not as expected")

    def test_model_output_shape(self):
        """Check expected output shape"""
        self.assertEqual(self.test_model.layers[-1].output_shape,
                         (None, 1, 10), "Output shape not as expected")

    def test_weights_update_on_step(self):
        """Check that weights change in one training step."""
        train_before = self.test_model.get_weights()
        dummy_image_batch = np.random.rand(1, 784)
        dummy_label = np.random.randint(low=0, high=9, size=1)
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

    def test_over_fit_small_dataset(self):
        """See if model overfits on small dataset."""
        dummy_image_batch = np.random.rand(100, 1, 784)
        dummy_label = np.random.randint(low=0, high=9, size=100)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=2)
        history = self.test_model.fit(dummy_image_batch,
                                      dummy_label,
                                      epochs=10,
                                      callbacks=[early_stop],
                                      batch_size=5)
        print(history.history['loss'])
        self.assertTrue(history.history['loss'][-1] > history.history['loss'][-2],
                        "Loss never tipped, unable to overfit")

    def test_zero_input_non_zero_input(self):
        """Verify difference in outputs for a dummy and non dummy case."""
        zero_predictions = self.test_model.predict(np.zeros((1, 784)))
        print(zero_predictions)
        dummy_image_batch = np.random.rand(1, 784)
        non_zero_predictions = self.test_model.predict(dummy_image_batch)
        print(non_zero_predictions)
        self.assertNotAllEqual(zero_predictions, non_zero_predictions,
                               "Same predictions for zero and valid inputs")

    def runTest(self):  # pylint: disable=invalid-name
        """Runs all the tests."""
        self.test_model_input_shape()
        self.test_model_output_shape()
        self.test_weights_update_on_step()
        self.test_over_fit_small_dataset()
        self.test_zero_input_non_zero_input()


def test_model():
    """ Unit tests the default model built from model.py.
    """
    logger = logging.getLogger(__name__)

    result = ModelSanityTest().run()
    logger.info("Errors:")
    logger.info(result.errors)
    logger.info("Failures:")
    logger.info(result.failures)
    return result


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    test_model()
