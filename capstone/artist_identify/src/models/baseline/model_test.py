# -*- coding: utf-8 -*-

import json
import numpy as np
import tensorflow as tf
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import model

class ModelSanityTest(tf.test.TestCase):
    def setUp(self):
      super(ModelSanityTest, self).setUp()
      self.test_model = tf.keras.model.load_model(model.make_model())
    
    def tearDown(self):
      del self.test_model
        
    def testModelInputShape(self):
      self.assertEqual(self.test_model.layers[0].input_shape, 
                       (None, 224, 224, 3), "Input shape not as expected")

    def testModelOutputShape(self):
      self.assertEqual(self.test_model.layers[-1].output_shape, 
                       (None, 45), "Output shape not as expected")

    def testWeightsUpdateOnStep(self):
      train_before = self.test_model.get_weights()
      dummy_image_batch = np.random.rand(1,224,224,3)
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
      dummy_image_batch = np.random.rand(1,224,224,3)
      dummy_label = np.random.randint(low=0, high=45, size=1)
      history = self.test_model.fit(dummy_image_batch, dummy_label, epochs=10, 
                                    batch_size=1)
      loss, accuracy = self.test_model.evaluate(dummy_image_batch, dummy_label)
      self.assertEqual(loss, 0)
      print(history.history['loss'])
      self.assertTrue(history.history['loss'][-1] == 0, 
                      "Loss not zero, unable to overfit")

    def testZeroInputNonZeroInput(self):
      zero_predictions = self.test_model.predict(np.zeros((1,224,224,3)))
      print(zero_predictions)
      dummy_image_batch = np.random.rand(1,224,224,3)
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
      
def test_model():
    """ Unit tests the default model built from model.py.
    """
    logger = logging.getLogger(__name__)

    result = ModelSanityTest().run()
    logger.log(result)

    return result


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    test_model()
