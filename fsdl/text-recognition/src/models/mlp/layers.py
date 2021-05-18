""" Build an MLP model """
from tensorflow import keras


class MLP(keras.layers.Layer):
    """Define layers of Multi Layer Perceptron."""
    def __init__(self, model_config):
        super().__init__()
        self.input_dim = model_config["input_dim"]
        self.num_classes = len(model_config["labels"])

        self.dropout = keras.layers.Dropout(0.5)
        self.fc1 = keras.layers.Dense(model_config["fc1_dim"], activation="relu")
        self.fc2 = keras.layers.Dense(model_config["fc2_dim"], activation="relu")
        self.fc3 = keras.layers.Dense(self.num_classes)

    def call(self, inputs, **kwargs):
        fc1 = self.fc1(inputs)
        fc1 = self.dropout(fc1)
        fc2 = self.fc2(fc1)
        fc2 = self.dropout(fc2)
        fc3 = self.fc3(fc2)
        return fc3
