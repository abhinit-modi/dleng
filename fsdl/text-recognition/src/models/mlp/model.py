""" Compile the MLP model """
from tensorflow import keras
from src.models.mlp.layers import MLP
from src.models.mlp.params import MODEL_CONFIG, COMPILE_CONFIG


def initialize(model_config=MODEL_CONFIG, compile_config=COMPILE_CONFIG) -> keras.Model:
    """Instantiates a model, builds it and compiles it with passed configs."""
    model = MLPModel(model_config)
    model.compile(optimizer=compile_config['optimizer'],
                  loss=compile_config['loss'],
                  metrics=compile_config['metrics'])
    return model


class MLPModel(keras.Model):
    """Multi Layer Perceptron Model."""
    def __init__(self, model_config=MODEL_CONFIG):
        super().__init__()
        self.mlp_block = MLP(model_config)

    def call(self, inputs, training=None, mask=None):
        return self.mlp_block(inputs)

    def build(self, input_shape):
        inputs = keras.layers.Input(shape=input_shape)
        return keras.Model(inputs=[inputs], outputs=self.call(inputs))

    def get_config(self):
        return self.mlp_block.get_config()
