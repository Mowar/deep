from deep import Layer
from deep.initializers import Zeros
from ..activations import activation_fun
import tensorflow as tf

class PredictionLayer(Layer):
    """
      Arguments
         - **activation**: Activation function to use.

         - **use_bias**: bool.Whther add bias term.
    """

    def __init__(self, activation='sigmoid', use_bias=True, **kwargs):
        self.activation = activation
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        output = activation_fun(self.activation, x)
        output = tf.reshape(output, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self,):
        config = {'activation': self.activation, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
