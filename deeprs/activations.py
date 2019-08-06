"""Built-in activation functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import warnings
from deep import backend as K
from deep.utils import deserialize_keras_object

from deep import Layer
from deep.initializers import Zeros
import tensorflow as tf


class Dice(Layer):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

      Input shape
        - Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.

      Output shape
        - Same shape as the input.

      Arguments
        - **axis** : Integer, the axis that should be used to compute data distribution (typically the features axis).

        - **epsilon** : Small float added to variance to avoid dividing by zero.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alphas = self.add_weight(shape=(input_shape[-1],), initializer=Zeros(
        ), dtype=tf.float32, name=self.name + 'dice_alpha')  # name='alpha_'+self.name
        super(Dice, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        inputs_normed = tf.layers.batch_normalization(
            inputs, axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

    def get_config(self, ):
        config = {'axis': self.axis, 'epsilon': self.epsilon}
        base_config = super(Dice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def softmax(x, axis=-1):
    """Softmax activation function.
    # Arguments
        x: Input tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                         'Received input: %s' % x)


def elu(x, alpha=1.0):
    """Exponential linear unit.
    # Arguments
        x: Input tensor.
        alpha: A scalar, slope of negative section.
    # Returns
        The exponential linear activation: `x` if `x > 0` and
        `alpha * (exp(x)-1)` if `x < 0`.
    # References
        - [Fast and Accurate Deep Network Learning by Exponential
           Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    """
    return K.elu(x, alpha)


def selu(x):
    """Scaled Exponential Linear Unit (SELU).
    SELU is equal to: `scale * elu(x, alpha)`, where alpha and scale
    are predefined constants. The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly (see `lecun_normal` initialization) and the number of inputs
    is "large enough" (see references for more information).
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # Returns
       The scaled exponential unit activation: `scale * elu(x, alpha)`.
    # Note
        - To be used together with the initialization "lecun_normal".
        - To be used together with the dropout variant "AlphaDropout".
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * K.elu(x, alpha)


def softplus(x):
    """Softplus activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The softplus activation: `log(exp(x) + 1)`.
    """
    return K.softplus(x)


def softsign(x):
    """Softsign activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The softsign activation: `x / (abs(x) + 1)`.
    """
    return K.softsign(x)


def relu(x, alpha=0., max_value=None, threshold=0.):
    """Rectified Linear Unit.
    With default values, it returns element-wise `max(x, 0)`.
    Otherwise, it follows:
    `f(x) = max_value` for `x >= max_value`,
    `f(x) = x` for `threshold <= x < max_value`,
    `f(x) = alpha * (x - threshold)` otherwise.
    # Arguments
        x: Input tensor.
        alpha: float. Slope of the negative part. Defaults to zero.
        max_value: float. Saturation threshold.
        threshold: float. Threshold value for thresholded activation.
    # Returns
        A tensor.
    """
    return K.relu(x, alpha=alpha, max_value=max_value, threshold=threshold)


def tanh(x):
    """Hyperbolic tangent activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The hyperbolic activation:
        `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
    """
    return K.tanh(x)


def sigmoid(x):
    """Sigmoid activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The sigmoid activation: `1 / (1 + exp(-x))`.
    """
    return K.sigmoid(x)


def hard_sigmoid(x):
    """Hard sigmoid activation function.
    Faster to compute than sigmoid activation.
    # Arguments
        x: Input tensor.
    # Returns
        Hard sigmoid activation:
        - `0` if `x < -2.5`
        - `1` if `x > 2.5`
        - `0.2 * x + 0.5` if `-2.5 <= x <= 2.5`.
    """
    return K.hard_sigmoid(x)


def exponential(x):
    """Exponential (base e) activation function.
    # Arguments
        x: Input tensor.
    # Returns
        Exponential activation: `exp(x)`.
    """
    return K.exp(x)


def linear(x):
    """Linear (i.e. identity) activation function.
    # Arguments
        x: Input tensor.
    # Returns
        Input tensor, unchanged.
    """
    return x


def serialize(activation):
    return activation.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='activation function')


def get(identifier):
    """Get the `identifier` activation function.
    # Arguments
        identifier: None or str, name of the function.
    # Returns
        The activation function, `linear` if `identifier` is None.
    # Raises
        ValueError if unknown identifier
    """
    if identifier is None:
        return linear
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        if isinstance(identifier, Layer):
            warnings.warn(
                'Do not pass a layer instance (such as {identifier}) as the '
                'activation argument of another layer. Instead, advanced '
                'activation layers should be used just like any other '
                'layer in a model.'.format(
                    identifier=identifier.__class__.__name__))
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'activation function identifier:', identifier)


def activation_fun(activation, fc):
    if isinstance(activation, str):
        fc = tf.keras.layers.Activation(activation)(fc)
    elif issubclass(activation, Layer):
        fc = activation()(fc)
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return fc
