# -*- coding: utf-8 -*-
"""Core Keras layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from deep import backend as K
from .. import activations
from deep import initializers
from deep import regularizers
from deep import constraints
from deep.base_layer import InputSpec
from deep.base_layer import Layer

from deep import interfaces

class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            The purpose of this argument is to preserve weight
            ordering when switching a model from one data format
            to another.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Example
    ```python
        model = Sequential()
        model.add(Conv2D(64, (3, 3),
                         input_shape=(3, 32, 32), padding='same',))
        # now: model.output_shape == (None, 64, 32, 32)
        model.add(Flatten())
        # now: model.output_shape == (None, 65536)
    ```
    """

    def __init__(self, data_format=None, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)
        self.data_format = K.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "Flatten" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '). '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:]))

    def call(self, inputs):
        if self.data_format == 'channels_first':
            # Ensure works for any dim
            permutation = [0]
            permutation.extend([i for i in
                                range(2, K.ndim(inputs))])
            permutation.append(1)
            inputs = K.permute_dimensions(inputs, permutation)

        return K.batch_flatten(inputs)

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(Flatten, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Dropout(Layer):
    """Applies Dropout to the input.
    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
           http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
    """
    @interfaces.legacy_dropout_support
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class Activation(Layer):
    """Applies an activation function to an output.
    # Arguments
        activation: name of activation function to use
            (see: [activations](../activations.md)),
            or alternatively, a Theano or TensorFlow operation.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    def __init__(self, activation, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.supports_masking = True
        self.activation = activations.get(activation)

    def call(self, inputs):
        return self.activation(inputs)

    def get_config(self):
        config = {'activation': activations.serialize(self.activation)}
        base_config = super(Activation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class Dense(Layer):
    """Just your regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Reshape(Layer):
    """Reshapes an output to a certain shape.
    # Arguments
        target_shape: target shape. Tuple of integers.
            Does not include the batch axis.
    # Input shape
        Arbitrary, although all dimensions in the input shaped must be fixed.
        Use the keyword argument `input_shape`
        (tuple of integers, does not include the batch axis)
        when using this layer as the first layer in a model.
    # Output shape
        `(batch_size,) + target_shape`
    # Example
    ```python
        # as first layer in a Sequential model
        model = Sequential()
        model.add(Reshape((3, 4), input_shape=(12,)))
        # now: model.output_shape == (None, 3, 4)
        # note: `None` is the batch dimension
        # as intermediate layer in a Sequential model
        model.add(Reshape((6, 2)))
        # now: model.output_shape == (None, 6, 2)
        # also supports shape inference using `-1` as dimension
        model.add(Reshape((-1, 2, 2)))
        # now: model.output_shape == (None, 3, 2, 2)
    ```
    """

    def __init__(self, target_shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def _fix_unknown_dimension(self, input_shape, output_shape):
        """Finds and replaces a missing dimension in an output shape.
        This is a near direct port of the internal Numpy function
        `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`
        # Arguments
            input_shape: original shape of array being reshaped
            output_shape: target shape of the array, with at most
                a single -1 which indicates a dimension that should be
                derived from the input shape.
        # Returns
            The new output shape with a `-1` replaced with its computed value.
        # Raises
            ValueError: if `input_shape` and `output_shape` do not match.
        """
        output_shape = list(output_shape)
        msg = 'total size of new array must be unchanged'

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('Can only specify one unknown dimension.')
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)

        return tuple(output_shape)

    def compute_output_shape(self, input_shape):
        if None in input_shape[1:]:
            # input shape (partially) unknown? replace -1's with None's
            return ((input_shape[0],) +
                    tuple(s if s != -1 else None for s in self.target_shape))
        else:
            # input shape known? then we can compute the output shape
            return (input_shape[0],) + self._fix_unknown_dimension(
                input_shape[1:], self.target_shape)

    def call(self, inputs):
        return K.reshape(inputs, (K.shape(inputs)[0],) + self.target_shape)

    def get_config(self):
        config = {'target_shape': self.target_shape}
        base_config = super(Reshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))