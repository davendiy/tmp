#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 29.10.2020
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import tensorflow as tf
import numpy as np
from abc import abstractmethod, ABC
from sklearn.linear_model import LinearRegression
from functools import reduce
import operator
import logging
import time
import rdtsc
import threading

# disable tensorflow warnings
logging.disable(logging.WARNING)


class Layer(ABC):

    name = 'layer'

    def __init__(self, width_in, width_out):
        self.w_in = width_in
        self.w_out = width_out

    @abstractmethod
    def generate(self):
        """ Generate analogue of such layer using given ml framework.
        """
        pass

    @abstractmethod
    def linear_parameters(self):
        pass

    @abstractmethod
    def output_shape(self):
        pass


class Conv(Layer, ABC):

    name = 'conv'

    def __init__(self, width_in, depth_in, depth_out,
                 filter_width, stride, padding):
        """ Convolutional layer

        Attack can't determine:
             - activation function (thus assume that it is always 'relu')
             - batch normalization (thus assume that there are not such option)

        :param width_in:     input width (assume that all the layers are square)
        :param depth_in:     input depth
        :param depth_out:    output depth (amount of channels)
        :param filter_width: filter width
                             (also assume that all the filters are square)
        :param stride:       stride
        :param padding:      padding of input with zeros
                             (valid, same or any positive number)
        """
        if padding == 'same':
            width_out = int(np.ceil(width_in / stride))
        else:
            if padding == 'valid': padding = 0
            width_out = (width_in - filter_width + 2 * padding) // stride + 1

        super().__init__(width_in, width_out)
        self.d_in = depth_in
        self.d_out = depth_out
        self.f = filter_width
        self.s = stride
        self.pd = padding

    def mac(self):
        """ Amount of multiply-and-accumulate operations.
        """
        return self.w_out * self.f ** 2 * self.d_in * self.d_out

    def new_neurons(self):
        """ Amount of new neurons during convolutional calculation.
        """
        #         amount of convolutions without the first one
        return ((self.w_out ** 2 - 1)

                # amount of new neurons during each
                # next convolution (such that there weren't in the previous)
                * self.f * self.d_in * self.s

                # amount of new neurons during the first convolution
                + (self.f ** 2 + self.w_in ** 2) * self.d_in)  \
                * self.d_out      # amount of filters

    def linear_parameters(self):
        return [self.mac(), self.new_neurons()]

    def output_shape(self):
        return self.w_out, self.w_out, self.d_out


class TFv2_Conv(Conv):

    def generate(self):
        return tf.keras.layers.Conv2D(
            filters=self.d_out,
            kernel_size=self.f,
            padding=self.pd,
            activation='relu',
            input_shape=(self.w_in, self.w_in, self.d_in)
        )


class Pool(Layer, ABC):

    name = 'pool'

    def __init__(self, width_in, depth_in, filter_width, stride):
        """ Pooling layer.

        Attack can't determine:
             - pooling type (thus assume that it is always 'max')

        :param width_in:     input width (assume that all the layers are square)
        :param depth_in:     input depth (depth out is always the same)
        :param filter_width: filter width
                             (also assume that all the filters are square)
        :param stride:       stride

        Note: assume that padding is always 'valid'
        """
        width_out = (width_in - filter_width) // stride + 2
        super().__init__(width_in, width_out)
        self.d_in = depth_in
        self.f = filter_width
        self.s = stride

    def mac(self):
        """ Amount of multiply-and-accumulate operations.
        """

        return self.w_out * self.f ** 2 * self.d_in * self.d_in

    def new_neurons(self):
        """ Amount of new neurons during convolutional calculation.
        """
        #         amount of convolutions without the first one
        return ((self.w_out ** 2 - 1)

                # amount of new neurons during each
                # next convolution (such that there weren't in the previous)
                * self.f * self.d_in * self.s

                # amount of new neurons during the first convolution
                # (filter has no neurons)
                + (self.w_in ** 2) * self.d_in)  \
                * self.d_in      # amount of filters

    def linear_parameters(self):
        return [self.mac(), self.new_neurons()]

    def output_shape(self):
        return self.w_out, self.w_out, self.d_in


class TFv2_Pool(Pool):

    def generate(self):
        return tf.keras.layers.MaxPool2D(
            kernel_size=self.f,
            padding='valid',
            input_shape=(self.w_in, self.w_in, self.d_in)
        )


class FullyConnected(Layer, ABC):

    name = 'fc'

    def __init__(self, width_in, width_out, last_layer=False):
        """ Fully Connected layer.

        Attack can't determine:
             - activation function (thus assume that it is always 'relu'
               except the last one)
             - batch normalization (thus assume that there are not such option)

        :param width_in:   input width (assume there was a Flatten layer before)
        :param width_out:  output width (amount of neurons)
        :param last_layer: True, if given layer should be the last one, thus
                           activation function won't be 'relu'.
        """
        super().__init__(width_in, width_out)
        self._last = last_layer

    def linear_parameters(self):
        return [self.mac(), ]

    def mac(self):
        """Amount of multiply and accumulate (MAC) operations."""
        return self.w_in * self.w_out

    def output_shape(self):
        return self.w_out,


class TFv2_FullyConnected(FullyConnected):

    def generate(self):
        activation = '' if self._last else 'relu'
        return tf.keras.layers.Dense(units=self.w_out, activation=activation,
                                     input_shape=(self.w_in, ))


class NotFoundError(ValueError):
    pass


NAMES = ('conv', 'pool', 'fc')
validators = {name: LinearRegression() for name in NAMES}

cid = time.pthread_getcpuclockid(threading.get_ident())

# different possible function for benchmarking
get_time1 = lambda : time.clock_gettime_ns(cid)
get_time2 = lambda : rdtsc.get_cycles()


# =========================== SEARCH SPACE =====================================

POSSIBLE_FILTER_WIDTHS = [3, 5, 7, 9, 11, 13, 15, 17]
POSSIBLE_STRIDES = [2, 3, 4]
POSSIBLE_FILTERS_AMOUNT = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
POSSIBLE_PADDINGS = ['same', 'valid']

POSSIBLE_UNITS = [8, 16, 32, 64, 128, 256, 512]

# ==============================================================================

# [y_1, y_2, y_3 ... y_p] - number of LLC misses on i-th probe
def get_probes(n: int, p: int, model: tf.keras.models.Model) -> np.ndarray:
    pass


# R = [r_1, r_2, r_3, ...]
def discriminator(a_traces: np.ndarray, v_traces: np.ndarray) -> np.ndarray:
    """ Just L2 difference between attack's traces and victim's.
    """
    s = ((a_traces - v_traces)**2).sum(axis=1)
    return np.sqrt(s / a_traces.shape[0])


def generate_input(input_shape):
    return tf.keras.layers.Input(shape=input_shape)


def possible_layers(input_shape):
    """ Yields all possible structures for the next layer.
    """
    for units in POSSIBLE_UNITS:
        # if the next layer is fully connected, then to the previous layer the
        # Flatten was applied, thus input dim will equal to the product of dims
        yield TFv2_FullyConnected(reduce(operator.mul, input_shape), units)

    for filter_width in POSSIBLE_FILTER_WIDTHS:
        for stride in POSSIBLE_STRIDES:
            for filters in POSSIBLE_FILTERS_AMOUNT:
                for padding in POSSIBLE_PADDINGS:
                    yield TFv2_Conv(width_in=input_shape[0],
                                    depth_in=input_shape[1],
                                    depth_out=filters,
                                    filter_width=filter_width,
                                    stride=stride,
                                    padding=padding)

    for filter_width in POSSIBLE_FILTER_WIDTHS:
        for stride in POSSIBLE_STRIDES:
            yield TFv2_Pool(width_in=input_shape[0], depth_in=input_shape[1],
                            stride=stride, filter_width=filter_width)


def random_layers(input_shape, amount=60):
    """ Yields _amount_ random possible structures for the next layer.
    """

    for _ in range(amount):

        # CONV
        filter_width = np.random.choice(POSSIBLE_FILTER_WIDTHS)
        stride = np.random.choice(POSSIBLE_STRIDES)
        filters = np.random.choice(POSSIBLE_FILTERS_AMOUNT)
        padding = np.random.choice(POSSIBLE_PADDINGS)
        TFv2_Conv(width_in=input_shape[0],
                    depth_in=input_shape[1],
                    depth_out=filters,
                    filter_width=filter_width,
                    stride=stride,
                    padding=padding)

        # POOL
        filter_width = np.random.choice(POSSIBLE_FILTER_WIDTHS)
        stride = np.random.choice(POSSIBLE_STRIDES)
        yield TFv2_Pool(width_in=input_shape[0], depth_in=input_shape[1],
                        stride=stride, filter_width=filter_width)

        # FullyConnected
        units = np.random.choice(POSSIBLE_UNITS)
        yield TFv2_FullyConnected(reduce(operator.mul, input_shape), units)


def prepare_validator(benchmark=get_time1):
    """ Train linear regression models on random layers in order to
    find the parameters for running machine.

    NOTE: this function gets executed time using time.time() function. Thus,
    if you want to run models on another machine you should modify this
    function and share generated models to the machine

    """
    # since we assume that executed time of layer is linear in both
    # its macs and new_neurons, we can train linear regression for small
    # input shape. It's important to use a lot of different models, but their
    # size can variate
    input_shape = [62, 62, 32]

    # random input data
    random = tf.constant(np.random.random(size=[1] + input_shape))

    # all parameters and all time values for each layer's type
    all_pars = {name: [] for name in NAMES}
    all_times = {name: [] for name in NAMES}

    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(inputs)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(x)

    # global model with few layers
    # use its execution time as offense for compilcation
    global_model = tf.keras.models.Model(inputs=inputs, outputs=x)
    global_model.compile()
    t = benchmark()
    global_model.predict(random)
    global_time = benchmark() - t

    # for each layer we need to find its linear parameters and executed time
    for layer in random_layers(input_shape):   # type: Layer
        a_pars = all_pars[layer.name]     # corresponding list of parameters
        a_times = all_pars[layer.name]

        a_pars.append(layer.linear_parameters())

        # create new neural model with one extra layer
        if layer.name == 'fc':
            tmp_x = tf.keras.layers.Flatten()(x)
            tmp_x = layer.generate()(tmp_x)
        else:
            tmp_x = layer.generate()(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        model.compile()

        # exec our model on random data and get its executed time
        t = benchmark()
        model.predict(random)
        a_times.append((benchmark() - t) - global_time)

    # fit each linear model with accumulated data
    for name in NAMES:
        validators[name].fit(all_pars[name], all_times[name])


def validate(layer, k_prev, k_hyp, threshold=10e-6) -> bool:
    hyp_time = k_hyp - k_prev
    theor_time = validators[layer.name].predict(layer.linear_parameters())
    return np.abs(theor_time - hyp_time) < threshold


def ganred(input_shape, v_traces: np.ndarray, eta):
    """ General algorithm according to the described one in paper.
    """
    n, p = v_traces.shape
    inputs = generate_input(input_shape)
    x = inputs
    k = [0]
    layers = [inputs, ]
    cur_l = 0
    while k[cur_l] < p:
        cur_l += 1
        best_k = k[cur_l - 1]
        best_layer = None
        for layer in possible_layers(input_shape):   # type: Layer
            tf_layer = layer.generate()
            x_tmp = tf_layer(x)
            tmp_model = tf.keras.models.Model(inputs=inputs, outputs=x_tmp)
            tmp_model.compile()

            a_traces = get_probes(n, p, tmp_model)
            r = discriminator(a_traces, v_traces)
            k_hyp = np.argmax(r > eta) - 1   # k_hyp is the last r_i that < eta
            if k_hyp > best_k and validate(layer, k[cur_l - 1], k_hyp):
                best_k = k_hyp
                best_layer = layer

        if best_layer is None:
            raise NotFoundError()

        best_tf_layer = best_layer.generate()
        x = best_tf_layer(x)
        layers.append(x)
        input_shape = best_layer.output_shape()
    return x, layers
