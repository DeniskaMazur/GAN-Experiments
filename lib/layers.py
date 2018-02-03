import theano.tensor as T

from lasagne.layers import Layer, Conv1DLayer, get_output
from lasagne.init import GlorotUniform

class InstanceNorm(Layer):
    def __init__(self, incoming, epsilon=1e-5, **kwargs):
        super(InstanceNorm, self).__init__(incoming, **kwargs)

        self.epsilon = epsilon

    def get_output_for(self, input, **kwargs):
        mean = T.mean(input, axis=[2, 3], keepdims=True)
        var = T.var(input, axis=[2, 3], keepdims=True)

        return (input - mean)

    def get_output_shape_for(self, input_shape):
        return input_shape


class GatedConv1DLayer(Layer):
    """
    Implementation of https://arxiv.org/pdf/1612.08083.pdf

    H_{l+1} = (H_l * W_l + b_l) * sigmoid(H_l + V_l + c_l)
    """
    def __init__(self, incoming, num_filters, filter_size, pad, nonlinearity=None, **kwargs):
        """
        Creates a GatedConv1DLayer instance
        :param incoming: incoming layer
        :param num_filters: int, number of conv filters
        :param filter_size: int, size of conv filters
        :param nonlinearity: activation function for h
        """
        super(GatedConv1DLayer, self).__init__(incoming, **kwargs)

        h = Conv1DLayer(incoming, num_filters, filter_size, pad=pad, nonlinearity=nonlinearity)
        g = Conv1DLayer(incoming, num_filters, filter_size, pad=pad, nonlinearity=T.nnet.sigmoid)

        self.h = get_output(h)
        self.g = get_output(g)

        self.get_output_shape_for = g.get_output_shape_for

    def get_output_for(self, input, **kwargs):
        return self.h * self.g
    