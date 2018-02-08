import lasagne
from lasagne.layers import (Layer, ElemwiseSumLayer, InputLayer,
                            NonlinearityLayer, Conv1DLayer)

from ..layers import GatedConv1DLayer, InstanceNorm1D

import theano.tensor as T


class Sound2SoundNet:
    """
    A sound to sound generator
    """

    def __init__(self, seq_len, n_chan=1025, n_base_filter=128, n_residual=7,
                 nonlinearity=None, norm_func=InstanceNorm1D):
        """
        :param seq_len: int, length of the sequence
        :param n_chan: int, number of channels
        :param n_base_filter: int, the base bumber of conv filters
        :param n_residual: int, number of residual blocks
        :param nonlinearity: nonlinearity for the generator
        :param norm_func: normalization function
        """
        self.seq_len = seq_len
        self.n_chan = n_chan
        self.n_base_filter = n_base_filter
        self.n_residual = n_residual

        self.nonlinearity = nonlinearity
        self.norm_func = norm_func

        self.input_var = T.tensor3("input spectrogram")

        self.layers = self._build_net()

    def _build_net(self):
        class net:
            inp = InputLayer((None, self.n_chan, self.seq_len,), input_var=self.input_var)
            conv1 = Conv1DLayer(inp, self.n_base_filter, 15, nonlinearity=self.nonlinearity)
            gate1 = GatedConv1DLayer(conv1, self.n_base_filter)

            conv2 = Conv1DLayer(gate1, self.n_base_filter*2, 5, stride=2, nonlinearity=self.nonlinearity)
            norm1 = self.norm_func(conv2)
            gate2 = GatedConv1DLayer(norm1, self.n_base_filter*2)

            conv3 = Conv1DLayer(gate2, self.n_base_filter*4, 4, stride=2, nonlinearity=self.nonlinearity)
            norm2 = self.norm_func(conv3)
            gate3 = GatedConv1DLayer(norm2, self.n_base_filter*4)

            resid = gate3
            for _ in range(7):
                resid = ResidualBlock1D(resid, self.n_base_filter * 8, 3)



        return net

'''
class ResidualBlock1D(Layer):
    """
    1 Dimensional Residual Layer
    """
    def __init__(self, incoming, num_filters, filter_size,
                 norm_layer=InstanceNorm1D, nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2), **kwargs):
        """
        :param incoming: the layer feeding into this layer, or the expected input shape.
        :param num_filters: number of convolutional filters
        :param filter_size: size of convolutional filters
        :param norm_layer: normalization technique to use
        """
        super(ResidualBlock1D, self).__init__(incoming, **kwargs)

        conv = Conv1DLayer(incoming, num_filters=num_filters, filter_size=filter_size, pad="same")
        norm = norm_layer(conv)
        nonlin = NonlinearityLayer(norm, nonlinearity)

        conv = Conv1DLayer(nonlin, num_filters=int(num_filters/2), filter_size=filter_size, pad="same")
        norm = norm_layer(conv)

        sum = ElemwiseSumLayer([incoming, norm])

        self.get_output_for = sum.get_output_for
        self.get_output_shape_for = sum.get_output_shape_for
        self.get_params = sum.get_params
'''


class ResidualBlock1D(Layer):

    def __init__(self, incoming, num_filters, filter_size=3, stride=1, num_layers=2):
        print incoming.output_shape
        super(ResidualBlock1D, self).__init__(incoming)

        conv = incoming
        if (num_filters != incoming.output_shape[1]) or (stride != 1):
            incoming = Conv1DLayer(incoming, num_filters, filter_size=1, stride=stride, pad=0, nonlinearity=None, b=None)
        for _ in range(num_layers):
            conv = Conv1DLayer(conv, num_filters, filter_size, pad='same')

        sum = ElemwiseSumLayer([conv, incoming])

        self.get_output_shape_for = sum.get_output_shape_for
        self.get_output_for = sum.get_output_for
        self.get_params = sum.get_params