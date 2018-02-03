from lasagne.layers import (Layer, ElemwiseSumLayer, InputLayer,
                            NonlinearityLayer, BatchNormLayer, Conv1DLayer)

import theano.tensor as T


class Sound2SoundNet:
    """
    A sound to sound generator
    """

    def __init__(self, seq_len, n_chan=1025, n_base_filter=128, n_residual=7,
                 nonlinearity=T.nnet.elu, norm_func=BatchNormLayer):
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

        self.Nonlinearity = lambda layer: NonlinearityLayer(layer, nonlinearity=nonlinearity)
        self.norm_func = norm_func

        self.input_var = T.tensor3("input spectrogram")

    def _build_net(self):
        class net:
            inp = InputLayer((None, self.n_chan, self.seq_len,), input_var=self.input_var)
            conv1 = Conv1DLayer(inp, self.n_base_filter, 15, nonlinearity=None)
            nonlin1 = self.Nonlinearity(conv1)

            conv2 = Conv1DLayer(nonlin1, self.n_base_filter*2, 5, 2, nonlinearity=None)
            norm1 = self.norm_func(conv2)
            nonlin2 = self.Nonlinearity(norm1)

            conv3 = Conv1DLayer(nonlin2, self.n_base_filter*4, 5, 2, nonlinearity=None)
            norm2 = self.norm_func(conv3)
            nonlin3 = self.Nonlinearity(norm2)

            _ = nonlin3
            for _ in range(self.n_residual):
                _ = ResidualBlock1D(_, self.n_base_filter*8, 3)
            residual = _


class ResidualBlock1D(Layer):
    """
    1 Dimensional Residual Layer
    """
    def __init__(self, incoming, num_filters, filter_size,
                 norm_layer=BatchNormLayer, nonlinearity=T.nnet.elu, **kwargs):
        """
        :param incoming: the layer feeding into this layer, or the expected input shape.
        :param num_filters: number of convolutional filters
        :param filter_size: size of convolutional filters
        :param norm_layer: normalization technique to use
        """
        super().__init__(incoming, **kwargs)

        conv = Conv1DLayer(incoming, num_filters=num_filters, filter_size=filter_size)
        norm = norm_layer(conv)
        nonlin = NonlinearityLayer(norm, nonlinearity)
        conv = Conv1DLayer(nonlin, num_filters=int(num_filters/2), filter_size=filter_size)
        norm = norm_layer(conv)
        sum = ElemwiseSumLayer([incoming, norm])

        self.get_output_for = sum.get_output_for
        self.get_output_shape_for = sum.get_output_shape_for
        self.get_params = sum.get_params


