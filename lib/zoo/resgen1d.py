from lasagne.layers import (ElemwiseSumLayer, InputLayer, NonlinearityLayer,
                             Conv1DLayer, get_output, get_all_params)

from ..layers import GatedConv1DLayer, InstanceNorm1D
from ..layers import PixelShuffle1DLayer

import theano
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
        self.n_input_chan = n_chan
        self.n_base_filter = n_base_filter
        self.n_residual = n_residual

        self.nonlinearity = nonlinearity
        self.norm_func = norm_func

        self.input_var = T.tensor3("input spectrogram")

        self.layers = self._build_net()
        self.input_var = self.layers.inp.input_var
        self.output_var = get_output(self.layers.out)
        self.output_shape = self.layers.out.output_shape

        self.generate = theano.function([self.input_var], self.output_var)
        self.params = get_all_params(self.layers.out, trainable=True)

    def _build_net(self):
        class net:
            inp = InputLayer((None, self.n_input_chan, self.seq_len,), input_var=self.input_var)
            conv1 = Conv1DLayer(inp, self.n_base_filter, 15, nonlinearity=self.nonlinearity, pad="same")
            gate1 = GatedConv1DLayer(conv1)

            conv2 = Conv1DLayer(gate1, self.n_base_filter*2, 5, stride=2, nonlinearity=self.nonlinearity, pad="same")
            norm1 = self.norm_func(conv2)
            gate2 = GatedConv1DLayer(norm1)

            conv3 = Conv1DLayer(gate2, self.n_base_filter*4, 4, stride=2, nonlinearity=self.nonlinearity, pad="same")
            norm2 = self.norm_func(conv3)
            gate3 = GatedConv1DLayer(norm2)

            resid = gate3
            for _ in range(self.n_residual):
                resid = ResidualBlock(resid)

            conv4 = Conv1DLayer(resid, self.n_base_filter*8, 5, pad="same")
            pixshuf1 = PixelShuffle1DLayer(conv4, 2)
            norm3 = self.norm_func(pixshuf1)
            gate4 = GatedConv1DLayer(norm3)

            conv5 = Conv1DLayer(gate4, self.n_base_filter*4, 5, pad="same")
            pixshuf2 = PixelShuffle1DLayer(conv5, 2)
            norm4 = self.norm_func(pixshuf2)
            gate5 = GatedConv1DLayer(norm4)

            out = Conv1DLayer(gate5, self.n_input_chan, 15, pad="same")

        return net


def ResidualBlock(incoming):
    num_filters = incoming.output_shape[1]

    conv1 = Conv1DLayer(incoming, num_filters*2, 3, pad="same")
    norm1 = InstanceNorm1D(conv1)
    gate1 = GatedConv1DLayer(norm1)

    conv2 = Conv1DLayer(gate1, num_filters, 3, pad="same")
    norm2 = InstanceNorm1D(conv2)
    gate2 = GatedConv1DLayer(norm2)

    return ElemwiseSumLayer([incoming, gate2])

