import lasagne
from lasagne.layers import (Layer, ElemwiseSumLayer, InputLayer,
                            NonlinearityLayer, Conv1DLayer)

from ..layers import GatedConv1DLayer, InstanceNorm1D
from ..layers import PixelShuffle1DLayer

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
            for _ in range(9):
                resid = ResidualBlock(resid)

            conv4 = Conv1DLayer(resid, self.n_base_filter*8, 5)
            pixshuf1 = PixelShuffle1DLayer(conv4, 2)
            gate4 = GatedConv1DLayer(pixshuf1, self.n_base_filter*8)

            #conv5 = Conv1DLayer(gate4, self.n_base_filter*4, 5)
            #pixshuf2 = PixelShuffle1DLayer(conv5, 2)
            # gate = GatedConv1DLayer(pixshuf2, self)

        return net


def ResidualBlock(incoming):
    shit = Conv1DLayer(incoming, 1024, 3, pad="same")
    shit = InstanceNorm1D(shit)
    shit = GatedConv1DLayer(shit, 1024)

    shit = Conv1DLayer(shit, 512, 3, pad="same")
    shit = InstanceNorm1D(shit)
    shit = GatedConv1DLayer(shit, 512)

    return ElemwiseSumLayer([incoming, shit])

