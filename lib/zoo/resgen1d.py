from .base_generator import BaseGenerator

from lasagne.nonlinearities import LeakyRectify
from lasagne.layers import (ElemwiseSumLayer, InputLayer, NonlinearityLayer,
                             Conv1DLayer, get_output, get_all_params)

from ..layers import GatedConv1DLayer, InstanceNorm1D
from ..layers import PixelShuffle1DLayer


class Sound2SoundNet(BaseGenerator):

    def __init__(self, inp_dims, n_base_filter=128, n_residual=7, nonlinearity=LeakyRectify(0.2), norm_func=InstanceNorm1D, **kwargs):
        """Creates a ResidualGenerator1D Instance
        
        Arguments:
            inp_dims list of int -- input shape
            n_residual int -- number of residual blocks
        
        Keyword Arguments:
            n_base_filter int -- number of base filters (default: {128})
            nonlinearity lasagne.nonlineatities -- convolution nonlinearity (default: {LeakyRectify})
        """
        
        # hyperparams
        self.n_base_filter = n_base_filter        
        self.n_residual = 7
        self.nonlinearity = nonlinearity
        self.norm_func = norm_func

        super(Sound2SoundNet, self).__init__(inp_dims, **kwargs)

    def _build_net(self):
        class net:
            inp = InputLayer(self.inp_dims)
            
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

            out = Conv1DLayer(gate5, self.inp_dims[1], 15, pad="same")

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
