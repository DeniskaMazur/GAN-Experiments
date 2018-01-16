from lasagne.layers import *
from lasagne.nonlinearities import LeakyRectify

from ..layers import *

from collections import OrderedDict

import theano
import theano.tensor as T


class Discriminator:
    """
    A Convolutional Discriminator, can be wasserstein
    """
    def __init__(self, generator, real_inp_var=T.tensor4("real inp"), wasserstein=False):
        """
        Build a Discriminator instance
        :param generator: Generator instance
        :param real_inp_var: tensor4, real input
        :param wasserstein: boolean, true - sigmoid output, false - None
        """
        self.n_input_channels = generator.n_input_channels
        self.inp_dims = generator.output_dims
        self.real_inp_var = real_inp_var

        self.wasserstein = wasserstein

        self.generator = generator
        self.model = self._build_network()

        self.fake_out = get_output(self.model["out"],
                                   {self.model["inp"] : self.generator.output_var})

        self.real_out = get_output(self.model["out"],
                                   {self.model["inp"] : self.real_inp_var})

        self.params = get_all_params(self.model["out"], trainable=True)

    def _build_network(self):
        """
        Build a discriminator network
        :return:
        """
        net = OrderedDict()

        net["inp"] = InputLayer([None, self.n_input_channels] + list(self.inp_dims))

        net["conv1"] = Conv2DLayer(net["inp"], 64, 5, stride=2, pad="same", nonlinearity=LeakyRectify(0.2))
        net["conv2"] = Conv2DLayer(net["conv1"], 128, 5, stride=2, pad="same", nonlinearity=LeakyRectify(0.2))
        net["conv3"] = Conv2DLayer(net["conv2"], 256, 5, stride=2, pad="same", nonlinearity=LeakyRectify(0.2))
        net["conv4"] = Conv2DLayer(net["conv3"], 256, 5, stride=2, pad="same", nonlinearity=LeakyRectify(0.2))
        net["conv5"] = Conv2DLayer(net["conv4"], 256, 5, stride=2, pad="same", nonlinearity=LeakyRectify(0.2))

        net["dense1"] = DenseLayer(net["conv5"], 512, nonlinearity=LeakyRectify(0.2))
        net["dense2"] = DenseLayer(net["dense1"], 512, nonlinearity=LeakyRectify(0.2))

        out_nonlin = None
        if not self.wasserstein:
            out_nonlin = T.nnet.sigmoid

        net["out"] = DenseLayer(net["dense2"], 1, nonlinearity=out_nonlin)
        
        return net