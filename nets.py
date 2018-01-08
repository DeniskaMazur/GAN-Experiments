from lasagne.layers import *
from lasagne.nonlinearities import LeakyRectify

from layers import *

from collections import OrderedDict

import theano
import theano.tensor as T


class ResGenerator:
    """
    An encoder-decoder generator, uses residual blocks on bottleneck
    """
    def __init__(self, n_chan=3, filter_size=(128, 128), input_var=None):
        """
        Builds A Generator instance
        :param n_chan:
        :param filter_size:
        """

        self.n_chan = 3
        self.filter_size = filter_size
        self.input_var = input_var

        self.layers = self._build_network()

        self.inp_var = self.layers["inp"].input_var
        self.output_var = get_output(self.layers["out"])
        self.params = get_all_params(self.layers["out"], trainable=True)

        self.generate = theano.function([self.inp_var],
                                        self.output_var,
                                        allow_input_downcast=True)

    def _build_network(self):
        net = OrderedDict()

        net["inp"] = InputLayer(tuple([None, self.n_chan]) + self.filter_size, self.input_var)

        net["conv1"] = Conv2DLayer(net["inp"], 32, 7, nonlinearity=LeakyRectify(0.2))

        net["conv2"] = Conv2DLayer(net["conv1"], 64, 5, stride=2, pad="same", nonlinearity=None)
        net["norm1"] = InstanceNorm(net["conv2"])
        net["lref1"] = NonlinearityLayer(net["norm1"], nonlinearity=LeakyRectify(0.2))

        net["conv3"] = Conv2DLayer(net["lref1"], 64, 5, stride=2, pad="same", nonlinearity=None)
        net["norm2"] = InstanceNorm(net["conv3"])
        net["lref2"] = NonlinearityLayer(net["norm2"], nonlinearity=LeakyRectify(0.2))

        net["resid1"] = ResidualBlock(net["lref2"], 3)
        for i in range(8):
            net["resid%d" % (i+2)] = ResidualBlock(net["resid%d" % (i+1)], 3)

        net["dec1"] = TransposedConv2DLayer(net["resid9"], 64, 3, stride=2, nonlinearity=None)
        net["norm3"] = InstanceNorm(net["dec1"])
        net["lref3"] = NonlinearityLayer(net["norm3"], nonlinearity=LeakyRectify(0.2))

        net["dec2"] = TransposedConv2DLayer(net["lref3"], 32, 4, stride=2, nonlinearity=None)
        net["norm4"] = InstanceNorm(net["dec2"])
        net["lref3"] = NonlinearityLayer(net["norm4"], nonlinearity=LeakyRectify(0.2))

        net["out"] = Conv2DLayer(net["lref3"], 3, 3, pad="same", nonlinearity=T.tanh)

        return net

    def generate_showable(self, src_image):
        pic = self.generate(src_image)
        return pic.transpose(0, 2, 3, 1)

    def get_output(self, input_var):
        return get_output(self.layers["out"], {self.layers["inp"] : input_var})


class Discriminator:
    """
    A Convolutional Discriminator, can be wasserstein
    """
    def __init__(self, generator, real_inp_var=T.tensor4("real inp"), wasserstein=False):
        """
        Build a Discriminator instance
        :param generator: Generator to discriminate
        :param real_inp_var: tensor4, real input
        :param wasserstein: boolean, true - sigmoid output, false - None
        """
        self.inp_n_chan = generator.n_chan
        self.inp_imsize = generator.filter_size
        self.real_inp_var = real_inp_var

        self.wasserstein = wasserstein

        self.generator = generator
        self.layers = self._build_network()

        self.fake_out = get_output(self.layers["out"],
                                   {self.layers["inp"] : self.generator.output_var})

        self.real_out = get_output(self.layers["out"],
                                   {self.layers["inp"] : real_inp_var})

        self.params = get_all_params(self.layers["out"], trainable=True)

    def _build_network(self):
        """
        Build a discriminator network
        :return:
        """
        net = OrderedDict()

        net["inp"] = InputLayer([None, self.inp_n_chan] + list(self.inp_imsize))

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