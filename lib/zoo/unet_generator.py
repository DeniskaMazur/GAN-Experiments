from collections import OrderedDict
from lasagne.layers import (InputLayer, ConcatLayer, Pool2DLayer, get_output,
                            DropoutLayer, Deconv2DLayer, batch_norm)

from lasagne.layers import Conv2DLayer as ConvLayer

import lasagne
from lasagne.init import HeNormal

import theano
import theano.tensor as T


class UnetGenerator:
    """
    Unet Generator, nuff said
    """

    def __init__(self, n_input_channels=1, pad='same', nonlinearity=lasagne.nonlinearities.elu, input_dim=(128, 128),
                 base_n_filters=64, do_dropout=False):
        self.n_input_channels = n_input_channels
        self.pad = pad
        self.nonlinearity = nonlinearity
        self.input_dim = input_dim
        self.base_n_filters = base_n_filters
        self.do_dropout = do_dropout

        self.model = self._build_network()
        self.input_var = self.model["input"].input_var
        self.output_var = get_output(self.model["output"])
        self.output_dims = self.model["output"].output_shape[2:]
        self.generate = theano.function([self.input_var], self.output_var)

    def _build_network(self):
        net = OrderedDict()
        net['input'] = InputLayer((None, self.n_input_channels, self.input_dim[0], self.input_dim[1]))

        net['contr_1_1'] = batch_norm(
            ConvLayer(net['input'], self.base_n_filters, 3, nonlinearity=self.nonlinearity, pad=self.pad, W=HeNormal(gain="relu")))
        net['contr_1_2'] = batch_norm(
            ConvLayer(net['contr_1_1'], self.base_n_filters, 3, nonlinearity=self.nonlinearity, pad=self.pad, W=HeNormal(gain="relu")))
        net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

        net['contr_2_1'] = batch_norm(
            ConvLayer(net['pool1'], self.base_n_filters * 2, 3, nonlinearity=self.nonlinearity, pad=self.pad, W=HeNormal(gain="relu")))
        net['contr_2_2'] = batch_norm(
            ConvLayer(net['contr_2_1'], self.base_n_filters * 2, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                      W=HeNormal(gain="relu")))
        net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)

        net['contr_3_1'] = batch_norm(
            ConvLayer(net['pool2'], self.base_n_filters * 4, 3, nonlinearity=self.nonlinearity, pad=self.pad, W=HeNormal(gain="relu")))
        net['contr_3_2'] = batch_norm(
            ConvLayer(net['contr_3_1'], self.base_n_filters * 4, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                      W=HeNormal(gain="relu")))
        net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)

        net['contr_4_1'] = batch_norm(
            ConvLayer(net['pool3'], self.base_n_filters * 8, 3, nonlinearity=self.nonlinearity, pad=self.pad, W=HeNormal(gain="relu")))
        net['contr_4_2'] = batch_norm(
            ConvLayer(net['contr_4_1'], self.base_n_filters * 8, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                      W=HeNormal(gain="relu")))
        l = net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)
        # the paper does not really describe where and how dropout is added. Feel free to try more options
        if self.do_dropout:
            l = DropoutLayer(l, p=0.4)

        net['encode_1'] = batch_norm(
            ConvLayer(l, self.base_n_filters * 16, 3, nonlinearity=self.nonlinearity, pad=self.pad, W=HeNormal(gain="relu")))
        net['encode_2'] = batch_norm(
            ConvLayer(net['encode_1'], self.base_n_filters * 16, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                      W=HeNormal(gain="relu")))
        net['upscale1'] = batch_norm(
            Deconv2DLayer(net['encode_2'], self.base_n_filters * 16, 2, 2, crop="valid", nonlinearity=self.nonlinearity,
                          W=HeNormal(gain="relu")))

        net['concat1'] = ConcatLayer([net['upscale1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
        net['expand_1_1'] = batch_norm(
            ConvLayer(net['concat1'], self.base_n_filters * 8, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                      W=HeNormal(gain="relu")))
        net['expand_1_2'] = batch_norm(
            ConvLayer(net['expand_1_1'], self.base_n_filters * 8, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                      W=HeNormal(gain="relu")))
        net['upscale2'] = batch_norm(
            Deconv2DLayer(net['expand_1_2'], self.base_n_filters * 8, 2, 2, crop="valid", nonlinearity=self.nonlinearity,
                          W=HeNormal(gain="relu")))

        net['concat2'] = ConcatLayer([net['upscale2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
        net['expand_2_1'] = batch_norm(
            ConvLayer(net['concat2'], self.base_n_filters * 4, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                      W=HeNormal(gain="relu")))
        net['expand_2_2'] = batch_norm(
            ConvLayer(net['expand_2_1'], self.base_n_filters * 4, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                      W=HeNormal(gain="relu")))
        net['upscale3'] = batch_norm(
            Deconv2DLayer(net['expand_2_2'], self.base_n_filters * 4, 2, 2, crop="valid", nonlinearity=self.nonlinearity,
                          W=HeNormal(gain="relu")))

        net['concat3'] = ConcatLayer([net['upscale3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
        net['expand_3_1'] = batch_norm(
            ConvLayer(net['concat3'], self.base_n_filters * 2, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                      W=HeNormal(gain="relu")))
        net['expand_3_2'] = batch_norm(
            ConvLayer(net['expand_3_1'], self.base_n_filters * 2, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                      W=HeNormal(gain="relu")))
        net['upscale4'] = batch_norm(
            Deconv2DLayer(net['expand_3_2'], self.base_n_filters * 2, 2, 2, crop="valid", nonlinearity=self.nonlinearity,
                          W=HeNormal(gain="relu")))

        net['concat4'] = ConcatLayer([net['upscale4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
        net['expand_4_1'] = batch_norm(
            ConvLayer(net['concat4'], self.base_n_filters, 3, nonlinearity=self.nonlinearity, pad=self.pad, W=HeNormal(gain="relu")))
        net['expand_4_2'] = batch_norm(
            ConvLayer(net['expand_4_1'], self.base_n_filters, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                      W=HeNormal(gain="relu")))

        net['output'] = ConvLayer(net['expand_4_2'], self.n_input_channels, 1, nonlinearity=T.tanh)

        return net