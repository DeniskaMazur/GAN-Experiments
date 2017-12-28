from lasagne.layers import (InputLayer, ConcatLayer, Pool2DLayer, ReshapeLayer, DimshuffleLayer, NonlinearityLayer,
                            DropoutLayer, Deconv2DLayer, batch_norm, get_output, get_all_params)

from lasagne.layers import Conv2DLayer as ConvLayer

import lasagne
from lasagne.init import HeNormal

import theano


class Unet:
    def __init__(self, pad='same', nonlinearity=lasagne.nonlinearities.elu, input_dim=(128, 128), base_n_filters=64, do_dropout=False):
        class net:
            input_l = InputLayer((None, 3, input_dim[0], input_dim[1]))

            contr_1_1 = batch_norm(ConvLayer(input_l, base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            contr_1_2 = batch_norm(ConvLayer(contr_1_1, base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            pool1 = Pool2DLayer(contr_1_2, 2)

            contr_2_1 = batch_norm(ConvLayer(pool1, base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            contr_2_2 = batch_norm(ConvLayer(contr_2_1, base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            pool2 = Pool2DLayer(contr_2_2, 2)

            contr_3_1 = batch_norm(ConvLayer(pool2, base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            contr_3_2 = batch_norm(ConvLayer(contr_3_1, base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            pool3 = Pool2DLayer(contr_3_2, 2)

            contr_4_1 = batch_norm(ConvLayer(pool3, base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            contr_4_2 = batch_norm(ConvLayer(contr_4_1, base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            l = pool4 = Pool2DLayer(contr_4_2, 2)

            # the paper does not really describe where and how dropout is added. Feel free to try more options
            if do_dropout:
                l = DropoutLayer(l, p=0.4)

            encode_1 = batch_norm(ConvLayer(l, base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            encode_2 = batch_norm(ConvLayer(encode_1, base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            upscale1 = batch_norm(Deconv2DLayer(encode_2, base_n_filters*16, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))

            concat1 = ConcatLayer([upscale1, contr_4_2], cropping=(None, None, "center", "center"))
            expand_1_1 = batch_norm(ConvLayer(concat1, base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            expand_1_2 = batch_norm(ConvLayer(expand_1_1, base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            upscale2 = batch_norm(Deconv2DLayer(expand_1_2, base_n_filters*8, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))

            concat2 = ConcatLayer([upscale2, contr_3_2], cropping=(None, None, "center", "center"))
            expand_2_1 = batch_norm(ConvLayer(concat2, base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            expand_2_2 = batch_norm(ConvLayer(expand_2_1, base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            upscale3 = batch_norm(Deconv2DLayer(expand_2_2, base_n_filters*4, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))

            concat3 = ConcatLayer([upscale3, contr_2_2], cropping=(None, None, "center", "center"))
            expand_3_1 = batch_norm(ConvLayer(concat3, base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            expand_3_2 = batch_norm(ConvLayer(expand_3_1, base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            upscale4 = batch_norm(Deconv2DLayer(expand_3_2, base_n_filters*2, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))

            concat4 = ConcatLayer([upscale4, contr_1_2], cropping=(None, None, "center", "center"))
            expand_4_1 = batch_norm(ConvLayer(concat4, base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
            expand_4_2 = batch_norm(ConvLayer(expand_4_1, base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))

            output = ConvLayer(expand_4_2, 3, 1, nonlinearity=None)

        self.net = net

        self.input_var = self.net.input_l.input_var
        self.out_var = get_output(self.net.output)

        self.params = get_all_params(self.net.output, trainable=True)

        self.generate = theano.function([self.input_var], self.out_var, allow_input_downcast=True)


class Discriminator:
    def __init__(self, generator):
        class net:
            input_l = InputLayer(generator.net.input_l)