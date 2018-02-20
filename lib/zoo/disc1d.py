import theano
import theano.tensor as T

from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer, DenseLayer)

from ..layers import GatedConv2DLayer, InstanceNorm1D


class Discriminator1D:

    def __init__(self, generator, real_input_var=T.tensor3, n_base_filter=128, norm_func=InstanceNorm1D, wasserstein=False):
        self.n_input_channels = generator.output_shape[1]
        self.inp_dims = generator.output_shape[2:]
        self.n_base_filter = n_base_filter
        self.wasserstein = wasserstein

        self.real_inp_var = real_input_var
        self.norm_func = norm_func

        self.generator = generator
        #self.model = self._build_network()

    def _build_network(self):
        class net:
            inp = InputLayer([None, self.n_input_channels] + list(self.inp_dims))

            # [None, 1, channels, time]
            resh = ReshapeLayer(inp, [-1, 1] + list(self.inp_dims))

            conv1 = Conv2DLayer(resh, self.n_base_filter, 3, stride=(1, 2), pad="same")
            gate1 = GatedConv2DLayer(conv1)

            conv2 = Conv2DLayer(gate1, self.n_base_filter*2, 3, stride=(2, 2), pad="same")
            norm1 = self.norm_func(conv2)
            gate2 = GatedConv2DLayer(norm1)

            conv3 = Conv2DLayer(gate2, self.n_base_filter*4, 3, stride=(2, 2), pad="same")
            norm2 = self.norm_func(conv3)
            gate3 = GatedConv2DLayer(norm2)

            conv4 = Conv2DLayer(gate3, self.n_base_filter*8, (6, 3), stride=(1, 2), pad="same")
            norm3 = self.norm_func(conv4)
            gate4 = GatedConv2DLayer(norm3)

            nonlinearity = None
            if not self.wasserstein:
                nonlinearity = T.nnet.sigmoid

            #out = DenseLayer(gate4, 1, nonlinearity=nonlinearity)
            
        return net