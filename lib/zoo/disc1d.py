from .base_discriminator import BaseDiscriminator
import theano.tensor as T
from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer, DenseLayer, get_output)
from ..layers import GatedConv2DLayer, InstanceNorm1D


class Discriminator1D(BaseDiscriminator):
    
     def __init__(self, generator, real_input_var=T.tensor3(), n_base_filter=128, norm_func=InstanceNorm1D, wasserstein=False):
         """Creates a Discriminator1D instance
        
        Arguments:
            generator -- Generator instance
        
        Keyword Arguments:
            real_input_var theano.tensor.tensor3 -- Real input (default: {T})
            n_base_filter int -- Base number of filters (default: {128})
            norm_func lasagne.layers.Layer -- Normalization layer (default: {InstanceNorm1D})
            wasserstein boolean -- Uses linear output if true, sigmoid if false (default: {False})
        """
        self.n_base_filter = n_base_filter
        self.norm_func = norm_func
        self.wasserstein = wasserstein
        
        super(Discriminator1D, self).__init__(generator, real_input_var)

    def _build_net():
        class net:
            inp = InputLayer(self.input_shape)

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

            out = DenseLayer(gate4, 1, nonlinearity=nonlinearity)
        
        return net
