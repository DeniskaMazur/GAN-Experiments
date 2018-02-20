import theano.tensor as T

from lasagne.layers import Layer, Conv1DLayer, Conv2DLayer, get_output

from .pixshuff import pixel_shuffle1d


class InstanceNorm2D(Layer):
    def __init__(self, incoming, epsilon=1e-5, **kwargs):
        super(InstanceNorm2D, self).__init__(incoming, **kwargs)

        self.epsilon = epsilon

    def get_output_for(self, input, **kwargs):
        mean = T.mean(input, axis=[2, 3], keepdims=True)
        var = T.var(input, axis=[2, 3], keepdims=True)

        return (input - mean) / T.sqrt(var + self.epsilon)

    def get_output_shape_for(self, input_shape):
        return input_shape


class InstanceNorm1D(Layer):
    def __init__(self, incoming, epsilon=1e-5, **kwargs):
        super(InstanceNorm1D, self).__init__(incoming, **kwargs)

        self.epsilon = epsilon

    def get_output_for(self, input, **kwargs):
        mean = T.mean(input, axis=2, keepdims=True)
        var = T.var(input, axis=2, keepdims=True)

        return (input - mean) / T.sqrt(var + self.epsilon)

    def get_output_shape_for(self, input_shape):
        return input_shape


class GatedConv1DLayer(Layer):
    """
    Implementation of https://arxiv.org/pdf/1612.08083.pdf

    H_{l+1} = (H_l * W_l + b_l) * sigmoid(H_l + V_l + c_l)
    """
    def __init__(self, incoming, filter_size=1, pad="same", nonlinearity=None, **kwargs):
        """
        Creates a GatedConv1DLayer instance
        :param incoming: incoming layer
        :param num_filters: int, number of conv filters
        :param filter_size: int, size of conv filters
        :param nonlinearity: activation function for h
        """
        super(GatedConv1DLayer, self).__init__(incoming, **kwargs)

        num_filters = incoming.output_shape[1]

        h = Conv1DLayer(incoming, num_filters, filter_size,
                        pad=pad, nonlinearity=nonlinearity, **kwargs)
        g = Conv1DLayer(incoming, num_filters, filter_size,
                            pad=pad, nonlinearity=T.nnet.sigmoid, **kwargs)

        self.h = get_output(h)
        self.g = get_output(g)

        self.get_output_shape_for = g.get_output_shape_for

    def get_output_for(self, input, **kwargs):
        return self.h * self.g
    

class GatedConv2DLayer(Layer):
    """
    Implementation of https://arxiv.org/pdf/1612.08083.pdf

    H_{l+1} = (H_l * W_l + b_l) * sigmoid(H_l + V_l + c_l)
    """
    def __init__(self, incoming, filter_size=1, pad="same", nonlinearity=None, **kwargs):
        """
        Creates a GatedConv1DLayer instance
        :param incoming: incoming layer
        :param num_filters: int, number of conv filters
        :param filter_size: int, size of conv filters
        :param nonlinearity: activation function for h
        """
        super(GatedConv1DLayer, self).__init__(incoming, **kwargs)

        num_filters = incoming.output_shape[1]

        h = Conv2DLayer(incoming, num_filters, filter_size,
                        pad=pad, nonlinearity=nonlinearity, **kwargs)
        g = Conv2DLayer(incoming, num_filters, filter_size,
                            pad=pad, nonlinearity=T.nnet.sigmoid, **kwargs)

        self.h = get_output(h)
        self.g = get_output(g)

        self.get_output_shape_for = g.get_output_shape_for

    def get_output_for(self, input, **kwargs):
        return self.h * self.g


class PixelShuffle1DLayer(Layer):
    """
    Inplementation of <the atricle>

    Rearranges elements in a tensor of shape ``[*, C*r, T]`` to a
    tensor of shape ``[C, T*r]
    """

    def __init__(self, incoming, upscale_factor, **kwargs):
       super(PixelShuffle1DLayer, self).__init__(incoming, **kwargs)

       self.upscale_factor = upscale_factor

    def get_output_for(self, input, **kwargs):
        return pixel_shuffle1d(input, self.upscale_factor)

    def get_output_shape_for(self, input_shape):
        batch, chan, time = input_shape

        return (batch, chan / self.upscale_factor, time * self.upscale_factor)
