import theano.tensor as T

from lasagne.layers import Layer, Conv2DLayer, NonlinearityLayer, ElemwiseSumLayer
from lasagne.nonlinearities import LeakyRectify


class InstanceNorm(Layer):
    def __init__(self, incoming, epsilon=1e-5, **kwargs):
        super(InstanceNorm, self).__init__(incoming, **kwargs)

        self.epsilon = epsilon

    def get_output_for(self, input, **kwargs):
        mean = T.mean(input, axis=[2, 3], keepdims=True)
        var = T.var(input, axis=[2, 3], keepdims=True)

        return (input - mean)

    def get_output_shape_for(self, input_shape):
        return input_shape


def ResidualBlock(input, filter_size):
    conv1 = Conv2DLayer(input, input.output_shape[1], filter_size, pad="same", nonlinearity=None)
    norm = InstanceNorm(conv1)
    relu = NonlinearityLayer(norm, LeakyRectify(0.2))
    conv2 = Conv2DLayer(relu, input.output_shape[1], filter_size, pad="same", nonlinearity=None)

    return ElemwiseSumLayer([input, conv2])
