import theano.tensor as T

from lasagne.layers import Layer, Conv2DLayer, NonlinearityLayer
from lasagne.nonlinearities import LeakyRectify


class InstanceNorm(Layer):
    def __init__(self, epsilon, **kwargs):
        self.epsilon = epsilon

    def get_output_for(self, input, **kwargs):
        mean = T.mean(input, axis=[2, 3])
        var = T.var(input, axis=[2, 3])

        return (input - mean) / T.sqrt(var + self.epsilon)


def ResidualBlock(input, filter_size):
    conv1 = Conv2DLayer(input, input.output_shape[1], filter_size, pad="same", nonlinearity=None)
    norm = InstanceNorm(conv1)
    relu = NonlinearityLayer(norm, LeakyRectify(0.2))
    conv2 = Conv2DLayer(relu, input.output_shape[1], filter_size, pad="same", nonlinearity=None)

    return conv2
