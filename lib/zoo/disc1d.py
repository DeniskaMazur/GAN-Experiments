import theano
import theano.tensor as T

from lasagne.layers import (InputLayer)


class Discriminator1D:

    def __init__(self, generator, real_input_var=T.tensor3, wasserstein=False):
        self.n_input_channels = generator.output_shape[1]
        self.inp_dims = generator.output_shape[2:]
        self.real_inp_var = real_inp_var

        self.wasserstein = wasserstein

        self.generator = generator
        self.model = self._build_network()

    def _build_network():
        class net:
            inp = InputLayer([None, self.n_input_channels] + list(self.inp_dims))

            