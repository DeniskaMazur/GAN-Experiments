from lasagne.layers import InputLayer, Layer, get_output, get_all_params

import theano

class BaseGenerator(object):
    """Abstract generator
    """

    def __init__(self, inp_dims):
        """Creates a BaseGenerator instance
        
        Arguments:
            n_dims list of int -- dimensions of input data
        """
        # hyperparams
        self.inp_dims = inp_dims

        # define network
        self.layers = self._build_net()

        # get variables
        self.input_var = self.layers.inp.input_var
        self.output_var = get_output(self.layers.out)
        
        # generation function
        self.generate = theano.function([self.input_var], self.output_var)

        # weights
        self.params = get_all_params(self.layers.out, trainable=True)

    def _build_net(self):
        """Defines network
        
        Returns:
            classobj -- a classobj, containg your networks layers 
        """

        class net:
            inp = InputLayer(self.inp_dims)
            out = Layer(inp)

        return net
