from collections import OrderedDict

class BaseGenerator(object):
    """
    Base class for generators
    """
    def __init__(self, n_input_channels, input_dim):
        self.n_input_channels = n_input_channels
        self.input_dim = input_dim

        self.model = self._build_network()

    def _build_network(self):
        """
        Builds lasagne model
        :return: dictionary name: lasagne layer
        """
        pass
