from lasagne.layers import InputLayer, Layer, get_output, get_all_params


class BaseDiscriminator(object):

    def __init__(self, generator, real_input_var):
        """Creates a BaseDiscriminator instance
        
        Arguments:
            generator  -- Your generator
            real_input_var tensor -- the real input variable
        """
        self.generator = generator
        self.input_shape = generator.output_shape

        self.fake_input_var = self.generator.output_var
        self.real_input_var = real_input_var
        
        self.layers = self._build_net()

        self.real_out = get_output(self.layers.out,
                                    {self.layers.inp: self.real_input_var})
        self.fake_out = get_output(self.layers.out,
                                    {self.layers.inp: self.fake_input_var})

        self.params = get_all_params(self.layers.out, trainable=True)
        
    def _build_net(self):
        """"Defines network[summary]
        """

        class net:
            inp = InputLayer(self.input_shape)
            out = Layer(inp)

        return net
