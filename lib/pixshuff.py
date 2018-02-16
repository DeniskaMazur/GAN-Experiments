def pixel_shuffle2d(X, upscale_factor):
    """
    Rearranges elements in a tensor of shape ``[*, C*r^2, H, W]`` to a
    tensor of shape ``[C, H*r, W*r]

    :param X - Tensor4 [batch_size, channels, height, width]
    :param upscale_factor int

    :return Tensor4
    """

    batch_size, channels, in_height, in_width = X.shape

    channels //= upscale_factor ** 2

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    X = X.reshape((batch_size, channels, upscale_factor, upscale_factor,
                  in_height, in_width))

    X = X.transpose(0, 1, 4, 2, 5, 3)

    return X.reshape((batch_size, channels, out_height, out_width))


def pixel_shuffle1d(X, upscale_factor):
    batch_size, channels, in_length = X.shape

    channels //= upscale_factor

    out_length = in_length * upscale_factor

    X = X.reshape((batch_size, channels, upscale_factor, in_length))
    X = X.transpose(0, 1, 3, 2)

    return X.reshape((batch_size, channels, out_length))
