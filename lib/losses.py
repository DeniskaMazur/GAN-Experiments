import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from lasagne.layers import get_output

srng = RandomStreams()


def wasserstein_gradient_penalty_loss(real_input, gen_output, discriminator_output_layer, gp_lambd=10):
    """
    The gradient penalty for the Wasserstein discriminator loss.
    See `Improved Training of Wasserstein GANs`
    (https://arxiv.org/abs/1704.00028) for more details.

    Args:
        real_input: real input
        gen_output: generator output
        discriminator_output_layer: discriminator output layer
        gp_lambd: gradient penalty coefficient
    """
    epsilon = srng.uniform(size=[1])

    x_hat = epsilon * real_input + (1 - epsilon) * gen_output

    real_score = get_output(discriminator_output_layer, real_input).mean()
    gen_score = get_output(discriminator_output_layer, gen_output).mean()
    x_hat_score = get_output(discriminator_output_layer, x_hat).mean()

    gradients = T.grad(x_hat_score, x_hat)[0]
    grad_penalty = ((T.sqrt(T.sum(gradients**2)) - 0.01)**2).mean()

    loss = gen_score - real_score + gp_lambd * grad_penalty

    return loss
