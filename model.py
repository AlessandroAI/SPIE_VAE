import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import time
import sys


class EntropyBottleneck(nn.Module):
    """The prior $P_\theta(Z)$."""

    def __init__(self, z_channels=64):
        super(EntropyBottleneck, self).__init__()
        self.prior_log_variance = torch.nn.Parameter(torch.ones((1, z_channels, 1, 1)))

    def forward(self, z):
        """Returns the term inside the expectation in Eq. 3 of the problem set, i.e.,
        $-\sum_i \log( \int_{z_i-0.5}^{z_i+0.5} p_i(z'_i) dz'_i )$ where p_i is a pdf
        of the i-th dimension of the prior. This allows us to estimate the rate term
        $KL( Q(Z|X=x) || P_\theta(Z) )$ for a box-shaped Q by sampling z from Q."""

        def antiderivative_of_normal(x, inverse_std):
            """Returns the antiderivative of a normal distribution with mean
            zero and standard deviation `1 / inverse_std`, evaluated at x."""
            return 0.5 * torch.erf((1 / np.sqrt(2)) * inverse_std * x)

        inverse_prior_std = torch.exp(-0.5 * self.prior_log_variance)

        p_tilde = (antiderivative_of_normal(z + 0.5, inverse_prior_std)
                   - antiderivative_of_normal(z - 0.5, inverse_prior_std))

        return -p_tilde.log().sum()

    def prior_std(self):
        """Returns a tensor of prior standard deviations."""
        return torch.exp(0.5 * self.prior_log_variance)

    def entropy_per_latent_dim(self):
        """Returns $H_P(Z) / dim(Z)$ (mostly for debugging purpose)."""
        return 0.5 * torch.mean(np.log(2*np.pi) + 1 + self.prior_log_variance)


class EncoderModel(nn.Module):
    """A box-shaped variational distribution $Q(Z|X)$."""

    def __init__(self, hidden_channels=4, z_channels=64):
        super(EncoderModel, self).__init__()
        self.z_shape = [z_channels, 4, 4]
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=hidden_channels,
            kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=z_channels,
            kernel_size=5, stride=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        """Returns the parameters that define $Q(Z|X=x)$ for given $x$."""
        hidden = torch.relu(self.conv1(x))
        q_mean = self.conv2(hidden)
        return q_mean

    def reparameterize(self, q_mean):
        """Draws $z ~ Q(Z|X=x)$ using the reparameterization trick."""
        # - the method `torch.rand_like` draws uniform samples from the interval [0, 1).
        # - for the VAE on the last problem set (with a Gaussian variational distribution),
        #   we had the following implementation:
        #   q_std = torch.exp(0.5 * q_log_variance) # Using torch.exp ensures that std > 0.
        #   eps = torch.randn_like(q_std) # `randn` is standard normal distribution.
        #   return q_mean + q_std * eps
        #
        eps = torch.rand_like(q_mean) - 0.5 # uniform within [-0.5, 0.5] for each coordinate
        return q_mean + eps



class DecoderModel(nn.Module):
    """Maps z to a reconstructed image."""

    def __init__(self, z_channels=64, hidden_channels=4):
        super(DecoderModel, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=z_channels, out_channels=hidden_channels,
            kernel_size=5, stride=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=hidden_channels, out_channels=3,
            kernel_size=5, stride=2, output_padding=1)
        nn.init.xavier_uniform_(self.deconv1.weight)
        nn.init.xavier_uniform_(self.deconv2.weight)

    def forward(self, z):
        """Returns a reconstructed image for given $z$."""
        hidden = torch.relu(self.deconv1(z))
        reconstruction = torch.clip(self.deconv2(hidden), 0, 1)
        return reconstruction

    def distortion(self, reconstruction, x):
        """Returns the 2-norm reconstruction error."""
        # set, where $g(z)$ is given by the argument `reconstrution`.
        #
        return torch.sum((reconstruction - x)**2)


def bit_rate_and_reconstruction(encoder_model, decoder_model, entropy_bottleneck, x):
    """Executes a round trip x --> z --> x';
    returns the estimated bit rate (to base 2), the distortion,
    the reconstructed images, and the total latent dimension."""
    q_mean = encoder_model(x)
    z = encoder_model.reparameterize(q_mean)
    bit_rate = entropy_bottleneck(z) / np.log(2)
    reconstructions = decoder_model(z)


    #bit_rate_x_given_z = -decoder_model.log_likelihood(logits, x) / np.log(2)
    #

    distortion = decoder_model.distortion(reconstructions, x)
    # Note: calculating `bit_rate_x_given_z` does not make sense for our lossy
    # setup because we no longer encode the true `x` using the likelihood model.

    return bit_rate, distortion, reconstructions, z.numel()
