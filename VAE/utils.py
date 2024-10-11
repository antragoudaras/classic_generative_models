import torch
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization got a negative std as input. "
                                    
    
    latent = torch.randn_like(std) * std + mean
    return latent


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    KLD = torch.exp(torch.mul(2,log_std)) + torch.pow(mean,2) - torch.ones_like(mean) - torch.mul(2,log_std)
    KLD = torch.mul(0.5, torch.sum(KLD, dim=-1))
    
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        Negative log likelihood in bits per dimension for the given image.
    """

    return torch.mul(elbo, torch.log2(torch.exp(torch.ones_like(elbo)))) / torch.prod(torch.tensor(img_shape[1:]))


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        grid - Grid of images for manifold representation.
    """

   

    grid_lower = 0.5 / grid_size
    grid_upper = (grid_size - 0.5) / grid_size

    percent_range = torch.linspace(grid_lower, grid_upper, steps=grid_size)
    normal = torch.distributions.Normal(loc=0, scale=1)
    icdf = normal.icdf(percent_range)
    z_x, z_y = torch.meshgrid(icdf, icdf)
    z = torch.cat(tuple(torch.dstack([z_x, z_y]))) # shape: (grid_size**2, 2)

    x = decoder(z).softmax(dim=1) # shape: (grid_size**2, 16, 28, 28)
    shape_x = x.shape
    x = torch.flatten(x.permute(0, 2, 3, 1), start_dim=0, end_dim=-2) # shape: (grid_size**2*28*28, 16)
    
    sample = torch.multinomial(x, num_samples=1)
    sample = sample.reshape(shape_x[0], shape_x[2], shape_x[3]).unsqueeze(1) #shape: (grid_size**2, 1, 28, 28)
    sample = sample.float() / 15 # normalize to [0,1] 4-bit images

    
    grid = make_grid(sample, nrow=grid_size, normalize=True, value_range=(0, 1))
    grid = grid.detach().cpu()

    return grid