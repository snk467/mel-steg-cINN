import random
import torch
from skimage.util import random_noise


class GaussianNoise:
    """Add Gaussian noise to a tensor"""

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def __call__(self, x: torch.Tensor):
        mean = random.choice(self.mean)
        variance = random.choice(self.variance)
        return torch.tensor(random_noise(x, mode='gaussian', mean=mean, var=variance, clip=True), dtype=x.dtype)
