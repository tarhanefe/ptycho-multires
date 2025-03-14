import torch 
import numpy as np 


def add_poisson_noise(image_tensor, noise_factor=1,device='cuda'):
    """
    Add poisson noise to the image.
    """
    image_tensor = np.random.poisson(image_tensor.cpu().detach() * noise_factor) / noise_factor
    image_tensor = torch.tensor(image_tensor).to(torch.complex64).to(device)
    return image_tensor

def add_gaussian_noise(image_tensor, noise_std=1,device='cuda'):
    """"
    Add Gaussian noise to the image.
    """
    image_tensor = image_tensor + (torch.randn(image_tensor.size()).to(device) * noise_std)**2
    return image_tensor