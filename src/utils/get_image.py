import torch 
import matplotlib.pyplot as plt
import numpy as np

def get_image(image_path,max_scale = 9,device = 'cuda'):
    image = np.load(image_path)
    image = image / 255
    image = (image - image.min())/(image.max() - image.min())
    image_tensor = torch.tensor(image).double().to(device).view(1, 1, 2**max_scale, 2**max_scale)
    image_tensor = torch.exp(1j * image_tensor)
    return image,image_tensor

