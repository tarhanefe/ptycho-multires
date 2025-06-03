import torch 
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_image(image_path,max_scale = 9,device = 'cuda'):
    try:
        image = np.load(image_path)
    except:
        image = cv2.imread(image_path,  cv2.IMREAD_GRAYSCALE)
    image = image / 255.
    image = (image - image.min())/(image.max() - image.min())
    image_tensor = torch.tensor(image).double().view(1, 1, 2**max_scale, 2**max_scale)
    image_tensor = torch.exp(1j * image_tensor).to(device)
    return image,image_tensor

