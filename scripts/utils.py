import matplotlib.pyplot as plt
import numpy as np
import torch

def extract_data(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):  # If the item is a list, recurse into it
            result.extend(extract_data(item))
        else:  # If the item is not a list, add it to the result
            result.append(item)
    return result


def unwrap_2d(phase):
    """
    Unwraps a 2D phase array using NumPy's 1D unwrap function.
    
    Parameters:
    phase (numpy array): The 2D phase array to be unwrapped.
    
    Returns:
    unwrapped_phase (numpy array): The 2D unwrapped phase array.
    """
    # Unwrap along the first axis (rows)
    unwrapped_phase = np.unwrap(phase, axis=0)
    unwrapped_phase = np.unwrap(unwrapped_phase, axis=1)
    
    return unwrapped_phase

def plot_results(model,image):
    plt.figure(figsize=(55, 10),dpi = 600)    

    plt.subplot(1, 5, 1)
    plt.imshow(image,cmap='gray')
    plt.title("Original Image")
    plt.colorbar()


    plt.subplot(1, 5, 2)
    phase = torch.angle(model.sols[-1][0,0,:,:].to('cpu'))
    phase = unwrap_2d(phase)
    plt.imshow(phase,cmap = 'gray')
    plt.title("Reconstructed Image")
    plt.colorbar()


    plt.subplot(1, 5, 3)
    plt.imshow(np.abs(np.array(image)-np.array(phase)),cmap='gray')
    plt.title("Loss")
    plt.colorbar()

    plt.subplot(1, 5, 4)
    plt.title("Log(Loss) vs Iterations")
    plt.plot(np.log10(extract_data(model.measures["loss"])),label = "Loss",color = 'blue')

    plt.subplot(1, 5, 5)
    plt.title("LR vs Iterations")
    plt.plot(model.lr_list,label = "LR",color = 'red')
    return None 