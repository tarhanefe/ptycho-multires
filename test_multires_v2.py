#%%
from ptycho_v2.multires_v2.class_multiressolver import *
import matplotlib.pyplot as plt
import torch
from ptycho_v2.tools_v2.ptychography import Ptychography as Ptychography

#%%
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
    
    # Unwrap along the second axis (columns)
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


#%%

def run_test():
    # Setting the operating device as cpu and the inital as ...
    max_scale = 7
    max_probe_size = 32
    max_shift = 8
    device = 'cuda'
    I_in = 15*np.array([1, 15, 10, 5, 10, 30, 100])
    I_out = 20*np.array([0, 0, 0, 7, 6, 5, 80])
    #I_out = 300*np.array([0, 0, 0, 0, 0, 0, 0, 0, 10])
    cycle = [0, -1, -1, -1,  1, 1, 1]
    lmbda = 0
    LR = 0.1
    tol = [1e-10] * 9
    tol_in = [1e-10] * 9

    linOperator = Ptychography(max_scale = max_scale,max_probe_size = max_probe_size ,max_shift = max_shift,device=device)

    image = plt.imread('images/peppers_reduced.jpg')/ 255
    image_tensor = torch.tensor(image).double().to(device).view(1, 1, 2**max_scale, 2**max_scale)
    image_tensor_ = torch.exp(1j * image_tensor)
    #Initiate the MultiRes class with the inital scale.
    multires = MultiRes(max_scale, device)
    #loss = Loss(linOperator,linOperator.apply(image_tensor_), lmbda = lmbda)
    loss = Loss(linOperator,linOperator.apply(image_tensor_))
    model = MultiResSolver(multires, loss, LR = LR,
                           I_in = I_in,
                           I_out = I_out,
                           tol = tol,
                           tol_in = tol_in,
                           cycle = cycle,
                           l1_type = "l1_row")
    
    model.solve_multigrid()
    model.print_time()
    return model

# %%
image = plt.imread('images/peppers_reduced.jpg')/255
model = run_test()

# %%
plot_results(model,image)



# %%
