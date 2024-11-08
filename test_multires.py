#%%
from ptycho.multires.class_multiressolver import *
import matplotlib.pyplot as plt
import torch.nn.functional as func
import torch
from ptycho.tools.ptychography import Ptychography2_v2 as Ptychography2

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
    
    scale = 7
    I_in = 15*np.array([1, 15, 10, 5, 2, 5, 10, 30, 100])
    I_out = 30*np.array([0, 0, 0, 0, 7, 7, 6, 5, 80])
    #I_out = 300*np.array([0, 0, 0, 0, 0, 0, 0, 0, 10])
    cycle = [0, -1, -1, -1, -1,1,  1, 1, 1]
    device = "cuda"
    lmbda = 1e-10
    LR = 0.01
    tol = [1e-10] * 9
    tol_in = [1e-10] * 9
    img_count = 10*10

    linOperator = Ptychography2(in_shape=(2**scale-1, 2**scale-1), n_img=img_count, probe_type='square',
                           probe_radius=30, defocus_factor=0, 
                           fov=170, threshold=0.3, device=device)

    image = plt.imread('images/peppers_reduced.jpg')[:2**scale-1, :2**scale-1] / 255
    image_tensor = torch.tensor(image).double().to(device).view(1, 1, 2**scale-1, 2**scale-1)
    image_tensor_ = torch.exp(1j * image_tensor)
    #image_tensor_ = torch.stack([image_tensor, image_tensor], dim=-1)
    #image_tensor_ = torch.view_as_complex(image_tensor_).to(torch.complex64)
    
    #Initiate the MultiRes class with the inital scale.
    multires = MultiRes(scale, device)
    loss = Loss(linOperator,linOperator.apply(image_tensor_), lmbda = lmbda)

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
image = plt.imread('images/peppers_reduced.jpg')[:2**7-1, :2**7-1] / 255
model = run_test()

# %%
plot_results(model,image)



# %%
