#%%
from ptycho.multires.class_multiressolver import *
import matplotlib.pyplot as plt
import torch.nn.functional as func
import torch
from ptycho.tools.ptychography import Ptychography2


def run_test():
    # Setting the operating device as cpu and the inital as ...
    
    scale = 9
    I_in = 2*np.array([1, 15, 10, 5, 2, 5, 10, 30, 100])
    I_out = 20*np.array([0, 0, 0, 0, 7, 7, 7, 5, 4])
    #I_out = 10*np.array([0, 0, 0, 0, 7, 0, 0, 0, 0])
    cycle = [0, -1, -1, -1, -1,1,  1, 1, 1]
    device = "cuda"
    lmbda = 1e-4
    LR = 1e-1
    tol = [1e-10] * 9
    tol_in = [1e-10] * 9


    linOperator = Ptychography2(in_shape=(2**scale-1, 2**scale-1), n_img=100, probe_type='defocus pupil',
                           probe_radius=80, defocus_factor=0, 
                           fov=520, threshold=0.3, device=device)

    image = plt.imread('images/peppers.jpg')[:511, :511] / 255
    image_tensor = torch.tensor(image).double().to(device).view(1, 1, 511, 511)
    image_tensor = torch.stack([image_tensor, image_tensor], dim=-1)
    image_tensor = torch.view_as_complex(image_tensor).to(torch.complex64)
    
    #Initiate the MultiRes class with the inital scale.
    multires = MultiRes(scale, device)
    loss = Loss(linOperator,linOperator.apply(image_tensor), lmbda = lmbda)

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

def plot_results(model,image):
    plt.figure(figsize=(15, 5),dpi = 600)
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(model.sols[-1][0,0,:,:].real.to('cpu'))
    plt.title("Reconstructed Image")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(model.loss.y[0, 0].real.to('cpu'))
    plt.title("Loss")
    plt.colorbar()
    return None 
# %%
image = plt.imread('images/peppers.jpg')[:511, :511] / 255
model = run_test()

# %%
plot_results(model,image)

# %%
