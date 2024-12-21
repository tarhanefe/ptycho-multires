#%%
from ptycho_v2.multires_v2.class_multiressolver import *
import matplotlib.pyplot as plt
import torch
from ptycho_v2.tools_v2.ptychography import Ptychography as Ptychography
from utils import *
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
    LR = 1e-2
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
