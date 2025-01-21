#%%
from ptycho.multires.class_multiressolver import *
import matplotlib.pyplot as plt
import torch.nn.functional as func
import torch
from ptycho.tools.ptychography import Ptychography2


def run_test():
    # Setting the operating device as cpu and the inital as ...
    
    scale = 9
    I_in = [1, 15, 10, 5, 2, 5, 10, 30, 100]
    I_out = [0, 0, 0, 0, 7, 7, 7, 5, 4]
    cycle = [0, -1, -1, -1, -1,1,  1, 1, 1]
    device = "cpu"
    lmbda = 1e-7
    LR = 1e-1
    tol = [1e-8] * 9
    tol_in = [1e-8] * 9


    linOperator = Ptychography2(in_shape=(2**scale-1, 2**scale-1), n_img=16, probe_type='defocus pupil',
                           probe_radius=200, defocus_factor=0, 
                           fov=520, threshold=0.3, device='cpu')

    image = plt.imread('images/peppers.jpg')[:511, :511] / 255
    image_tensor = torch.tensor(image).double().to(device).view(1, 1, 511, 511)
    image_tensor = torch.exp(1j * image_tensor)
    #Initiate the MultiRes class with the inital scale.
    multires = MultiRes(scale, device)
    loss = Loss(linOperator,linOperator.apply_linop(image_tensor), lmbda = lmbda)

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
model = run_test()
# %%

scale = 9
image = plt.imread('images/peppers.jpg')[:511, :511] / 255
image_tensor = torch.tensor(image).double().to('cpu').view(1, 1, 511, 511)
image_tensor = torch.exp(1j*image_tensor)

linOperator = Ptychography2(in_shape=(2**scale-1, 2**scale-1), n_img=16, probe_type='defocus pupil',
                           probe_radius=25, defocus_factor=0, 
                           fov=120, threshold=0.3, device='cpu')
#b = np.absolute(linOperator.apply_linop((image_tensor))[0,0,:,:])
#plt.imshow(b)
#plt.show()
#a = np.absolute(linOperator.apply_linop((model.sols[-1]))[0,0,:,:])

plt.figure(figsize=(20, 5),dpi = 120)
plt.subplot(1, 3, 1)
plt.imshow(image,cmap='gray')
plt.title(r"(a) Phase of GT $(\angle x)$")
plt.colorbar()
plt.subplot(1, 3, 2)
phase = torch.angle(model.sols[-1])
plt.imshow(phase[0,0,:,:], cmap='gray')
plt.title(r"(a) Phase of Recontruction $(\angle \hat{x})$")
plt.colorbar()
plt.subplot(1, 3, 3)
loss = (image - phase[0,0,:,:].cpu().numpy())**2
plt.imshow(loss,cmap='gray')
plt.title(r"(c) MSE: $(\angle x - \angle \hat{x})^2$")
plt.colorbar()
plt.tight_layout()
plt.savefig('/home/efe/Desktop/Multiresolution-Framework-for-Fourier-Ptychography/n_figs/linear.png')
# %%
