#%%
import torch
from matplotlib import pyplot as plt
from time import time
import numpy as np
from ptycho.tools.ptychography import Ptychography2_v2 as Ptychography2
from ptycho.tools.u_electron_microscopy import initialize_physical_params, get_proj, get_ring_average
from ptycho.tools.u_ptychography import get_overlap_img,get_overlap_area
#%%
# Initialize object
size = 127
n_img = 10*10
param = initialize_physical_params(shape=size, pix_size=1)
x = get_proj(param).unsqueeze(0).unsqueeze(0)
device = "cpu"
#%%
# Initialize forward operators
ptycho_fwd = Ptychography2(in_shape=(size, size), n_img=n_img, probe_type='square',
                           probe_radius=32, defocus_factor=0, 
                           fov=170, threshold=0.3, device=device)
#%%
# Plot probe
probe = ptycho_fwd.probe
plt.figure(figsize=(10, 5),dpi = 600)
plt.subplot(1, 2, 1)
plt.imshow(torch.abs(probe))
plt.title("Probe magnitude")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(torch.angle(probe))
plt.title("Probe phase")
plt.colorbar()
plt.show()

# Plot overlap img
overlap_img = get_overlap_img(ptycho_fwd.probe, ptycho_fwd.shifts, n_dim=2)
plt.figure(dpi = 600)
plt.imshow(overlap_img)
plt.title("Overlapped probes")
plt.colorbar()
plt.show()

overlap_area = get_overlap_area(ptycho_fwd.probe, ptycho_fwd.shifts)
plt.figure(dpi = 600,figsize=(10,10))
plt.imshow(overlap_area)
plt.locator_params(axis='x', nbins=30)  # Increase number of x-axis ticks to 20
plt.locator_params(axis='y', nbins=30)
plt.title("Overlapped probes")
plt.colorbar()
plt.show()
#%%