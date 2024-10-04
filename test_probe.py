#%%
import torch
from matplotlib import pyplot as plt
from time import time

from ptycho.tools.ptychography import Ptychography2
from ptycho.tools.u_electron_microscopy import initialize_physical_params, get_proj, get_ring_average
from ptycho.tools.fisher_information import compute_fi, hartley_transform_fim
from ptycho.tools.u_ptychography import get_overlap_img

# Initialize object
size = 511
n_img = 4*4
param = initialize_physical_params(shape=size, pix_size=1)
x = get_proj(param).unsqueeze(0).unsqueeze(0)
device = "cpu"

# Initialize forward operators
ptycho_fwd = Ptychography2(in_shape=(size, size), n_img=n_img, probe_type='defocus pupil',
                           probe_radius=150, defocus_factor=0, 
                           fov=520, threshold=0.3, device=device)

# Plot probe
probe = ptycho_fwd.probe
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(torch.abs(probe))
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(torch.angle(probe))
plt.colorbar()
plt.show()

# Plot overlap img
overlap_img = get_overlap_img(ptycho_fwd.probe, ptycho_fwd.shifts, n_dim=2)
plt.figure()
plt.imshow(overlap_img)
plt.colorbar()
plt.show()

# %%
