#%%
import torch
from matplotlib import pyplot as plt
from time import time
import numpy as np
from ptycho_v2.tools_v2.ptychography import Ptychography
from ptycho_v2.tools_v2.u_electron_microscopy import initialize_physical_params, get_proj, get_ring_average
from ptycho_v2.tools_v2.u_ptychography import get_overlap_img,get_overlap_area
#%%
# Initialize object
max_scale = 9
max_probe_size = 128
max_shift = 32
device = 'cuda'
#%%
# Initialize forward operators
linop = Ptychography(max_scale = max_scale,max_probe_size = max_probe_size ,max_shift = max_shift,device=device)
image = plt.imread('images/peppers.jpg')[:2**max_scale, :2**max_scale] / 255
image_tensor = torch.tensor(image).double().to(device).view(1, 1, 2**max_scale, 2**max_scale)
image_tensor_ = torch.exp(1j * image_tensor)
image_tensor_ = image_tensor_[:,:,:2**(max_scale-1),:2**(max_scale-1)]
image_tensor_ = image_tensor_[:,:,:2**(max_scale-2),:2**(max_scale-2)]
image_tensor_ = image_tensor_[:,:,:2**(max_scale-3),:2**(max_scale-3)]
image_tensor_ = image_tensor_[:,:,:2**(max_scale-4),:2**(max_scale-4)]
image_tensor_ = image_tensor_[:,:,:2**(max_scale-5),:2**(max_scale-5)]
image_tensor_ = image_tensor_[:,:,:2**(max_scale-6),:2**(max_scale-6)]

#%%

m = linop.apply(image_tensor_)



#%%
# Plot probe
probe = linop.probe.cpu()
plt.figure(figsize=(30,30),dpi = 200)
plt.subplot(2, 2, 1)
plt.imshow(torch.abs(probe))
plt.title("Probe magnitude")
plt.colorbar()
plt.subplot(2, 2, 2)
plt.imshow(torch.angle(probe))
plt.title("Probe phase")

# Plot overlap img
overlap_img = get_overlap_img(linop.probe, linop.shifts, n_dim=2)
#plt.figure(dpi = 600)
plt.subplot(2, 2, 3)
plt.imshow(overlap_img.cpu())
plt.title("Overlapped probes")

overlap_area = get_overlap_area(probe, linop.shifts)
plt.subplot(2, 2, 4)
plt.imshow(overlap_area)
plt.locator_params(axis='x', nbins=30)  # Increase number of x-axis ticks to 20
plt.locator_params(axis='y', nbins=30)
plt.title("Overlapped probes")
#%%