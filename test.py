#%%
import torch 
import numpy as np
from ptychography import Ptychography, Ptychography2

n = 128
x = torch.randn(1,1,n,n)
ptycho_fwd = Ptychography2(in_shape=(n, n), n_img=16, probe_type='defocus pupil',
                           probe_radius=25, defocus_factor=0, 
                           fov=120, threshold=0.3, device='cpu')

y = ptycho_fwd.apply_linopT(x)
# %%
#apply()
#apply_linop()
#apply_linopT()

#Ptychography2 class:
#apply_single()

#%%
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(y[0,0].cpu().numpy())

plt.figure()
plt.imshow(y[0,1].cpu().numpy())

plt.figure()
plt.imshow(y[0,2].cpu().numpy())

plt.figure()
plt.imshow(y[0,3].cpu().numpy())
#%%
from experience import *

run_test()
# %%
import torch
n = 128
x = torch.randn(1,1,n,n)
x = torch.stack([x, x], dim=-1)
x = torch.view_as_complex(x).to(torch.complex64)

a = x.abs().sum().item()        
b = torch.view_as_real(x).abs().sum().item()
# %%
